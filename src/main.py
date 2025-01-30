import os
import csv
import time
import json

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
from rich.table import Table
from threading import Thread
from omegaconf import OmegaConf
from queue import PriorityQueue
from argparse import ArgumentParser
from multiprocessing import Process
from datetime import datetime, timedelta

import workers
from utils import console


tag = "[white]main[/white]  :"


def main(args, params):
    td_start = datetime.now()
    ts_start = td_start.strftime("%y%m%d-%H%M%S")

    # filesystem setup
    p_log = os.path.join(
        args.output,
        "logs",
        f"{ts_start}_{params.seeker.metric}_{params.seeker.seed}_{params.initialization}",
    )
    p_playlist = os.path.join(p_log, "playlist")
    pf_playlist = os.path.join(p_log, f"playlist_{ts_start}.csv")
    pf_master_recording = os.path.join(p_log, f"master-recording_{ts_start}.mid")
    pf_player_recording = os.path.join(p_log, f"player-recording_{ts_start}.mid")
    params.seeker.pf_recording = pf_player_recording

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(p_log):
        if args.verbose:
            console.log(f"{tag} creating new logging folder at '{p_log}'")
        os.makedirs(p_log)
        os.makedirs(p_playlist)  # folder for copy of MIDI files
    write_log(pf_playlist, "position", "start time", "file path", "similarity")
    console.log(f"{tag} filesystem set up complete")

    # worker setup
    scheduler = workers.Scheduler(
        params.scheduler,
        args.bpm,
        p_log,
        pf_master_recording,
        p_playlist,
        td_start,
        params.initialization == "recording",
    )
    seeker = workers.Seeker(
        params.seeker,
        args.tables,
        args.dataset,
        p_playlist,
        args.bpm,
    )
    player = workers.Player(params.player, args.bpm, td_start)
    recorder = workers.Recorder(
        params.recorder,
        args.bpm,
        pf_player_recording,
    )
    # data setup
    match params.initialization:
        case "recording":  # collect user recording
            ts_recording_len = recorder.run()
            pf_seed = pf_player_recording
        case "kickstart":  # use specified file as seed
            try:
                if params.kickstart_path:
                    pf_seed = os.path.join(args.dataset, params.kickstart_path)
                    console.log(f"{tag} [cyan]KICKSTART[/cyan] - '{pf_seed}'")
            except AttributeError:
                console.log(
                    f"{tag} no file specified to kickstart from, choosing randomly"
                )
                pf_seed = seeker.get_random()
                console.log(f"{tag} [cyan]RANDOM INIT[/cyan] - '{pf_seed}'")
        case "random" | _:  # choose random file from library
            pf_seed = seeker.get_random()
            console.log(f"{tag} [cyan]RANDOM INIT[/cyan] - '{pf_seed}'")

    if params.seeker.mode == "playlist":
        with open("data/matches.json", "r") as f:
            seeker.playlist = json.load(f)
        console.log(f"{tag} loaded playlist from 'data/matches.json'")
        for i, (q, ms) in enumerate(seeker.playlist.items()):
            if i == seeker.matches_pos:
                pf_seed = os.path.join(seeker.p_dataset, f"{q}.mid")
                console.log(f"{tag} added '{pf_seed}' to playlist from matches file")
                continue

    seeker.played_files.append(os.path.basename(pf_seed))

    # run
    q_playback = PriorityQueue()
    td_start = datetime.now() + timedelta(seconds=1)
    ts_queue = 0
    n_files = 1
    # offset by recording length if necessary
    scheduler.td_start = td_start
    if scheduler.init_outfile(
        pf_master_recording,
        ts_recording_len if params.initialization == "recording" else 0,
    ):
        console.log(f"{tag} successfully initialized recording")
    else:
        console.log(f"{tag} [red]error initializing recording, exiting")
        raise FileExistsError("Couldn't initialize MIDI recording file")
    try:
        ts_queue += scheduler.enqueue_midi(pf_seed, q_playback)  # type: ignore
        write_log(
            pf_playlist,
            n_files,
            datetime.now().strftime("%y-%m-%d %H:%M:%S"),
            pf_seed,
            0,
        )

        # start playback
        player.td_start = td_start
        player.td_last_note = td_start
        thread_player = Thread(target=player.play, name="player", args=(q_playback,))
        thread_player.start()
        metronome = workers.Metronome(params.metronome, args.bpm, td_start)
        process_metronome = Process(target=metronome.tick, name="metronome")
        process_metronome.start()

        # play for set number of transitions
        while n_files < params.n_transitions:
            if q_playback.qsize() < params.n_min_queue_length:
                pf_next_file, similarity = seeker.get_next()
                ts_queue += scheduler.enqueue_midi(pf_next_file, q_playback)
                console.log(f"{tag} queue time is now {ts_queue:.01f} seconds")
                n_files += 1
                write_log(
                    pf_playlist,
                    n_files,
                    datetime.now().strftime("%y-%m-%d %H:%M:%S"),
                    pf_next_file,
                    similarity,
                )

            time.sleep(0.1)
            ts_queue -= 0.1
            if not thread_player.is_alive():
                console.log(f"{tag} player ran out of notes, exiting")
                thread_player.join(0.1)
                break

        metronome.stop()

        # all necessary files queued, wait for playback to finish
        console.log(f"{tag} waiting for playback to finish...")
        while q_playback.qsize() > 0:
            time.sleep(0.1)
        thread_player.join(timeout=0.1)
    except KeyboardInterrupt:
        console.log(f"{tag}[yellow] CTRL + C detected, saving and exiting...")
        # dump queue to stop player
        while q_playback.qsize() > 0:
            try:
                _ = q_playback.get()
            except:
                if args.verbose:
                    console.log(f"{tag} [yellow]ouch!")
                pass
        thread_player.join(timeout=0.1)

    if args.verbose:
        console.log(f"{tag} stopping metronome")
    process_metronome.kill()
    process_metronome.join(timeout=0.5)
    pygame.mixer.quit()

    # print playlist
    table = Table(title="PLAYLIST")
    with open(pf_playlist, mode="r") as file:
        headers = file.readline().strip().split(",")  # Read the header line
        for header in headers:
            table.add_column(header)  # Add columns to the table

        for line in file:
            row = line.strip().split(",")  # Read each subsequent line
            table.add_row(*row)  # Add the row to the table
    console.print(table)

    # run complete, save and exit
    console.save_text(os.path.join(p_log, f"{ts_start}.log"))
    console.log(f"{tag}[green bold] session complete, exiting")


def write_log(filename: str, *args):
    """Write the provided arguments as a row to the specified CSV file."""
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(args)


if __name__ == "__main__":
    # load arguments and parameters
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--dataset", type=str, default=None, help="path to MIDI files")
    parser.add_argument(
        "--params", type=str, default=None, help="path to parameter file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="directory in which to store outputs (logs, recordings, etc...)",
    )
    parser.add_argument(
        "--tables",
        type=str,
        default=None,
        help="directory in which precomputed tables are stored",
    )
    parser.add_argument(
        "-t",
        "--tick",
        action="store_true",
        help="play metronome during playback",
    )
    parser.add_argument(
        "--bpm",
        type=int,
        help="bpm to record and play at, in bpm",
    )
    parser.add_argument(
        "--kickstart",
        type=str,
        help="use provided midi file as prompt",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose output",
    )
    args = parser.parse_args()
    params = OmegaConf.load(args.params)

    # copy these so that they only have to be specified once
    params.scheduler.n_beats_per_segment = params.n_beats_per_segment
    params.metronome.n_beats_per_segment = params.n_beats_per_segment

    console.log(f"{tag} loading with arguments:\n\t{args}")
    console.log(f"{tag} loading with parameters:\n\t{params}")

    if not os.path.exists(args.tables):
        console.log(f"{tag} [red bold]ERROR[/red bold]: table directory not found, exiting...")
        exit()  # TODO: handle this better

    main(args, params)
