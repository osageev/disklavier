import os
import csv
import time
import pygame
from datetime import datetime, timedelta
from argparse import ArgumentParser
from omegaconf import OmegaConf
from queue import PriorityQueue
import mido
from threading import Thread
from multiprocessing import Process
from workers import Metronome, Player, Scheduler, Seeker
from utils import console


tag = "[white]main[/white]  :"


def main(args, params):
    td_start = datetime.now() + timedelta(seconds=1)
    ts_start = td_start.strftime("%y%m%d-%H%M%S")

    # filesystem setup
    p_log = os.path.join(args.output, "logs", f"{ts_start}")
    pf_recording = os.path.join(p_log, f"recording_{ts_start}.mid")
    p_playlist = os.path.join(p_log, "playlist")
    pf_playlist = os.path.join(p_log, f"playlist_{ts_start}.csv")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(p_log):
        console.log(f"{tag} creating new logging folder at '{p_log}'")
        os.makedirs(p_log)
        os.makedirs(p_playlist)  # folder for copy of MIDI files
    with open(pf_playlist, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "timestamp", "filepath"])
    console.log(f"{tag} filesystem set up complete")

    # worker setup
    scheduler = Scheduler(
        params.scheduler, args.bpm, p_log, pf_recording, p_playlist, td_start
    )
    seeker = Seeker(params.seeker, args.tables, args.dataset)
    player = Player(params.player, args.bpm, td_start)

    # data setup
    pf_seed = None
    match params.initialization:
        case "recording":  # collect user recording
            # TODO: get recording if no seed file is specified
            # seeker.played_files.append(recording_path)
            raise NotImplementedError
        case "kickstart":  # use specified file as seed
            try:
                if params.kickstart_path:
                    pf_seed = os.path.join(args.dataset, params.kickstart_path)
                    console.log(f"{tag} [cyan]KICKSTART[/cyan] - '{pf_seed}'")
                    seeker.played_files.append(pf_seed)
            except AttributeError:
                console.log(f"{tag} no file specified to kickstart from")
        case "random" | _:  # choose random file from library
            pf_seed = seeker.get_random()
            console.log(f"{tag} [cyan]RANDOM INIT[/cyan] - '{pf_seed}'")

    if init_outfile(pf_recording, scheduler, args.bpm):
        console.log(f"{tag} successfully initialized recording")
    else:
        console.log(f"{tag} [red]error initializing recording, exiting")
        raise FileExistsError("Couldn't initialize MIDI recording file")

    # run
    q_playback = PriorityQueue()

    dt_run = timedelta(seconds=40)  # try to keep as a multiple of the segment length
    td_start = datetime.now()
    td_now = datetime.now()
    ts_queue = 0
    n_files = 0
    try:
        scheduler.td_start = td_start
        ts_queue += scheduler.add_midi_to_queue(pf_seed, q_playback)  # type: ignore
        with open(pf_playlist, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    n_files,
                    td_now.strftime("%y-%m-%d %H:%M:%S"),
                    pf_seed,
                ]
            )

        # start playback
        player.td_start = td_start
        player.td_last_note = td_start
        thread_player = Thread(target=player.play, name="player", args=(q_playback,))
        thread_player.start()
        metronome = Metronome(params.metronome, args.bpm, td_start)
        process_metronome = Process(target=metronome.tick, name="metronome")
        process_metronome.start()
        while td_now - td_start < dt_run:
            td_now = datetime.now()
            # check whether more segments need to be added to the queue
            # start loading files into queue
            if ts_queue < params.ts_min_queue_length:
                pf_next_file = seeker.get_next()
                ts_queue += scheduler.add_midi_to_queue(pf_next_file, q_playback)
                console.log(f"{tag} queue time is now {ts_queue:.01f} seconds")
                with open(pf_playlist, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            n_files,
                            td_now.strftime("%y-%m-%d %H:%M:%S"),
                            pf_next_file,
                        ]
                    )
                n_files += 1

            time.sleep(0.01)

        # dump queue to stop player
        while q_playback.qsize() > 0:
            try:
                _ = q_playback.get()
            except:
                console.log(f"{tag} [yellow]ouch!")
                pass
        thread_player.join(timeout=0.1)
    except KeyboardInterrupt:
        console.log(f"{tag}[yellow] CTRL + C detected, saving and exiting...")
        # dump queue to stop player
        while q_playback.qsize() > 0:
            try:
                _ = q_playback.get()
            except:
                console.log(f"{tag} [yellow]ouch!")
                pass
        thread_player.join(timeout=0.1)

    console.log(f"{tag} stopping metronome")
    process_metronome.terminate()
    process_metronome.join(timeout=0.1)
    if process_metronome.is_alive():
        console.log(
            f"{tag}[yellow] metronome process did not terminate, forcefully killing..."
        )
        process_metronome.kill()
        process_metronome.join(timeout=0.5)
    pygame.mixer.quit()

    # run complete, save and exit
    console.save_text(os.path.join(p_log, f"{ts_start}.log"))
    console.log(f"{tag}[green bold] session complete, exiting")


def init_outfile(pf_midi: str, scheduler: Scheduler, bpm: int = 60) -> bool:
    midi = mido.MidiFile()
    tick_track = mido.MidiTrack()

    # default timing messages
    tick_track.append(
        mido.MetaMessage("track_name", name=os.path.basename(pf_midi), time=0)
    )
    tick_track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            clocks_per_click=36,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )
    tick_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    tick_track.append(mido.MetaMessage("end_of_track", time=1))

    # transition messages
    mm_transitions = scheduler.gen_transitions(n_stamps=10, do_ticks=True)
    for mm_transition in mm_transitions:
        tick_track.append(mm_transition)

    midi.tracks.append(tick_track)

    # write to file
    midi.save(pf_midi)

    return os.path.isfile(pf_midi)


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
        "--random_init",
        action="store_true",
        help="dont wait for user input, just start playing from a random selection within the provided dataset.",
    )
    parser.add_argument(
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
        "--num_transitions",
        type=int,
        help="number of transitions to run for",
    )
    parser.add_argument(
        "--kickstart",
        type=str,
        help="use provided midi file as prompt",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="sequential playback mode. always choose next neighbor (if available)",
    )
    args = parser.parse_args()
    params = OmegaConf.load(args.params)

    # copy these so that they only have to be specified once
    params.scheduler.n_beats_per_segment = params.n_beats_per_segment
    params.metronome.n_beats_per_segment = params.n_beats_per_segment

    console.log(f"{tag} loading with args:\n\t{args}")
    console.log(f"{tag} loading with params:\n\t{params}")

    if not os.path.exists(args.tables):
        console.log("[red bold]ERROR[/red bold]: table directory not found, exiting...")
        exit()  # TODO: handle this better

    main(args, params)
