import os
import time
from datetime import datetime, timedelta
from argparse import ArgumentParser
from omegaconf import OmegaConf
from queue import PriorityQueue
import mido

from workers import Scheduler, Seeker
from utils import console
from utils.midi import MidiEvent


tag = "[white]main[/white]  :"


def main(args, params):
    s_start = datetime.now().strftime("%y%m%d-%H%M%S")

    # filesystem setup
    log_path = os.path.join(args.output, "logs", f"{s_start}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(log_path):
        console.log(f"{tag} creating new logging folder at '{log_path}'")
        os.makedirs(log_path)
    console.log(f"{tag} filesystem set up complete")

    # worker setup
    scheduler = Scheduler(params.scheduler, args.bpm, log_path)
    seeker = Seeker(params.seeker, args.tables, args.dataset)
    # data setup
    ## random init check
    seed_path = None
    try:
        if args.random_init:
            seed_path = seeker.get_random()
            console.log(f"{tag} [cyan]RANDOM INIT[/cyan] - '{seed_path}'")
    except AttributeError:
        console.log(f"{tag} not randomly initializing")
    ## kickstart check
    try:
        if args.kickstart:
            seed_path = os.path.join(args.dataset, args.kickstart)
            console.log(f"{tag} [cyan]KICKSTART[/cyan] - '{seed_path}'")
    except AttributeError:
        console.log(f"{tag} not kickstarting")
    ## init recording file
    recording_path = os.path.join(log_path, f"recording_{datetime.now().strftime("%y%m%d-%H%M")}.mid")
    if init_outfile(recording_path, scheduler, args.bpm):
        console.log(f"{tag} successfully initialized recording")
        mido.MidiFile(recording_path).print_tracks()
    else:
        console.log(f"{tag} [red]error initializing recording, exiting")
        return 1 # TODO: handle this better

    if seed_path is None:
        # TODO: get recording if no seed file is specified
        pass
    
    # run
    q_playback = PriorityQueue()

    t_run = timedelta(seconds=32) # try to keep as a multiple of the segment length
    t_start = datetime.now()
    t_now = datetime.now()
    t_queue = 0
    try:
        while t_now - t_start < t_run:
            t_now = datetime.now()
            # check whether more segments need to be added to the queue
            # start loading files into queue
            if t_queue < params.min_queue_length:
                next_file_path = seeker.get_random()
                t_queue += scheduler.add_midi_to_queue(next_file_path, q_playback)
                console.log(f"{tag} queue time is now {t_queue}")

                return
    except KeyboardInterrupt:
        console.log(f"{tag} [orange]CTRL + C detected, saving and exiting...")

    # run complete, save and exit
    console.save_text(os.path.join(log_path, f"{s_start}.log"))
    console.log(f"{tag}[green bold] session complete, exiting")


def send_midi_from_queue(midi_queue: PriorityQueue, midi_out_port, stop_event):
    start_time = time.time()

    while not stop_event.is_set():
        item = midi_queue.get()

        if item is None:  # check for the stop signal
            break

        absolute_time, msg = item
        sleep_time = absolute_time - (time.time() - start_time)

        if sleep_time > 0:
            time.sleep(sleep_time)

        print(f"sending: {msg}")
        midi_queue.task_done()


def init_outfile(file_path: str, scheduler: Scheduler, bpm: int = 60) -> bool:
    midi = mido.MidiFile()
    tick_track = mido.MidiTrack()

    # regular messages
    tick_track.append(
        mido.MetaMessage("track_name", name=os.path.basename(file_path), time=0)
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
    tick_track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)
    )
    tick_track.append(mido.MetaMessage("end_of_track", time=1))

    t_transitions = scheduler.gen_transitions_cgpt(10, do_ticks=True)
    for transition_msg in t_transitions:
        tick_track.append(transition_msg)

    midi.tracks.append(tick_track)

    # write to file
    midi.save(file_path)

    return os.path.isfile(file_path)


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
        "-n",
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

    console.log(f"{tag} loading with args:\n\t{args}")
    console.log(f"{tag} loading with params:\n\t{params}")

    if not os.path.exists(args.tables):
        console.log("[red bold]ERROR[/red bold]: table directory not found, exiting...")
        exit() # TODO: handle this better

    main(args, params)
