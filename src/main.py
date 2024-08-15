import os
import random
from datetime import datetime
from omegaconf import OmegaConf
from argparse import ArgumentParser
import threading
from queue import Queue

from loader import Loader
from scheduler import PlaybackScheduler
from player import Player
from utils import console

P = "[white]main[/white]  :"


def main(args, params):
    output_dir = filesystem_setup(params.filesystem)

    loader = Loader(params.filesystem.dataset_path, args.tempo)
    console.log(f"{P} loader initialized")
    loader.load_midi_files()

    queue_lock = threading.Lock()
    cmd_q = Queue()
    midi_q = []

    params.scheduler.duration_t = loader.duration_t
    params.scheduler.ticks_per_beat = loader.ticks_per_beat
    scheduler = PlaybackScheduler(params.scheduler, args.tempo, params.midi.out_port)

    params.player.ticks_per_beat = loader.ticks_per_beat
    player = Player(params.player, params.midi.out_port, queue_lock, cmd_q)

    random.seed(params.seed)
    if params.playback.start_mode == "random":
        seed_file = random.choice(loader.midi_files)

    scheduler.start_scheduling(seed_file)


def filesystem_setup(params) -> str:
    if not os.path.exists(params.table_path):
        console.log("[red bold]table dir not found, exiting...")
        raise NotADirectoryError

    now = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = os.path.join(params.output_path, now)
    plot_dir = os.path.join(output_dir, "plots")
    playlist_dir = os.path.join(output_dir, "playlist")

    if not os.path.exists(output_dir):
        console.log(f"{P} creating new output folder: '{output_dir}'")
        os.makedirs(output_dir)
    if not os.path.exists(plot_dir):
        console.log(f"{P} creating new plots folder: '{plot_dir}'")
        os.makedirs(plot_dir)
    if not os.path.exists(playlist_dir):
        console.log(f"{P} creating new playlist folder: '{playlist_dir}'")
        os.makedirs(playlist_dir)

    console.log(f"{P}[green bold] filesystem set up complete")

    return output_dir


if __name__ == "__main__":
    # load args and params
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "--param_file", default=None, type=str, help="path to param file, in .yaml"
    )
    parser.add_argument(
        "--tempo",
        default=80,
        type=int,
        help="tempo to record and play at, in bpm",
    )
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    console.log(f"{P} loading with arguments:\n\t{args}")
    console.log(f"{P} loading with parameters:\n\t{params}")

    main(args, params)
