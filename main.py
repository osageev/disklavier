import os
import time

# import logging
# import logging.config
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf
from threading import Thread, Event
from rich.prompt import Confirm, IntPrompt

from overseer.overseer import Overseer
from utils import console, tick


p = "[white]main[/white]  :"


def check_tempo(tempo) -> int:
    console.log(f"{p} running at {tempo}bpm")
    console.log(f"{p} how does this sound?")
    stop_event = Event()

    metro_thread = Thread(target=tick, args=(tempo, stop_event))
    metro_thread.start()

    time.sleep(60 / tempo * 4)

    stop_event.set()
    metro_thread.join()

    if Confirm.ask(f"{p} was that bpm ok?", default=True):
        return int(tempo)
    else:
        while True:
            new_tempo = IntPrompt.ask(
                f"{p} enter a new tempo between [b]20[/b] and [b]200[/b] [60]",
                default=60,
            )
            if new_tempo >= 20 and new_tempo <= 200:
                check_tempo(new_tempo)
            console.log(f"{p} [prompt.invalid]tempo must be between 20 and 200")


if __name__ == "__main__":
    # load args and params
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument(
        "--param_file", default=None, help="path to parameter file, in .yaml"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory in which to store outputs (metrics files, logs, recordings)",
    )
    parser.add_argument(
        "--log_config",
        default=None,
        help="where the logging config file is found",
    )
    parser.add_argument(
        "-f",
        "--force_rebuild",
        action="store_true",
        help="whether to rebuild similarity metrics",
    )
    parser.add_argument(
        "-k",
        "--kickstart",
        action="store_true",
        help="dont wait for user input, just start playing",
    )
    parser.add_argument(
        "--tempo",
        type=int,
        help="tempo to record and play at, in bpm",
    )
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    # logging.config.fileConfig(args.log_config)
    # logger = logging.getLogger('main')

    # filesystem setup
    output_dir = "general"  # one output dir only
    # output_dir = f"{datetime.now().strftime('%y-%m-%d')}"   # daily output dirs
    # output_dir = f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}"  # unique output dirs
    log_dir = os.path.join(args.output_dir, output_dir, "logs")
    record_dir = os.path.join(args.output_dir, output_dir, "records")

    if not os.path.exists(args.output_dir):
        console.log(f"{p} creating new outputs folder: '{args.output_dir}'")
        os.mkdir(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, output_dir)):
        console.log(f"{p} creating new outputs folder: '{output_dir}'")
        os.mkdir(os.path.join(args.output_dir, output_dir))
    if not os.path.exists(log_dir):
        console.log(f"{p} creating new logging folder: '{log_dir}'")
        os.mkdir(log_dir)
    if not os.path.exists(record_dir):
        console.log(f"{p} creating new recordings folder: '{record_dir}'")
        os.mkdir(record_dir)
    if os.path.exists("data/playlist"):
        for file in os.listdir("data/playlist"):
            os.remove(os.path.join("data/playlist", file))
    else:
        console.log(f"{p} creating new playlist folder: 'data/playlist'")
        os.mkdir("data/playlist")
    console.log(f"{p} filesystem is set up")

    if args.tempo:
        # playback_tempo = check_tempo(args.tempo)
        playback_tempo = args.tempo

    # run!
    overseer = Overseer(
        params,
        args.data_dir,
        os.path.join(args.output_dir, output_dir),
        record_dir,
        playback_tempo,
        args.force_rebuild,
        args.kickstart,
    )
    overseer.start()

    console.log(f"{p} [green bold]session complete, saving log")
    console.save_text(
        os.path.join(log_dir, f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}.log")
    )
