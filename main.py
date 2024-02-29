import os
import mido
import logging
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf

from playback import Player

if __name__=="__main__":
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "--data_dir", default="input_data", help="location of MIDI files"
    )
    parser.add_argument(
        "--param_file", default=None, help="path to parameter file, in .yaml"
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="directory in which to store logs",
    )
    parser.add_argument(
        "--record_dir",
        default="recordings",
        help="directory in which to store recordings",
    )
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    print(f"running with arguments:\n{args}")
    print(f"running with parameters:\n{params}")

    if not os.path.exists(args.log_dir):
        print(f"creating new logging folder: '{args.log_dir}'")
        os.mkdir(args.log_dir)
    if not os.path.exists(args.record_dir):
        print(f"creating new recording folder: '{args.record_dir}'")
        os.mkdir(args.record_dir)

    logFormatter = logging.Formatter("%(asctime)s [%(name)-6.6s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger("main")

    log_name = f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}.log"
    fileHandler = logging.FileHandler("{0}/{1}".format(args.log_dir, log_name))
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    player = Player(params)
    player.start_recording()
    print("DONE")
