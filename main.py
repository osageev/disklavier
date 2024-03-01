import os
import mido
import logging
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf

from playback import Player
from similarity import Similarity

if __name__=="__main__":
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "--data_dir", default=None, help="location of MIDI files"
    )
    parser.add_argument(
        "--param_file", default=None, help="path to parameter file, in .yaml"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory in which to store outputs (metrics files, logs, recordings)",
    )
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    print(f"running with arguments:\n{args}")
    print(f"running with parameters:\n{params}")

    output_dir = f"{datetime.now().strftime('%y-%m-%d')}"
    # output_dir = f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}"
    log_dir = os.path.join(args.output_dir, output_dir, "logs")
    record_dir = os.path.join(args.output_dir, output_dir, "records")
    
    if not os.path.exists(os.path.join(args.output_dir, output_dir)):
        print(f"creating new outputs folder: '{output_dir}'")
        os.mkdir(os.path.join(args.output_dir, output_dir))   
    if not os.path.exists(log_dir):
        print(f"creating new logging folder: '{log_dir}'")
        os.mkdir(log_dir)
    if not os.path.exists(record_dir):
        print(f"creating new recordings folder: '{record_dir}'")
        os.mkdir(record_dir)

    logFormatter = logging.Formatter("%(asctime)s [%(name)-6.6s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger("main")

    log_name = f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}.log"
    fileHandler = logging.FileHandler("{0}/{1}".format(log_dir, log_name))
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    similarity = Similarity(args.data_dir, os.path.join(args.output_dir, output_dir), params.similarity)

    player = Player(similarity, record_dir, params)
    player.start_recording()
    print("INITIALIZATION DONE")
