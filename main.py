import os
import logging
import logging.config
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf

from overseer.overseer import Overseer
from utils import console


if __name__=="__main__":
    # load args and params
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
    parser.add_argument(
        "--log_config",
        default=None,
        help="where the logging config file is found",
    )
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    logging.config.fileConfig(args.log_config)
    logger = logging.getLogger('main')
    p = '[white]main[/white]  : '

    # filesystem setup
    output_dir = f"{datetime.now().strftime('%y-%m-%d')}"   # daily output dirs
    # output_dir = f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}"  # unique output dirs
    log_dir = os.path.join(args.output_dir, output_dir, "logs")
    record_dir = os.path.join(args.output_dir, output_dir, "records")
    
    if not os.path.exists(os.path.join(args.output_dir, output_dir)):
        console.log(f"{p}creating new outputs folder: '{output_dir}'")
        os.mkdir(os.path.join(args.output_dir, output_dir))   
    if not os.path.exists(log_dir):
        console.log(f"{p}creating new logging folder: '{log_dir}'")
        os.mkdir(log_dir)
    if not os.path.exists(record_dir):
        console.log(f"{p}creating new recordings folder: '{record_dir}'")
        os.mkdir(record_dir)
    console.log(f"{p}filesystem is set up")

    # run!
    overseer = Overseer(params, args.data_dir, os.path.join(args.output_dir, output_dir), record_dir)
    overseer.start()

    console.log(f"{p}[green bold]session complete, saving log")
    console.save_text(os.path.join(log_dir, f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}.log"))
