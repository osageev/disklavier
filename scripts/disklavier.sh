#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/all-time" --param_file "params/disklavier.yaml" --output_dir "outputs" --log_config "logging.conf"