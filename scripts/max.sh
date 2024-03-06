#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --force_rebuild --data_dir "inputs/all-time" --param_file "params/max.yaml" --output_dir "outputs"