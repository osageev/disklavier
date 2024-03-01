#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/beats" --param_file "params/disklavier.yaml" --output_dir "outputs"