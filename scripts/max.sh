#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/test" --param_file "params/max.yaml" --output_dir "outputs" --tempo 90 -k