#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --dataset "20240621"\
    --data_dir "data/datasets/20240621/play"\
    --param_file "params/disklavier.yaml"\
    --output_dir "data/outputs"\
    -i -s -e 8 --tempo $1