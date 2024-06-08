#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --data_dir "data/datasets/careful"\
    --param_file "params/disklavier.yaml"\
    --output_dir "data/outputs"\
    -e 8\
    --tempo $1