#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --data_dir "data/datasets/careful"\
    --param_file "params/disklavier.yaml"\
    --output_dir "data/outputs"\
    --kickstart "data/datasets/careful/20231220-80-03_0000-0008.mid"\
    -p \
    --tempo $1