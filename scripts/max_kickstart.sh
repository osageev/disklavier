#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --data_dir "data/datasets/test"\
    --param_file "params/max.yaml"\
    --output_dir "data/outputs"\
    --kickstart "data/datasets/test"\
    -e 4\
    --tempo $1