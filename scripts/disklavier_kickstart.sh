#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --data_dir "data/datasets/20240606/play"\
    --param_file "params/disklavier.yaml"\
    --output_dir "data/outputs"\
    --kickstart "RAND"\
    -p \
    --tempo $1