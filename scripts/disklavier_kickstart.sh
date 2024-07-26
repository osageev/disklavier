#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --dataset "test"\
    --data_dir "data/datasets/test/play"\
    --param_file "params/disklavier_test.yaml"\
    --output_dir "data/outputs"\
    -i -n 4 -e 8 --tempo $1