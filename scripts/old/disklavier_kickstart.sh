#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --dataset "test"\
    --data_dir "data/datasets/test/dataset samples"\
    --param_file "params/disklavier_test.yaml"\
    --output_dir "data/outputs"\
    -i -n 6 --tempo $1