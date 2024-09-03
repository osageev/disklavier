#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --dataset "test_micro"\
    --data_dir "data/datasets/test_micro"\
    --param_file "params/max.yaml"\
    --output_dir "data/outputs"\
    -i -n 6 --tempo $1