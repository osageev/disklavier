#!/bin/bash
pwd
cd "$(dirname "$0")"/..
python main.py\
    --dataset "test"\
    --data_dir "data/datasets/test/play"\
    --param_file "params/disklavier.yaml"\
    --output_dir $1\
    -k $2\
	--tempo $3\
    -n $4\
    -p