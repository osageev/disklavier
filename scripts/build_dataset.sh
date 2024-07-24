#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py\
    --data_dir "data/datasets/test"\
    --dataset_name "test"\
    -rt

python build_tables.py\
    --data_dir "data/datasets/test"\
    --dataset_name "test"\
    --metric "pitch_histogram"\
    -nrst
