#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py\
    --data_dir "data/datasets/20240621"\
    --dataset_name "20240621"\
    --build_train \
    -t -r

python build_tables.py\
    --data_dir "data/datasets/20240621"\
    --metric "pitch_histogram"\
