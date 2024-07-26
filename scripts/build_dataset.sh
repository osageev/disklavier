#!/bin/bash
DATASET="20240621"

DATASET_NAME="test"
cd "$(dirname "$0")"/..
python build_dataset.py\
    --data_dir "data/datasets/$DATASET"\
    --dataset_name "$DATASET"\
    -rt

python build_tables.py\
    --data_dir "data/datasets/$DATASET"\
    --dataset_name "$DATASET"\
    --metric "pitch_histogram"\
    -nrst
