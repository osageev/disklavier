#!/bin/bash
DATASET_NAME="test"
cd "$(dirname "$0")"/..
# python build_dataset.py\
#     --data_dir "data/datasets/$DATASET_NAME"\
#     --dataset_name "$DATASET_NAME"\
#     -rt

python build_tables.py\
    --data_dir "data/datasets/$DATASET_NAME"\
    --dataset_name "$DATASET_NAME"\
    --metric "full_pitch_histogram"\
    -nrst
