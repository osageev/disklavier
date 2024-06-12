#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py\
    --data_dir "data/datasets/tmp"\
    --build_train \
    -t