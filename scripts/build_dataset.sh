#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py\
    --data_dir "data/datasets/20240606"\
    --build_train \
    -t