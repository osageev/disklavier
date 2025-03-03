#!/bin/bash
DATASET_NAME="20250110"
cd "$(dirname "$0")"/.. || exit
python src/main.py\
    --dataset "data/datasets/$DATASET_NAME/augmented"\
    --params "params/max.yaml"\
    --tables "data/tables/$DATASET_NAME"\
    --bpm "$1"