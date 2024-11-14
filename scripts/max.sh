#!/bin/bash
DATASET_NAME="20240621"
cd "$(dirname "$0")"/.. || exit
python src/main.py\
    --dataset "data/datasets/$DATASET_NAME/train"\
    --params "params/max.yaml"\
    --output "data/outputs"\
    --tables "data/tables/$DATASET_NAME"\
    --verbose --bpm "$1"