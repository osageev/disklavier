#!/bin/bash
DATASET_NAME="train"
cd "$(dirname "$0")"/.. || exit
python src/main.py\
    --dataset "data/datasets/$DATASET_NAME/synthetic"\
    --params "params/disklavier.yaml"\
    --output "data/outputs"\
    --tables "data/tables/$DATASET_NAME"\
    --bpm "$1"