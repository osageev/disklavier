#!/bin/bash
DATASET_NAME="20250110"
cd "$(dirname "$0")"/.. || exit
python src/main.py\
    --dataset "data/datasets/$DATASET_NAME/augmented"\
    --params "params/disklavier.yaml"\
    --output "data/outputs/logs"\
    --tables "data/tables/$DATASET_NAME"\
    --bpm "$1"