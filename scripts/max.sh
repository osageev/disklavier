#!/bin/bash
DATASET_NAME="test"
cd "$(dirname "$0")"/..
python src/main.py\
    --dataset "data/datasets/$DATASET_NAME/play"\
    --params "params/max_$DATASET_NAME.yaml"\
    --output "data/outputs"\
    --tables "data/tables/$DATASET_NAME"\
	--kickstart "20231220-080-01_0000-0005.mid"\
    --bpm $1