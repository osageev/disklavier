#!/bin/bash
DATASET="maestro"

cd "$(dirname "$0")"/.. || exit
python src/build_dataset.py\
    --data_dir "/media/nova/Datasets/maestro/midi"\
    --out_dir "/media/nova/Datasets/maestro/segments"\
    --dataset_name "$DATASET"\
    -ast
