#!/bin/bash
cd "$(dirname "$0")"/..
python dataset/dataset.py --data_dir "inputs/trimmed midi" --output_dir "inputs/no tempo" -t