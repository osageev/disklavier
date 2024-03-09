#!/bin/bash
cd "$(dirname "$0")"/..
python dataset/dataset.py --data_dir "inputs/test datasets" --output_dir "inputs/test" -t