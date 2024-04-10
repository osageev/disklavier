#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "data/datasets/careful" --param_file "params/multi_player.yaml" --output_dir "data/outputs" --tempo $1