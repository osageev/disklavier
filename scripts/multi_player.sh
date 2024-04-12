#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "data/datasets/careful" --param_file "params/multi_player.yaml" --output_dir "data/outputs" --kickstart "data/datasets/test datasets/octaveC-70-01.mid" --tempo $1