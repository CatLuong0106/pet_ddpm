#!/bin/bash

HRRT_PATH="/home/luongcn/pet_ddpm/raw_data/HRRT_NX_pair/HRRT_DH485"
NX_PATH="/home/luongcn/pet_ddpm/raw_data/HRRT_NX_pair/NX_DH485"
TRAIN_COND_DATA_PATH="/home/luongcn/pet_ddpm/data/train_conditional"
SCRIPT_LOC=/home/luongcn/pet_ddpm/src/conditional_sampling.py

python $SCRIPT_LOC --path_x=$NX_PATH --path_x_prior=$NX_PATH --output_path=$TRAIN_COND_DATA_PATH