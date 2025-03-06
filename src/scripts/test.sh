#!/bin/bash
TEST_PATH="/home/luongcn/pet_ddpm/raw_data/HRRT_FDG_JB737"
DEST_PATH="/home/luongcn/pet_ddpm/data/image_dir"
SCRIPT_LOC=/home/luongcn/pet_ddpm/src/get_test_img.py

python $SCRIPT_LOC --test_path=$TEST_PATH --dest_path=$DEST_PATH