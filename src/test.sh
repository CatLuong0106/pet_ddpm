#!/bin/bash
TEST_PATH="/home/luongcn/pet_ddpm/raw_data/HRRT_FDG_JB737"
DEST_PATH="/home/luongcn/pet_ddpm/data/image_dir"

python get_test_img.py --test_path=$TEST_PATH --dest_path=$DEST_PATH