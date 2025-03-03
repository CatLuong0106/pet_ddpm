#!/bin/bash

# Define variables
NETWORK="training-runs/00013-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32/network-snapshot-000400.pkl"
OUTDIR="results"
IMAGE_DIR="data/image_dir"
IMAGE_SIZE=512 # has to be 8 * psize
VIEWS=20
NAME="denoise"
STEPS=100
SIGMA_MIN=0.003
SIGMA_MAX=10
ZETA=0.3
PAD=64
PSIZE=64
# PSIZE=56

# Run the Python script with specified arguments
python inverse_nodist.py \
    --network="$NETWORK" \
    --outdir="$OUTDIR" \
    --image_dir="$IMAGE_DIR" \
    --image_size="$IMAGE_SIZE" \
    --views="$VIEWS" \
    --name="$NAME" \
    --steps="$STEPS" \
    --sigma_min="$SIGMA_MIN" \
    --sigma_max="$SIGMA_MAX" \
    --zeta="$ZETA" \
    --pad="$PAD" \
    --psize="$PSIZE"