#!/bin/bash

# Define variables
NUM_PROC=4
OUTDIR="training-runs"
DATA_PATH="/home/luongcn/pet_ddpm/data/data.mat"
COND=0
ARCH="ddpmpp"
BATCH_SIZE=16
LEARNING_RATE=1e-4
DROPOUT=0.05
AUGMENT=0
REAL_P=0.5
PADDING=1
TICK=2
SNAP=10
PAD_WIDTH=64

# Run the training command
torchrun --standalone --nproc_per_node=$NUM_PROC train.py \
    --outdir=$OUTDIR \
    --data="$DATA_PATH" \
    --cond=$COND \
    --arch=$ARCH \
    --batch=$BATCH_SIZE \
    --lr=$LEARNING_RATE \
    --dropout=$DROPOUT \
    --augment=$AUGMENT \
    --real_p=$REAL_P \
    --padding=$PADDING \
    --tick=$TICK \
    --snap=$SNAP \
    --pad_width=$PAD_WIDTH