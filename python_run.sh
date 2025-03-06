#!/bin/bash

# Define variables
OUTDIR=training-runs
DATA_PATH=/home/luongcn/pet_ddpm/data/data.mat
COND=0
ARCH=ddpmpp
BATCH=4
LR=1e-4
DROPOUT=0.05
AUGMENT=0
REAL_P=0.5
PADDING=1
TICK=2
SNAP=10
PAD_WIDTH=64
IMSIZE=484

# Run the training script with variables
python train.py \
  --outdir="$OUTDIR" \
  --data="$DATA_PATH" \
  --cond="$COND" \
  --arch="$ARCH" \
  --batch="$BATCH" \
  --lr="$LR" \
  --dropout="$DROPOUT" \
  --augment="$AUGMENT" \
  --real_p="$REAL_P" \
  --padding="$PADDING" \
  --tick="$TICK" \
  --snap="$SNAP" \
  --pad_width="$PAD_WIDTH" \
  --imsize="$IMSIZE"