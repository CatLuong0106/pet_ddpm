#!/bin/bash

python train.py --outdir=training-runs --data=/home/luongcn/pet_ddpm/data/data.mat --cond=0 --arch=ddpmpp --batch=4 --lr=1e-4 --dropout=0.05 --augment=0 --real_p=0.5 --padding=1 --tick=2 --snap=10 --pad_width=64