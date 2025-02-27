#!/bin/bash

srun --ntasks-per-node=4 -p gpu-h100 --nodes=1 --gpus=1 --time=3-00:00:0 --pty bash