#!/bin/bash

srun --ntasks-per-node=4 -p gpu-a100 --nodes=2 --gpus=4 --time=3-00:00:0 --pty bash