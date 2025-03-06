# ARC Setup
- `module load anaconda/3.0`
- `module load cuda`
- `eval "$(ssh-agent -s)"` and then `ssh-add {your-key}`, and `ssh -T git@github.com` to establish connection to GitHub repo
- 

# Pipeline
- Cleaning up slices with little to no information --> Only picking the "representative" slices, that means slices with distinct information. --> Process the image in training.

# Code notes: (Fill this out later)
- When selecting the PAD size for testing, PAD value must be `(k + 1)P - N` where `k = floor(N/P)` and `P` is the patchsize and `N` is the image size