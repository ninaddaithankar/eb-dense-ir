#!/bin/bash

#SBATCH --account=beex-dtai-gh
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=144

### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

### LOG INFO ###
#SBATCH --job-name=dense_ir
#SBATCH --output=logs/slurm/dense_ir-%A.log

mkdir -p logs/slurm/

python train.py \
    --batch_size 4096 \
    --margin 0.5 \
    --lr 1e-4 \
    --max_steps 12000 \
    --shared_encoder \
    --encoder "microsoft/deberta-v3-base" \
    --devices -1 \
    --num_workers 16 \
    --strategy "ddp" \
