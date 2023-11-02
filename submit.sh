#!/bin/bash
#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -J gpt2-v7

mkdir -p logs

# https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html

# Debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

srun python train.py training.batch_size=200 training.block_size=20 compile=True max_epochs=100
