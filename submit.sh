#!/bin/bash
#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH -L scratch,cfs
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@120
#SBATCH --requeue
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -J gpt2-v2


mkdir -p logs

# https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

#srun python train.py training.batch_size=200 training.block_size=20 compile=True max_epochs=100

srun python train.py max_epochs=10 compile=True data.train_data=data/trackml_fixed_length/v2_evt6200_train.bin data.val_data=data/trackml_fixed_length/v2_evt800_val.bin model.n_embd=1024 training.batch_size=2048 model.block_size=22 validation.num_batches=5 data.do_randomize=False log_interval=1000 init_from=resume ckpt_path=outputs/2023-11-10/18-41-10/best.ckpt optimizer.learning_rate=0.0001 slurm.auto_requeue=True
