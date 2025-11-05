#!/bin/bash

#SBATCH --job-name=amazon18_data_process
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=amazon18_data_process.log
#SBATCH --error=amazon18_data_process.err

export NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=1
echo "NODE_RANK="$NODE_RANK
echo "NNODES="$SLURM_NNODES
echo "NPROC_PER_NODE="$NPROC_PER_NODE
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
##WORLD_SIZE=$(($SLURM_NNODES *  $SLURM_NTASKS_PER_NODE))
##echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

python amazon18_data_process.py \
    --dataset Industrial_and_Scientific \
    --user_k 5 \
    --item_k 5 \
    --st_year 2017 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 11 \
    --metadata_file /mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/meta_Industrial_and_Scientific.jsonl \
    --reviews_file /mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/Industrial_and_Scientific.jsonl \
    --output_path /mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/