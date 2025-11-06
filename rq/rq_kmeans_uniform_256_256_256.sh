#!/bin/bash

#SBATCH --job-name=rq_kmeans_uniform_256_256_256
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=rq_kmeans_uniform_256_256_256.log
#SBATCH --error=rq_kmeans_uniform_256_256_256.err

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

free -h

nvidia-smi

python rqkmeans_faiss.py \
      --dataset /mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_item_text_train.npy \
      --output_root /mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_train_uniform_256_256_256.json \
      --num_levels 3 \
      --codebook_size 256 \
      --uniform
