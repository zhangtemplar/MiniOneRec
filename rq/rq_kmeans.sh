#!/bin/bash

#SBATCH --job-name=rq_kmeans_%j
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=rq_kmeans_%j.log
#SBATCH --error=rq_kmeans_%j.err

export NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=8
echo "NODE_RANK="$NODE_RANK
echo "NNODES="$SLURM_NNODES
echo "NPROC_PER_NODE="$NPROC_PER_NODE
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
##WORLD_SIZE=$(($SLURM_NNODES *  $SLURM_NTASKS_PER_NODE))
##echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

# set default value
# qwen 0.6B eval
dataset="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_item_text_train.npy"
output="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqkmeans/rankagi_output_v2_train_4096_4096.json"
test_dataset="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_item_text_eval.npy"
test_output="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqkmeans/rankagi_output_v2_eval_4096_4096.json"
max_beam_size=5
codebook_size="4096 4096"

# check arguments
dataset="${1:-$dataset}"
output="${2:-$output}"
test_dataset="${3:-$test_dataset}"
test_output="${4:-$test_output}"
max_beam_size="${5:-$max_beam_size}"
codebook_size="${6:-$codebook_size}"

free -h

nvidia-smi

echo $dataset $output $test_dataset $test_output $num_levels $codebook_size

python rqkmeans_faiss.py \
      --dataset $dataset \
      --output_root $output \
      --test_data $test_dataset \
      --test_data_output $test_output \
      --max_beam_size $max_beam_size \
      --codebook_size $codebook_size --uniform
