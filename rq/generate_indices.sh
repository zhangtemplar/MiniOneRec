#!/bin/bash

#SBATCH --job-name=generate_indices_%j
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=generate_indices_%j.log
#SBATCH --error=generate_indices_%j.err

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
ckpt_path="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqvae/rankagi_output_v2_8B/Nov-08-2025_00-10-16/best_loss_model.pth"
dataset="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_item_text_train_8B.npy"
output="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqvae/rankagi_output_v2_train_512_512_512_8B.json"
test_dataset="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_item_text_eval_8B.npy"
test_output="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqvae/rankagi_output_v2_eval_512_512_512_8B.json"

# check arguments
dataset="${1:-$dataset}"
output="${2:-$output}"
test_dataset="${3:-$test_dataset}"
test_output="${4:-$test_output}"
ckpt_path="${5:-$ckpt_path}"

free -h

nvidia-smi

echo $dataset $output $test_dataset $test_output $ckpt_path

# for 1024D / qwen 0.6B
python generate_indices.py \
    --dataset $dataset \
    --output_file $output \
    --ckpt_path $ckpt_path

python generate_indices.py \
    --dataset $test_dataset \
    --output_file $test_output \
    --ckpt_path $ckpt_path