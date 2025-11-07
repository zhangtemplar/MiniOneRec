#!/bin/bash
# usage: merrec_text2emb.sh [dataset] [output] [model_name]

#SBATCH --job-name=merrec_text2emb_%j
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=merrec_text2emb_%j.log
#SBATCH --error=merrec_text2emb_%j.err

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
dataset="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/rankagi_output_v2/src/merrec/item_text/item_text_train.jsonl"
output="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_item_text_train_8B.npy"
embedding_model="Qwen/Qwen3-Embedding-8B"

# check arguments
dataset="${1:-$dataset}"
output="${2:-$output}"
embedding_model="${3:-$embedding_model}"

nvidia-smi

echo $dataset $output $embedding_model

python merrec_text2emb.py \
      --dataset $dataset \
      --root $output \
      --plm_name qwen \
      --plm_checkpoint $embedding_model
