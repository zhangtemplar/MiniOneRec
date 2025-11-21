#!/bin/bash

#SBATCH --job-name=convert_codes_to_parquet_%j
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=convert_codes_to_parquet_%j.log
#SBATCH --error=convert_codes_to_parquet_%j.err
#SBATCH --exclude=metavmds1-a4-173,metavmds1-a4-148,metavmds1-a4-239,metavmds1-a4-5,metavmds1-a4-156

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
dataset="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqkmeans/rankagi_output_v2_train_8B_256x6.json"
test_dataset="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqkmeans/rankagi_output_v2_eval_8B_256x6.json"
output="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqkmeans/qwen_8B_rqkmeans_256x6_merrec_item_to_sid.parquet"

# check arguments
dataset="${1:-$dataset}"
test_dataset="${2:-$test_dataset}"
output="${3:-$output}"

free -h

echo $dataset $output $test_dataset

python convert_codes_to_parquet.py \
      --train_json $dataset \
      --output $output \
      --eval_json $test_dataset \
      --validate
