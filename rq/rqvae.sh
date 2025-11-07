#!/bin/bash

#SBATCH --job-name=ravae_%j
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=ravae_%j.log
#SBATCH --error=ravae_%j.err

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
output="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqvae/rankagi_output_v2_train_512_512_512_0.6B.json"
test_dataset="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_item_text_eval.npy"
test_output="/mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rqvae/rankagi_output_v2_eval_512_512_512_0.6B.json"
batch_size=3
codebook_size=256

# check arguments
dataset="${1:-$dataset}"
output="${2:-$output}"
test_dataset="${3:-$test_dataset}"
test_output="${4:-$test_output}"
batch_size="${5:-$batch_size}"
codebook_size="${6:-$codebook_size}"

free -h

nvidia-smi

echo $dataset $output $test_dataset $test_output $num_levels $codebook_size

# for 4096D / qwen 8B
# python rqvae.py \
#     --data_path $dataset \
#     --ckpt_dir $output \
#     --layers 4096 2048 1024 512 256 128 \
#     --e_dim 128 \
#     --batch_size 65536 \
#     --num_emb_list 512 512 512 \
#     --lr 1e-3 \
#     --epochs 500 \
#     --warmup_epochs 10 \
#     --eval_step 10 \
#     --kmeans_init True \
#     --kmeans_iters 100 \
#     --sk_epsilons 0.0 0.01 0.05 \
#     --sk_iters 50 \
#     --beta 0.25 \
#     --quant_loss_weight 1.0 \
#     --learner AdamW \
#     --weight_decay 1e-5 \
#     --lr_scheduler_type constant \
#     --num_workers 8 \
#     --device cuda:0

# for 1024D / qwen 0.6B
python rqvae.py \
    --data_path $dataset \
    --ckpt_dir $output \
    --layers 1024 512 256 128 \
    --e_dim 64 \
    --batch_size 131072 \
    --num_emb_list 512 512 512 \
    --lr 2e-3 \
    --epochs 500 \
    --warmup_epochs 10 \
    --eval_step 10 \
    --kmeans_init True \
    --kmeans_iters 100 \
    --sk_epsilons 0.0 0.01 0.05 \
    --sk_iters 50 \
    --beta 0.25 \
    --quant_loss_weight 1.0 \
    --learner AdamW \
    --weight_decay 1e-5 \
    --lr_scheduler_type constant \
    --num_workers 16 \
    --device cuda:0