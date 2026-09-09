#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

export WANDB_MODE="${WANDB_MODE:-offline}"
DATA_FILE="${DATA_FILE:-$ROOT/data/pretrain/npy}"
CKPT="${CKPT:?set CKPT to a pretrained .pth}"
PROJ_DIR="${PROJ_DIR:-out/lora_finetune}"

python lora_train.py \
    --load_model "$CKPT" \
    --wandb "${WANDB_PROJECT:-rwkvts_lora}" \
    --proj_dir "$PROJ_DIR" \
    --data_file "$DATA_FILE" \
    --ctx_len 816 --epoch_steps 1000 --epoch_count 20 --epoch_begin 0 --epoch_save 5 \
    --micro_bsz 32 --accumulate_grad_batches 1 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 5e-5 --lr_final 1e-6 --warmup_steps 100 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy auto --grad_cp 0 \
    --enable_progress_bar True --sma_window 3 --validate_only 0 --dataset_type npy --num_vars 1 \
    --forecast_len 96 --loss_type mse --do_normalize False \
    --lora_rank 8 --lora_alpha 16 --lora_target_modules receptance,key,value,output
