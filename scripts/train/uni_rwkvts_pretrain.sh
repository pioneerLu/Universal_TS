#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

export WANDB_MODE="${WANDB_MODE:-offline}"
DATA_FILE="${DATA_FILE:-$ROOT/data/pretrain/npy}"
CACHE_DIR="${CACHE_DIR:-$ROOT/data/cache/pretrain}"
PROJ_DIR="${PROJ_DIR:-out/pretrain}"
DEVICES="${DEVICES:-1}"

python train.py \
    --load_model "${CKPT:-}" \
    --wandb "${WANDB_PROJECT:-rwkvts}" \
    --proj_dir "$PROJ_DIR" \
    --data_file "$DATA_FILE" \
    --cache_dir "$CACHE_DIR" \
    --ctx_len 816 --epoch_steps 5000 --epoch_count 3 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 64 --accumulate_grad_batches 1 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 1e-4 --lr_final 2e-7 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices "$DEVICES" --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 \
    --enable_progress_bar True --sma_window 3 --validate_only 0 --dataset_type npy --num_vars 1 \
    --forecast_len 96 --loss_type mse --do_normalize False
