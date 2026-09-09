#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

export WANDB_MODE="${WANDB_MODE:-offline}"
DATA_FILE="${DATA_FILE:-$ROOT/data/station/wind/f1-4.npy}"
CKPT="${CKPT:-$ROOT/out/scale_up_3_shuffle/checkpoints/best-0.172.pth}"
PROJ_DIR="${PROJ_DIR:-out/multi_from_uni}"

python train.py \
    --load_model "$CKPT" \
    --wandb "${WANDB_PROJECT:-rwkvts}" \
    --proj_dir "$PROJ_DIR" \
    --data_file "$DATA_FILE" \
    --ctx_len 672 --epoch_steps 200 --epoch_count 3 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 64 --accumulate_grad_batches 1 --n_layer 6 --n_embd 512 --pre_ffn 0 \
    --lr_init 1e-4 --lr_final 5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --enable_progress_bar True --sma_window 3 --validate_only 0 --dataset_type multi_npy --num_vars 3 \
    --select_indices 0,1 --feature_used 0,1,2 --forecast_len 96 --do_normalize True
