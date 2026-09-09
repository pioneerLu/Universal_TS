#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

CKPT="${CKPT:?set CKPT to a .pth checkpoint}"
DATA_FILE="${DATA_FILE:-$ROOT/data/station/wind/f1-4.npy}"
PROJ_DIR="${PROJ_DIR:-out/infer}"
DEVICE="${DEVICE:-cpu}"

python inference.py \
    --load_model "$CKPT" \
    --proj_dir "$PROJ_DIR" \
    --data_file "$DATA_FILE" \
    --ctx_len 816 --n_layer 6 --n_embd 512 \
    --micro_bsz 1 --accelerator cpu --devices 1 --precision bf16 --strategy ddp \
    --sma_window 3 --num_vars 2 \
    --select_indices 3 --feature_used 0,3 --horizon 96 --grad_cp 0 \
    --device "$DEVICE"
