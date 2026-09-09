#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

CKPT="${CKPT:?set CKPT to a LoRA checkpoint}"
DATA_FILE="${DATA_FILE:?set DATA_FILE}"
PROJ_DIR="${PROJ_DIR:-out/lora_inference}"
DEVICE="${DEVICE:-cpu}"

python lora_inference.py \
    --load_model "$CKPT" \
    --proj_dir "$PROJ_DIR" \
    --data_file "$DATA_FILE" \
    --ctx_len 816 --n_layer 6 --n_embd 512 \
    --num_vars 1 --select_indices 0 --forecast_len 96 \
    --horizon 168 --device "$DEVICE" --merge_lora True \
    --accelerator cpu --devices 1 --precision bf16 --strategy auto
