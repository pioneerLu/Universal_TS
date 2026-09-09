#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python preprocess_data.py \
    --data_file "${DATA_FILE:?set DATA_FILE to an npy directory or file}" \
    --cache_dir "${CACHE_DIR:-$ROOT/data/cache/pretrain}" \
    --input_len "${CTX_LEN:-816}" \
    --output_len 0 \
    --split 0.8 \
    --norm True \
    --stride 1
