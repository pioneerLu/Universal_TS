#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

python inference.py \
    --load_model "${CKPT:?set CKPT}" \
    --proj_dir "${PROJ_DIR:-out/infer}" \
    --data_file "${DATA_FILE:?set DATA_FILE}" \
    --ctx_len "${CTX_LEN:-816}" --n_layer 6 --n_embd 512 \
    --micro_bsz 1 --accelerator cpu --devices 1 --precision bf16 --strategy ddp \
    --sma_window 3 --horizon "${HORIZON:-96}" --grad_cp 0 \
    --device "${DEVICE:-cpu}" \
    --select_indices "${SELECT:-3}" --feature_used "${FEATURES:-0,3}"
