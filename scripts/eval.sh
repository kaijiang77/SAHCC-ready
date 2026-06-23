#!/usr/bin/env bash
set -euo pipefail

CKPT_PATH="${CKPT_PATH:?Please set CKPT_PATH}"
DATA_CFG="${DATA_CFG:-shha}"

python -m src.eval data="${DATA_CFG}" eval.ckpt_path="${CKPT_PATH}" "$@"
