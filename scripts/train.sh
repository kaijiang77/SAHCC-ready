#!/usr/bin/env bash
set -euo pipefail

DATA_CFG="${DATA_CFG:-shha}"
EXP_CFG="${EXP_CFG:-shha_sah}"
GPU_IDS="${GPU_IDS:-0}"
TRAINER_DEVICES="${TRAINER_DEVICES:-}"

if [[ $# -gt 0 && "$1" != *=* && "$1" != +* && "$1" != -* ]]; then
  GPU_IDS="$1"
  shift
fi

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

if [[ -z "${TRAINER_DEVICES}" && -n "${GPU_IDS}" ]]; then
  IFS=',' read -ra _gpu_ids <<< "${GPU_IDS}"
  TRAINER_DEVICES="${#_gpu_ids[@]}"
fi

ARGS=()
if [[ -n "${EXP_CFG}" ]]; then
  ARGS+=(+experiment="${EXP_CFG}")
else
  ARGS+=(data="${DATA_CFG}")
fi

if [[ -n "${TRAINER_DEVICES}" ]]; then
  ARGS+=(trainer.devices="${TRAINER_DEVICES}")
fi

ARGS+=("$@")
python -m src.train "${ARGS[@]}"
