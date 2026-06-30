#!/usr/bin/env bash
set -euo pipefail

python tools/prepare_shha.py --help >/dev/null
python -m py_compile \
  src/train.py \
  src/eval.py \
  src/data/collate.py \
  src/data/crowd_datamodule.py \
  src/data/transforms.py \
  src/data/unified_dataset.py \
  src/models/backbone.py \
  src/models/matcher.py \
  src/models/sah.py \
  src/models/sahcc.py \
  src/models/vgg.py \
  src/modules/lit_crowd.py \
  src/modules/losses.py \
  src/modules/metrics.py \
  tools/prepare_shha.py

python - <<'PY2'
import cv2
import hydra
import lightning
import numpy
import omegaconf
import PIL
import scipy
import torch
import torchvision

print("Smoke check passed.")
PY2
