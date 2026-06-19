# SAHCC

This repository contains the SHHA reproduction code for SAHCC crowd counting experiments.

## Paper

The paper draft is available at [assets/paper.pdf](assets/paper.pdf).

## Structure

```text
assets/paper.pdf          Paper draft
configs/                  Training, evaluation, and SHHA experiment configs
src/                      Model, training, evaluation, losses, and metrics
scripts/                  Train/eval entry scripts and annotation-noise experiments
tools/prepare_shha.py     SHHA dataset preparation script
data/README.md            Expected local dataset layout
```

Only ShanghaiTech Part A (SHHA) is part of the public data-preparation path. Other dataset conversion scripts and local experiment utilities are intentionally excluded.

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n sahcc python=3.12
conda activate sahcc
pip install -r requirements.txt
```

This project was tested with Python 3.12, PyTorch 2.9.0, Lightning 2.6.1, and CUDA 13.0. If `pip install -r requirements.txt` cannot resolve the PyTorch wheels for your platform, install `torch` and `torchvision` from the official PyTorch selector first, then install the remaining requirements:

```bash
pip install lightning==2.6.1 hydra-core==1.3.2 omegaconf==2.3.0 tensorboard==2.20.0
pip install numpy==1.26.4 scipy==1.16.2 pillow==11.3.0 opencv-python==4.11.0.86
```

## Pretrained Backbone

The model requires the ImageNet pretrained VGG16-BN backbone weights. Download the official PyTorch weight file and save it under `pretrained/` with the filename expected by this project:

```bash
curl -L -o pretrained/vgg16_bn-6c64b313.pt \
  https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
```

The training code loads this file from:

```text
pretrained/vgg16_bn-6c64b313.pt
```

Do not place this file under `weights/`; `weights/` is reserved for training checkpoints and other runtime outputs. The `pretrained/` directory is ignored by Git except for its README.

## Prepare SHHA

Download ShanghaiTech Part A and place it at:

```text
data/ShanghaiTech/part_A
```

Then build the unified layout:

```bash
python tools/prepare_shha.py --overwrite
```

To use another source path:

```bash
python tools/prepare_shha.py \
  --source-dir /path/to/ShanghaiTech/part_A \
  --output-root data \
  --raw-mode skip \
  --image-mode symlink \
  --overwrite
```

The script writes:

```text
data/unified/SHHA/images/
data/unified/SHHA/annotations/
data/unified/SHHA/splits/train.txt
data/unified/SHHA/splits/test.txt
data/unified/SHHA/meta.json
```

## Train

```bash
bash scripts/train.sh
```

The default experiment is `shha_baseline`. To switch experiment config:

```bash
EXP_CFG=shha_sah bash scripts/train.sh
```

## Evaluate

```bash
CKPT_PATH=/path/to/checkpoint.ckpt bash scripts/eval.sh
```

## Results

The public release focuses on ShanghaiTech Part A (SHHA). Fill the table below with the numbers from the final paper/release checkpoint.

| Method | Dataset | Train Split | Test Split | MAE | RMSE | Checkpoint |
| --- | --- | --- | --- | ---: | ---: | --- |
| SAHCC | SHHA | train | test | TBD | TBD | Coming soon |

## Checkpoints

Model checkpoints are not stored in Git. Release checkpoints through GitHub Releases, Hugging Face, Google Drive, or another stable file host, then update the `Checkpoint` column above and the command below:

```bash
CKPT_PATH=/path/to/sahcc_shha.ckpt bash scripts/eval.sh
```

## Quick Check

Run the smoke test before publishing changes:

```bash
bash scripts/check.sh
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{sahcc2026,
  title = {Local Spacing-Aware Hungarian Matching for Crowd Counting},
  author = {SAHCC Authors},
  booktitle = {European Conference on Computer Vision},
  year = {2026}
}
```

Please also update `CITATION.cff` with the final author list, repository URL, and publication metadata before the official release.
