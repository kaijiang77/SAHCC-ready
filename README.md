# Local Spacing-Aware Hungarian Matching for Stable Point-Supervised Crowd Counting

<p align="center">
  <img src="assets/fig/Matcher.png" alt="Local Spacing-Aware Hungarian Matching" width="95%">
</p>

## Experimental Results

Main results on crowd counting (MAE/RMSE). The table below includes the point-based methods from Table 1 of the paper. Lower is better, and the best result for each metric is highlighted in bold.

| Method | SHHA MAE | SHHA RMSE | SHHB MAE | SHHB RMSE | UCF-QNRF MAE | UCF-QNRF RMSE | JHU-Crowd++ MAE | JHU-Crowd++ RMSE | NWPU(T) MAE | NWPU(T) RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2PNet | 52.74 | 85.06 | 6.25 | 9.90 | 85.32 | 154.50 | - | - | 83.28 | 553.92 |
| CLTR | 56.90 | 95.20 | 6.50 | 10.60 | 85.80 | 141.30 | 59.50 | 240.60 | 74.30 | 333.80 |
| PET | 49.34 | 78.77 | 6.19 | 9.69 | 79.53 | 144.32 | 58.50 | 238.00 | 74.40 | 328.50 |
| APGCC | 48.80 | 76.70 | **5.60** | **8.70** | 80.10 | 136.60 | **54.30** | **225.90** | 71.40 | **284.40** |
| P2R | 51.02 | 79.68 | 6.17 | 9.84 | 83.26 | 138.11 | 58.83 | 253.10 | - | - |
| Ours | **47.23** | **75.17** | 6.14 | 9.50 | **76.91** | **135.92** | 58.32 | 250.62 | **68.90** | 306.70 |

SHHA training log (MAE/RMSE: 46.59/75.49): [assets/logs/train.log](assets/logs/train.log)

SHHA checkpoint (MAE/RMSE: 46.59/75.49): [GitHub Release](https://github.com/kaijiang77/SAHCC-ready/releases)

---

## Installation

```bash
git clone https://github.com/kaijiang77/SAHCC-ready.git
cd SAHCC-ready

conda create -n sahcc python=3.12
conda activate sahcc
pip install -r requirements.txt
```

The project was tested with Python 3.12, PyTorch 2.9.0, Lightning 2.6.1, and CUDA 13.0. The provided `requirements.txt` installs CUDA 13.0 PyTorch wheels by default. If your platform requires a different CUDA version, install `torch` and `torchvision` from the official PyTorch selector first, then install the remaining dependencies.

---

## Dataset Preparation

This release uses ShanghaiTech Part A (SHHA). Large datasets are not included in this repository. Please download the dataset separately and place it under `data/ShanghaiTech/part_A`.

Expected raw data structure:

```text
data/ShanghaiTech/part_A/
├── train_data/
│   ├── images/
│   └── ground_truth/
└── test_data/
    ├── images/
    └── ground_truth/
```

The preparation script also accepts `ground-truth/` and `gt/` as annotation directory names.

Build the project dataset layout with:

```bash
python tools/prepare_shha.py --overwrite
```

By default, the script creates symlinks for images and writes compressed point annotations into `data/unified/SHHA`.

Processed data structure:

```text
data/unified/SHHA/
├── images/
│   ├── train-IMG_*.jpg
│   └── test-IMG_*.jpg
├── annotations/
│   ├── train-IMG_*.npz
│   └── test-IMG_*.npz
├── splits/
│   ├── train.txt
│   └── test.txt
└── meta.json
```

Each annotation file stores point-level labels in NumPy format. The main fields are:

```text
points:   N x 2 array with x/y point coordinates
labels:   N array, where each point is a foreground person label
count:    image-level crowd count
mean_mnn: local nearest-neighbor spacing used by SAH matching
```

`data/unified/SHHA/` is generated runtime data and is ignored by Git.

---

## Quick Start

The commands below show the minimal path for reproducing the SHHA experiment.

### 1. Download VGG16-BN Backbone Weight

```bash
curl -L -o pretrained/vgg16_bn-6c64b313.pt \
  https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
```

### 2. Prepare SHHA

Place ShanghaiTech Part A under `data/ShanghaiTech/part_A`, then run:

```bash
python tools/prepare_shha.py --overwrite
```

### 3. Train

```bash
EXP_CFG=shha_sah bash scripts/train.sh
```

### 4. Evaluate

```bash
CKPT_PATH=/path/to/checkpoint.ckpt bash scripts/eval.sh
```

---

## Training

The recommended SHHA training entry is the shell script:

```bash
EXP_CFG=shha_sah bash scripts/train.sh
```

The same run can also be launched directly with Python:

```bash
python -m src.train +experiment=shha_sah trainer.devices=1
```

`+experiment=shha_sah` loads `configs/experiment/shha_sah.yaml`, which selects the SHHA data config, the SAHCC model, and the SAH matcher. The default SHHA dataset root is `data/unified/SHHA`.

Useful training overrides:

```bash
# Use GPU 0 through the shell script.
GPU_IDS=0 EXP_CFG=shha_sah bash scripts/train.sh

# Use multiple visible GPUs through the shell script.
GPU_IDS=0,1 EXP_CFG=shha_sah bash scripts/train.sh

# Override Hydra options directly.
python -m src.train +experiment=shha_sah trainer.devices=1 trainer.max_epochs=2000 data.batch_size=8
```

Training logs are written to `logs/<run_name>/`, and checkpoints are written to `weights/<run_name>/` by default. For the SHHA experiment, `run_name` is `shha_sah` unless overridden.

---

## Evaluation

Evaluate a trained checkpoint with:

```bash
CKPT_PATH=/path/to/checkpoint.ckpt bash scripts/eval.sh
```

The equivalent Python command is:

```bash
python -m src.eval data=shha eval.ckpt_path=/path/to/checkpoint.ckpt
```

The default evaluation uses the SHHA test split defined in `configs/data/shha.yaml`. The evaluation script reports the counting metrics produced by the Lightning validation loop, including MAE and RMSE.

The VGG16-BN file under `pretrained/` is only the backbone initialization weight. The trained SHHA checkpoint should be downloaded from GitHub Releases and passed through `eval.ckpt_path`.

Useful evaluation overrides:

```bash
# Select GPU 0 for evaluation.
CUDA_VISIBLE_DEVICES=0 python -m src.eval data=shha eval.ckpt_path=/path/to/checkpoint.ckpt trainer.devices=1

# Evaluate a checkpoint downloaded from GitHub Releases.
python -m src.eval data=shha eval.ckpt_path=weights/shha_sah/best.ckpt
```

---

## Project Structure

```text
SAHCC-ready/
├── configs/       Hydra configs for data, model, matcher, trainer, and experiments
├── scripts/       shell entry points for training, evaluation, and repository checks
├── src/           core implementation: data modules, models, losses, and training logic
├── tools/         SHHA preprocessing utilities
├── assets/        paper figure, paper PDF, and released training log
├── data/          dataset placeholder and local generated data
├── pretrained/    external backbone weights, not model checkpoints
├── weights/       local training checkpoints, ignored by Git
├── logs/          local TensorBoard logs, ignored by Git
└── outputs/       local Hydra/runtime outputs, ignored by Git
```

The public code path is centered on SHHA. Other datasets and their preprocessing scripts are intentionally not included in this release.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you find this project useful, please consider citing our paper. The formal citation will be updated after the camera-ready metadata is finalized.

---

## Acknowledgements

This project is built upon [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet). We also thank the authors of [PET](https://github.com/cxliu0/PET) and [P2R](https://github.com/Elin24/P2RLoss) for their inspiring work on point-based crowd counting.

