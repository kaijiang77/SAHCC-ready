from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import eval_transform, train_transform


@dataclass(frozen=True)
class DatasetParams:
    name: str
    train_split: str = "train"
    val_split: str = "test"
    patch_size: int = 128
    train_patch_num: int = 4
    train_scale: bool = True
    train_patch: bool = True
    train_flip: bool = True
    train_scale_min: float = 0.7
    train_scale_max: float = 1.3
    train_scale_min_size: int = 128
    train_flip_prob: float = 0.5


DATASET_PARAMS: dict[str, DatasetParams] = {
    "SHHA": DatasetParams(name="SHHA", val_split="test", patch_size=128, train_patch_num=4),
}


def _random_crop(image: torch.Tensor, points: np.ndarray, mean_mnn: np.ndarray, patch_size: int, num_patches: int):
    c, h, w = image.shape
    patches = np.zeros((num_patches, c, patch_size, patch_size), dtype=np.float32)
    point_list = []
    mean_list = []

    for i in range(num_patches):
        top = random.randint(0, max(0, h - patch_size))
        left = random.randint(0, max(0, w - patch_size))
        bottom, right = top + patch_size, left + patch_size

        patch = image[:, top:bottom, left:right]
        ph, pw = patch.shape[1:]
        if ph != patch_size or pw != patch_size:
            patch = torch.nn.functional.pad(
                patch.unsqueeze(0), (0, patch_size - pw, 0, patch_size - ph), mode="constant"
            ).squeeze(0)
        patches[i] = patch.cpu().numpy()

        mask = (points[:, 0] >= left) & (points[:, 0] < right) & (points[:, 1] >= top) & (points[:, 1] < bottom)
        pts = points[mask].copy()
        pts[:, 0] -= left
        pts[:, 1] -= top
        mnn = mean_mnn[mask].copy()
        point_list.append(torch.as_tensor(pts, dtype=torch.float32))
        mean_list.append(torch.as_tensor(mnn, dtype=torch.float32))

    return patches, point_list, mean_list


class UnifiedCrowdDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        transform,
        train: bool,
        scale: bool,
        patch: bool,
        flip: bool,
        patch_size: int,
        patch_num: int,
        scale_min: float,
        scale_max: float,
        scale_min_size: int,
        flip_prob: float,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.train = train
        self.scale = scale
        self.patch = patch
        self.flip = flip
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_min_size = scale_min_size
        self.flip_prob = flip_prob

        self.images_dir = root / "images"
        self.annotations_dir = root / "annotations"
        split_file = root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"split file not found: {split_file}")
        self.sample_ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        ann = np.load(str(self.annotations_dir / f"{sample_id}.npz"), allow_pickle=True)

        image_name = str(ann["image_filename"].item())
        image_path = self.images_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        image_h, image_w = image.shape[:2]
        image = self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        points = ann["points"].astype(np.float32).reshape(-1, 2)
        mean_mnn = ann["mean_mnn"].astype(np.float32).reshape(-1)
        labels = ann["labels"].astype(np.int64).reshape(-1)
        if not (points.shape[0] == mean_mnn.shape[0] == labels.shape[0]):
            raise ValueError(f"annotation shape mismatch: {sample_id}")

        if self.train and self.scale:
            s = random.uniform(self.scale_min, self.scale_max)
            h, w = image.shape[1:]
            if s * min(h, w) > self.scale_min_size:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), scale_factor=s, mode="bilinear", align_corners=False
                ).squeeze(0)
                points = points * s
                mean_mnn = mean_mnn * s

        if self.train and self.patch:
            images_np, points_list, mean_list = _random_crop(
                image=image, points=points, mean_mnn=mean_mnn, patch_size=self.patch_size, num_patches=self.patch_num
            )
            labels_list = [torch.ones(len(p), dtype=torch.long) for p in points_list]
        else:
            images_np = image.unsqueeze(0).cpu().numpy()
            points_list = [torch.as_tensor(points, dtype=torch.float32)]
            mean_list = [torch.as_tensor(mean_mnn, dtype=torch.float32)]
            labels_list = [torch.as_tensor(labels, dtype=torch.long)]

        if self.train and self.flip and random.random() < self.flip_prob:
            _, _, _, width = images_np.shape
            images_np = images_np[:, :, :, ::-1].copy()
            for i, pts in enumerate(points_list):
                if len(pts) > 0:
                    p = pts.clone()
                    p[:, 0] = width - p[:, 0]
                    points_list[i] = p

        image_id = int("".join(ch for ch in sample_id if ch.isdigit()) or 0)
        targets = []
        for pts, mnn, lbs in zip(points_list, mean_list, labels_list):
            targets.append(
                {
                    "points": pts.float(),
                    "mean_mnn": mnn.float(),
                    "labels": lbs.long(),
                    "image_id": torch.tensor([image_id], dtype=torch.long),
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "image_size": torch.tensor([image_w, image_h], dtype=torch.long),
                }
            )
        return torch.tensor(images_np, dtype=torch.float32), targets


def build_unified_datasets(cfg_data):
    params = DATASET_PARAMS.get(str(cfg_data.name), DatasetParams(name=str(cfg_data.name)))
    root = Path(cfg_data.root)
    train_split = cfg_data.get("train_split", params.train_split)
    configured_val_split = cfg_data.get("val_split", params.val_split)
    patch_size = int(cfg_data.get("patch_size", params.patch_size))
    train_patch_num = int(cfg_data.get("train_patch_num", params.train_patch_num))
    train_scale = bool(cfg_data.get("train_scale", params.train_scale))
    train_patch = bool(cfg_data.get("train_patch", params.train_patch))
    train_flip = bool(cfg_data.get("train_flip", params.train_flip))
    train_scale_min = float(cfg_data.get("train_scale_min", params.train_scale_min))
    train_scale_max = float(cfg_data.get("train_scale_max", params.train_scale_max))
    train_scale_min_size = int(cfg_data.get("train_scale_min_size", params.train_scale_min_size))
    train_flip_prob = float(cfg_data.get("train_flip_prob", params.train_flip_prob))
    val_split = configured_val_split if (root / "splits" / f"{configured_val_split}.txt").exists() else "test"

    train_set = UnifiedCrowdDataset(
        root=root,
        split=train_split,
        transform=train_transform(),
        train=True,
        scale=train_scale,
        patch=train_patch,
        flip=train_flip,
        patch_size=patch_size,
        patch_num=train_patch_num,
        scale_min=train_scale_min,
        scale_max=train_scale_max,
        scale_min_size=train_scale_min_size,
        flip_prob=train_flip_prob,
    )
    val_set = UnifiedCrowdDataset(
        root=root,
        split=val_split,
        transform=eval_transform(),
        train=False,
        scale=False,
        patch=False,
        flip=False,
        patch_size=patch_size,
        patch_num=1,
        scale_min=train_scale_min,
        scale_max=train_scale_max,
        scale_min_size=train_scale_min_size,
        flip_prob=0.0,
    )
    train_eval_set = UnifiedCrowdDataset(
        root=root,
        split=train_split,
        transform=eval_transform(),
        train=True,
        scale=False,
        patch=False,
        flip=False,
        patch_size=patch_size,
        patch_num=1,
        scale_min=train_scale_min,
        scale_max=train_scale_max,
        scale_min_size=train_scale_min_size,
        flip_prob=0.0,
    )
    return train_set, val_set, train_eval_set
