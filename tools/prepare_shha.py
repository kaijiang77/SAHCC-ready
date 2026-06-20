#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DATASET_NAME = "SHHA"
DEFAULT_SOURCE_SUBDIR = Path("ShanghaiTech") / "part_A"
SPLIT_MAP = {"train_data": "train", "test_data": "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def should_generate(dst: Path, overwrite: bool) -> bool:
    return overwrite or not dst.exists()


def is_hidden_artifact(path: Path) -> bool:
    return path.name.startswith(".") or path.name.startswith("._")


def make_sample_id(split: str, stem: str) -> str:
    safe_stem = stem.replace("/", "-").replace("\\", "-").strip()
    return f"{split}-{safe_stem}"


def ensure_array(data, dtype=None, shape_last: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(data if data is not None else [], dtype=dtype)
    if shape_last is not None:
        arr = arr.reshape(-1, shape_last)
    return arr


def compute_mean_mnn(points: np.ndarray, k: int = 3) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)

    points = ensure_array(points, dtype=np.float32, shape_last=2)
    if points.shape[0] == 1:
        return np.ones((1,), dtype=np.float32)

    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32))
    np.fill_diagonal(dist, np.inf)

    k = min(int(k), max(1, points.shape[0] - 1))
    nearest = np.partition(dist, kth=k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1).astype(np.float32)


def normalize_mean_mnn(
    points: np.ndarray,
    mean_mnn: Optional[np.ndarray],
    fill_policy: str,
    mean_k: int,
) -> np.ndarray:
    points = ensure_array(points, dtype=np.float32, shape_last=2)
    if mean_mnn is not None:
        mean_mnn = np.asarray(mean_mnn, dtype=np.float32).reshape(-1)
        if mean_mnn.shape[0] == points.shape[0]:
            return mean_mnn
        if mean_mnn.size > 0:
            return np.full((points.shape[0],), float(mean_mnn.mean()), dtype=np.float32)

    if fill_policy == "compute":
        return compute_mean_mnn(points, k=mean_k)
    if fill_policy == "ones":
        return np.ones((points.shape[0],), dtype=np.float32)
    return np.zeros((points.shape[0],), dtype=np.float32)


def make_labels(points: np.ndarray) -> np.ndarray:
    return np.ones((points.shape[0],), dtype=np.int64)


def image_wh(image_path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(image_path) as img:
        width, height = img.size
    return np.asarray([width, height], dtype=np.int32)


def load_mat(path: Path) -> Dict[str, object]:
    import scipy.io as io

    return io.loadmat(str(path))


def parse_shha_mat(
    mat_path: Optional[Path],
    fill_policy: str,
    mean_k: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if mat_path is None or not mat_path.exists():
        points = np.zeros((0, 2), dtype=np.float32)
        mean_mnn = normalize_mean_mnn(points, None, fill_policy, mean_k)
        return points, mean_mnn, {"annotation_keys": np.asarray([], dtype="<U1")}

    mat = load_mat(mat_path)
    if "annPoints" in mat:
        points = ensure_array(mat["annPoints"], dtype=np.float32, shape_last=2)
    else:
        points = ensure_array(mat["image_info"][0][0][0][0][0], dtype=np.float32, shape_last=2)

    extra = {
        "annotation_keys": np.asarray(sorted(k for k in mat.keys() if not k.startswith("__")), dtype="<U64"),
        "source_mat_name": np.asarray(mat_path.name),
    }
    mean_mnn = normalize_mean_mnn(points, mat.get("mean_mnn"), fill_policy, mean_k)
    return points, mean_mnn, extra


def resolve_gt_dir(split_dir: Path) -> Path:
    candidates = [
        split_dir / "ground_truth",
        split_dir / "ground-truth",
        split_dir / "gt",
    ]
    for path in candidates:
        if path.exists():
            return path
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"SHHA ground-truth directory not found. Expected one of: {expected}")


def materialize_path(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def discover_shha_samples(
    source_dir: Path,
    unified_dir: Path,
    image_mode: str,
    fill_policy: str,
    mean_k: int,
    overwrite: bool,
) -> Dict[str, int]:
    split_counts: Dict[str, int] = {}
    images_dir = unified_dir / "images"
    ann_dir = unified_dir / "annotations"
    splits_dir = unified_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for source_split, unified_split in SPLIT_MAP.items():
        img_dir = source_dir / source_split / "images"
        refine_dir = source_dir / source_split / "refine_gt"
        density_dir = source_dir / source_split / "density"

        if not img_dir.exists():
            raise FileNotFoundError(f"SHHA image directory not found: {img_dir}")
        gt_dir = resolve_gt_dir(source_dir / source_split)

        split_ids: List[str] = []
        for image_path in sorted(img_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if is_hidden_artifact(image_path):
                continue

            stem = image_path.stem
            sample_id = make_sample_id(unified_split, stem)
            image_dst_name = f"{sample_id}{image_path.suffix.lower()}"
            materialize_path(image_path, images_dir / image_dst_name, image_mode, overwrite)

            gt_path = gt_dir / f"GT_{stem}.mat"
            if not gt_path.exists():
                raise FileNotFoundError(f"SHHA annotation file not found: {gt_path}")
            refine_path = refine_dir / f"GT_{stem}.mat"
            density_path = density_dir / f"{stem}.npy"
            ann_path = ann_dir / f"{sample_id}.npz"

            if should_generate(ann_path, overwrite):
                points, mean_mnn, mat_extra = parse_shha_mat(gt_path, fill_policy, mean_k)
                refined_points, refined_mean_mnn, _ = parse_shha_mat(
                    refine_path if refine_path.exists() else None,
                    fill_policy,
                    mean_k,
                )

                annotation: Dict[str, object] = {
                    "dataset": np.asarray(DATASET_NAME),
                    "split": np.asarray(unified_split),
                    "sample_id": np.asarray(sample_id),
                    "image_filename": np.asarray(image_dst_name),
                    "source_image_path": np.asarray(str(image_path)),
                    "source_gt_path": np.asarray(str(gt_path)),
                    "source_refine_gt_path": np.asarray(str(refine_path) if refine_path.exists() else ""),
                    "image_wh": image_wh(image_path),
                    "points": points,
                    "mean_mnn": mean_mnn,
                    "labels": make_labels(points),
                    "count": np.asarray([points.shape[0]], dtype=np.int32),
                    "has_gt": np.asarray([1 if gt_path.exists() else 0], dtype=np.uint8),
                    "has_refine_gt": np.asarray([1 if refine_path.exists() else 0], dtype=np.uint8),
                    "refined_points": refined_points,
                    "refined_mean_mnn": refined_mean_mnn,
                    "annotation_format": np.asarray("mat"),
                    **mat_extra,
                }

                if density_path.exists():
                    annotation["source_density_path"] = np.asarray(str(density_path))
                    annotation["density"] = np.load(str(density_path)).astype(np.float32)
                    annotation["has_density"] = np.asarray([1], dtype=np.uint8)
                else:
                    annotation["source_density_path"] = np.asarray("")
                    annotation["density"] = np.zeros((0,), dtype=np.float32)
                    annotation["has_density"] = np.asarray([0], dtype=np.uint8)

                np.savez_compressed(ann_path, **annotation)
            split_ids.append(sample_id)

        with (splits_dir / f"{unified_split}.txt").open("w", encoding="utf-8") as f:
            for sample_id in split_ids:
                f.write(f"{sample_id}\n")
        split_counts[unified_split] = len(split_ids)

    return split_counts


def write_meta(
    source_dir: Path,
    raw_dir: Path,
    unified_dir: Path,
    split_counts: Dict[str, int],
    extra_fields: Iterable[str],
) -> None:
    meta = {
        "dataset": DATASET_NAME,
        "raw_dir": str(raw_dir),
        "source_dir": str(source_dir),
        "unified_dir": str(unified_dir),
        "annotation_format": "npz",
        "image_storage": "images/<split>-<stem>.<ext>",
        "annotation_storage": "annotations/<split>-<stem>.npz",
        "split_storage": "splits/<split>.txt",
        "common_fields": [
            "dataset",
            "split",
            "sample_id",
            "image_filename",
            "image_wh",
            "points",
            "mean_mnn",
            "labels",
            "count",
            "has_gt",
        ],
        "extra_fields": sorted(extra_fields),
        "split_counts": split_counts,
    }

    with (unified_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_shha_dataset(
    source_dir: Path,
    output_root: Path,
    raw_mode: str,
    image_mode: str,
    fill_policy: str,
    mean_k: int,
    overwrite: bool,
) -> Dict[str, object]:
    raw_dir = output_root / "raw" / DATASET_NAME
    unified_dir = output_root / "unified" / DATASET_NAME

    if not source_dir.exists():
        raise FileNotFoundError(f"SHHA source dataset directory not found: {source_dir}")

    if overwrite and unified_dir.exists():
        shutil.rmtree(unified_dir)

    (unified_dir / "images").mkdir(parents=True, exist_ok=True)
    (unified_dir / "annotations").mkdir(exist_ok=True)
    (unified_dir / "splits").mkdir(exist_ok=True)

    if raw_mode != "skip":
        materialize_path(source_dir, raw_dir, raw_mode, overwrite)

    split_counts = discover_shha_samples(
        source_dir=source_dir,
        unified_dir=unified_dir,
        image_mode=image_mode,
        fill_policy=fill_policy,
        mean_k=mean_k,
        overwrite=overwrite,
    )
    extra_fields = {
        "annotation_format",
        "annotation_keys",
        "source_mat_name",
        "source_refine_gt_path",
        "has_refine_gt",
        "refined_points",
        "refined_mean_mnn",
        "source_density_path",
        "density",
        "has_density",
    }
    write_meta(source_dir, raw_dir, unified_dir, split_counts, extra_fields)

    return {
        "dataset": DATASET_NAME,
        "source_dir": str(source_dir),
        "raw_dir": str(raw_dir),
        "unified_dir": str(unified_dir),
        "split_counts": split_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the unified SHHA dataset layout under data/raw and data/unified."
    )
    parser.add_argument(
        "--source-root",
        default="data",
        help="Root that contains ShanghaiTech/part_A.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Direct path to ShanghaiTech part_A. Overrides --source-root when set.",
    )
    parser.add_argument(
        "--output-root",
        default="data",
        help="Root where raw/SHHA and unified/SHHA will be created.",
    )
    parser.add_argument(
        "--raw-mode",
        default="symlink",
        choices=["symlink", "copy", "skip"],
        help="How to materialize data/raw/SHHA.",
    )
    parser.add_argument(
        "--image-mode",
        default="symlink",
        choices=["symlink", "copy"],
        help="How to materialize unified images.",
    )
    parser.add_argument(
        "--fill-missing-mean-mnn",
        default="compute",
        choices=["compute", "ones", "zeros"],
        help="How to fill mean_mnn when the raw annotation does not provide it.",
    )
    parser.add_argument(
        "--mean-k",
        default=2,
        type=int,
        help="Nearest-neighbor count used when computing missing mean_mnn.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing raw/unified outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    source_dir = Path(args.source_dir).resolve() if args.source_dir else source_root / DEFAULT_SOURCE_SUBDIR
    output_root = Path(args.output_root).resolve()

    summary = build_shha_dataset(
        source_dir=source_dir,
        output_root=output_root,
        raw_mode=args.raw_mode,
        image_mode=args.image_mode,
        fill_policy=args.fill_missing_mean_mnn,
        mean_k=args.mean_k,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
