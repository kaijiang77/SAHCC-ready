from __future__ import annotations

import re
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from src.data.collate import collate_fn_crowd
from src.data.crowd_datamodule import CrowdDataModule
from src.data.transforms import eval_transform
from src.data.unified_dataset import UnifiedCrowdDataset
from src.modules.lit_crowd import LitCrowdModel


def _list_from_cfg(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [x.strip() for x in value.split(',') if x.strip()]
    return [str(value)]


def _int_list_from_cfg(value) -> set[int]:
    out = set()
    for item in _list_from_cfg(value):
        try:
            out.add(int(item))
        except ValueError:
            match = re.search(r'(\d+)$', item)
            if match:
                out.add(int(match.group(1)))
    return out


def _select_device(cfg: DictConfig) -> torch.device:
    accelerator = str(cfg.trainer.get('accelerator', 'auto')).lower()
    if accelerator in {'gpu', 'cuda', 'auto'} and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def _safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_') or 'sample'


def _target_image_size(target) -> tuple[int, int]:
    size = target.get('image_size')
    if size is None:
        return 0, 0
    if torch.is_tensor(size):
        size = size.detach().cpu().tolist()
    return int(size[0]), int(size[1])


def _target_image_id(target) -> int | None:
    image_id = target.get('image_id')
    if image_id is None:
        return None
    if torch.is_tensor(image_id):
        return int(image_id.reshape(-1)[0].item())
    if isinstance(image_id, (list, tuple)):
        return int(image_id[0])
    return int(image_id)


def _draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, color: tuple[int, int, int]) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _load_label_font(size: int):
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _scaled_label_style(image: Image.Image) -> tuple[int, int, int, int]:
    short_side = max(1, min(image.size))
    font_size = max(14, min(42, round(short_side * 0.032)))
    padding = max(6, round(font_size * 0.45))
    line_gap = max(3, round(font_size * 0.25))
    border = max(1, round(font_size * 0.08))
    return font_size, padding, line_gap, border


def _draw_count_label(image: Image.Image, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    font_size, padding, line_gap, border = _scaled_label_style(image)
    font = _load_label_font(font_size)
    measure = ImageDraw.Draw(image)
    sizes = []
    for text, _ in lines:
        bbox = measure.textbbox((0, 0), text, font=font)
        sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))

    box_w = max((w for w, _ in sizes), default=0) + padding * 2
    box_h = sum(h for _, h in sizes) + line_gap * max(0, len(sizes) - 1) + padding * 2
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (0, 0, box_w, box_h),
        radius=max(2, padding // 2),
        fill=(255, 255, 255, 185),
        outline=(20, 20, 20, 120),
        width=border,
    )

    y = padding
    for (text, color), (_, h) in zip(lines, sizes):
        draw.text((padding, y), text, fill=(*color, 255), font=font)
        y += h + line_gap
    image.alpha_composite(overlay)


def _save_visualization_triplet(
    image_path: Path,
    output_dir: Path,
    sample_id: str,
    gt_points,
    pred_points,
    radius: int,
) -> None:
    image = Image.open(image_path).convert('RGBA')
    gt_image = image.copy()
    pred_image = image.copy()
    compare_image = image.copy()

    gt_draw = ImageDraw.Draw(gt_image)
    pred_draw = ImageDraw.Draw(pred_image)
    compare_draw = ImageDraw.Draw(compare_image)
    gt_color = (0, 180, 70)
    pred_color = (220, 40, 40)

    for x, y in gt_points:
        _draw_point(gt_draw, float(x), float(y), radius, gt_color)
        _draw_point(compare_draw, float(x), float(y), radius, gt_color)

    for x, y in pred_points:
        _draw_point(pred_draw, float(x), float(y), radius, pred_color)
        _draw_point(compare_draw, float(x), float(y), radius, pred_color)

    _draw_count_label(gt_image, [(f'GT  : {len(gt_points)}', gt_color)])
    _draw_count_label(pred_image, [(f'Pred: {len(pred_points)}', pred_color)])
    _draw_count_label(compare_image, [(f'GT  : {len(gt_points)}', gt_color), (f'Pred: {len(pred_points)}', pred_color)])

    prefix = output_dir / _safe_name(sample_id)
    gt_image.convert('RGB').save(f'{prefix}_annotation.jpg', quality=95)
    pred_image.convert('RGB').save(f'{prefix}_prediction.jpg', quality=95)
    compare_image.convert('RGB').save(f'{prefix}_comparison.jpg', quality=95)


def _should_visualize(
    target,
    selected_sample_ids: set[str],
    selected_image_ids: set[int],
    visualize_all: bool,
    rendered: int,
    max_images: int,
) -> bool:
    sample_id = str(target.get('sample_id', ''))
    image_path = Path(str(target.get('image_path', '')))
    aliases = {sample_id, image_path.name, image_path.stem}
    image_id = _target_image_id(target)

    if selected_sample_ids or selected_image_ids:
        sample_hit = bool(aliases & selected_sample_ids)
        image_hit = image_id in selected_image_ids if image_id is not None else False
        return sample_hit or image_hit
    if visualize_all:
        return max_images <= 0 or rendered < max_images
    return rendered < max(1, max_images)


def _build_visualization_loader(cfg: DictConfig, split: str) -> DataLoader:
    dataset = UnifiedCrowdDataset(
        root=Path(cfg.data.root),
        split=split,
        transform=eval_transform(),
        train=False,
        scale=False,
        patch=False,
        flip=False,
        patch_size=int(cfg.data.get('patch_size', 128)),
        patch_num=1,
        scale_min=float(cfg.data.get('train_scale_min', 1.0)),
        scale_max=float(cfg.data.get('train_scale_max', 1.0)),
        scale_min_size=int(cfg.data.get('train_scale_min_size', 128)),
        flip_prob=0.0,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        collate_fn=collate_fn_crowd,
        num_workers=int(cfg.data.get('num_workers', 0)),
    )


def _to_float(value) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _metric_value(metrics: dict, key: str) -> float | None:
    if key in metrics:
        return _to_float(metrics[key])
    for metric_key, value in metrics.items():
        if str(metric_key).startswith(f'{key}/'):
            return _to_float(value)
    return None


def _format_metric(value: float | None) -> str:
    return '-' if value is None else f'{value:.2f}'


def _make_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def rule() -> str:
        return '+' + '+'.join('-' * (width + 2) for width in widths) + '+'

    def line(cells: list[str]) -> str:
        return '| ' + ' | '.join(cell.ljust(width) for cell, width in zip(cells, widths)) + ' |'

    out = [rule(), line(headers), rule()]
    out.extend(line(row) for row in rows)
    out.append(rule())
    return '\n'.join(out)


def print_evaluation_summary(results: list[dict], cfg: DictConfig) -> None:
    metrics = results[0] if results else {}
    thresholds = [float(t) for t in cfg.eval.get('thresholds', [0.5])]
    primary_threshold = float(cfg.eval.get('primary_threshold', 0.5))
    rows = []
    for threshold in thresholds:
        suffix = str(threshold)
        mae = _metric_value(metrics, f'val/mae@{suffix}')
        rmse = _metric_value(metrics, f'val/mse@{suffix}')
        rows.append([f'{threshold:.2f}', _format_metric(mae), _format_metric(rmse)])

    primary_mae = _metric_value(metrics, 'mae')
    primary_rmse = _metric_value(metrics, 'mse')
    split = str(cfg.data.get('val_split', 'test'))

    print('\nEvaluation Results')
    print(f'Dataset    : {cfg.data.name}')
    print(f'Split      : {split}')
    print(f'Checkpoint : {cfg.eval.ckpt_path}')
    print(_make_table(['Threshold', 'MAE', 'RMSE'], rows))
    print(
        f'Primary @{primary_threshold:.2f}: '
        f'MAE={_format_metric(primary_mae)}, RMSE={_format_metric(primary_rmse)}'
    )


def visualize_predictions(model: LitCrowdModel, cfg: DictConfig) -> None:
    vis_cfg = cfg.eval.visualization
    split = str(vis_cfg.get('split', cfg.data.get('val_split', 'test')))
    dataloader = _build_visualization_loader(cfg, split=split)
    device = _select_device(cfg)
    model.to(device)
    model.eval()

    output_dir = Path(vis_cfg.get('output_dir', 'visualizations')) / str(cfg.data.name) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(vis_cfg.get('threshold', cfg.eval.get('primary_threshold', 0.5)))
    radius = int(vis_cfg.get('radius', 4))
    max_images = int(vis_cfg.get('max_images', 20))
    visualize_all = bool(vis_cfg.get('all', False))
    selected_sample_ids = set(_list_from_cfg(vis_cfg.get('sample_ids', [])))
    selected_image_ids = _int_list_from_cfg(vis_cfg.get('image_ids', []))

    rendered = 0
    requested = len(selected_sample_ids) + len(selected_image_ids)
    matched_samples: set[str] = set()
    matched_image_ids: set[int] = set()

    with torch.no_grad():
        for samples, targets in dataloader:
            target = targets[0]
            if not _should_visualize(target, selected_sample_ids, selected_image_ids, visualize_all, rendered, max_images):
                continue

            image_path = Path(str(target['image_path']))
            if not image_path.exists():
                raise FileNotFoundError(f'visualization image not found: {image_path}')

            outputs = model(samples.to(device))
            logits = outputs['pred_logits']
            if logits.dim() == 3:
                logits = logits.squeeze(-1)
            prob = torch.sigmoid(logits[0])
            points = outputs['pred_points'][0]
            width, height = _target_image_size(target)
            keep = prob > threshold
            if width > 0 and height > 0:
                keep = keep & (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)

            pred_points = points[keep].detach().cpu().numpy()
            gt_points = target['points'].detach().cpu().numpy()
            sample_id = str(target.get('sample_id', image_path.stem))
            image_id = _target_image_id(target)
            _save_visualization_triplet(image_path, output_dir, sample_id, gt_points, pred_points, radius)
            rendered += 1

            aliases = {sample_id, image_path.name, image_path.stem}
            matched_samples.update(aliases & selected_sample_ids)
            if image_id in selected_image_ids:
                matched_image_ids.add(image_id)

            if requested and len(matched_samples) + len(matched_image_ids) >= requested:
                break
            if not visualize_all and not requested and rendered >= max(1, max_images):
                break

    print(f'Visualization saved to: {output_dir} ({rendered} sample(s))')


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if not cfg.eval.ckpt_path:
        raise ValueError('Please provide eval.ckpt_path')

    datamodule = CrowdDataModule(cfg.data)
    model = LitCrowdModel.load_from_checkpoint(cfg.eval.ckpt_path, cfg=cfg)
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    results = trainer.validate(model, datamodule=datamodule, verbose=False)
    print_evaluation_summary(results, cfg)

    if bool(cfg.eval.visualization.get('enabled', False)):
        visualize_predictions(model, cfg)


if __name__ == '__main__':
    main()
