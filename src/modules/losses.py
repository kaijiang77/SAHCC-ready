import torch
import torch.nn.functional as F
from torch import nn


def _is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _world_size() -> int:
    return torch.distributed.get_world_size() if _is_dist_initialized() else 1


class CrowdCriterion(nn.Module):
    def __init__(
        self,
        matcher: nn.Module,
        point_loss_coef: float,
        cls_pos_weight: float = 1.0,
        cls_neg_weight: float = 0.5,
    ):
        super().__init__()
        if cls_pos_weight < 0 or cls_neg_weight < 0:
            raise ValueError("Classification sample weights must be non-negative")
        self.matcher = matcher
        self.weight_dict = {"loss_ce": 1.0, "loss_points": float(point_loss_coef)}
        self.cls_pos_weight = float(cls_pos_weight)
        self.cls_neg_weight = float(cls_neg_weight)

    @staticmethod
    def _src_perm_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _num_gt(self, targets, device: torch.device):
        num_gt = sum(len(t["labels"]) for t in targets)
        num_gt = torch.as_tensor([num_gt], dtype=torch.float32, device=device)
        if _is_dist_initialized():
            torch.distributed.all_reduce(num_gt, op=torch.distributed.ReduceOp.SUM)
        return torch.clamp(num_gt / _world_size(), min=1.0)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_points = self._num_gt(targets, outputs["pred_logits"].device)

        logits = outputs["pred_logits"].squeeze(-1)  # [B, N]
        b, n = logits.shape
        idx = self._src_perm_idx(indices)
        target_fg = torch.zeros((b, n), dtype=torch.float32, device=logits.device)
        target_fg[idx] = 1.0
        sample_w = torch.full_like(target_fg, self.cls_neg_weight)
        sample_w = sample_w + (self.cls_pos_weight - self.cls_neg_weight) * target_fg
        loss_ce = F.binary_cross_entropy_with_logits(logits, target_fg, weight=sample_w, reduction="mean")

        total_matches = sum(len(src) for src, _ in indices)
        if total_matches == 0:
            loss_points = outputs["pred_points"].sum() * 0.0
        else:
            src_points = outputs["pred_points"][idx]
            tgt_points = torch.cat([t["points"][j] for t, (_, j) in zip(targets, indices)], dim=0)
            loss_points = F.mse_loss(src_points, tgt_points, reduction="none").sum() / num_points.squeeze(0)

        return {"loss_ce": loss_ce, "loss_points": loss_points}
