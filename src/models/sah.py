import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcherCrowd(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_point: float = 1.0,
        point_source: str = "anchor_points",
        k: float = 1.2,
        alpha_min: float = 0.03,
    ):
        super().__init__()
        if point_source not in {"anchor_points", "pred_points"}:
            raise ValueError(f"Unsupported point_source: {point_source}")
        if cost_class == 0 and cost_point == 0:
            raise ValueError("cost_class and cost_point cannot both be 0")

        self.cost_class = float(cost_class)
        self.cost_point = float(cost_point)
        self.point_source = point_source
        self.k = float(k)
        self.alpha_min = float(alpha_min)

    @torch.no_grad()
    def forward(self, outputs, targets):
        logits = outputs["pred_logits"]
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        prob_fg = torch.sigmoid(logits)

        src_points_all = outputs[self.point_source]
        indices = []
        for i in range(prob_fg.shape[0]):
            src_prob = prob_fg[i]
            src_points = src_points_all[i]
            tgt_points = targets[i]["points"]
            tgt_mean = targets[i]["mean_mnn"].float().clamp(min=1e-6)

            q = src_points.shape[0]
            m = tgt_points.shape[0]
            if q == 0 or m == 0:
                indices.append(
                    (
                        torch.empty((0,), dtype=torch.int64, device=src_points.device),
                        torch.empty((0,), dtype=torch.int64, device=src_points.device),
                    )
                )
                continue

            cost_class = -src_prob[:, None].expand(q, m)
            cost_point = torch.cdist(src_points, tgt_points, p=2)
            point_weight = torch.clamp(self.k / tgt_mean, min=self.alpha_min)[None, :]
            cost = self.cost_class * cost_class + self.cost_point * (cost_point * point_weight)

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=src_points.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=src_points.device),
                )
            )
        return indices


def build_matcher_crowd(args):
    return HungarianMatcherCrowd(
        cost_class=args.set_cost_class,
        cost_point=args.set_cost_point,
        point_source=args.matcher_point_source,
        k=args.matcher_k,
        alpha_min=args.matcher_alpha_min,
    )
