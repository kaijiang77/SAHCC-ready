from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.trainer.states import TrainerFn
from hydra.core.hydra_config import HydraConfig

from src.models.backbone import build_backbone
from src.models.matcher import build_matcher_crowd
from src.models.sahcc import P2PNet
from src.modules.losses import CrowdCriterion
from src.modules.metrics import THRESHOLDS, summarize_count_metrics, update_count_errors


class LitCrowdModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg
        backbone = build_backbone(cfg.model.backbone)
        self.model = P2PNet(backbone, row=cfg.model.row, line=cfg.model.line)
        matcher = build_matcher_crowd(cfg.matcher)
        self.criterion = CrowdCriterion(
            matcher=matcher,
            point_loss_coef=cfg.model.point_loss_coef,
            cls_pos_weight=cfg.model.get('cls_pos_weight', 1.0),
            cls_neg_weight=cfg.model.get('cls_neg_weight', 0.5),
        )
        self.best_mae = float('inf')
        self.best_mse = float('inf')
        self.val_abs_err = {t: [] for t in THRESHOLDS}
        self.val_sq_err = {t: [] for t in THRESHOLDS}

    def forward(self, samples):
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        batch_size = samples.size(0)
        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if 'loss_ce' in loss_dict:
            loss_ce = loss_dict['loss_ce'] * weight_dict.get('loss_ce', 1.0)
            self.log('train/loss_ce', loss_ce, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if 'loss_points' in loss_dict:
            loss_points = loss_dict['loss_points'] * weight_dict.get('loss_points', 1.0)
            self.log(
                'train/loss_points',
                loss_points,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        return loss

    def on_validation_epoch_start(self):
        self.val_abs_err = {t: [] for t in THRESHOLDS}
        self.val_sq_err = {t: [] for t in THRESHOLDS}

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)

        logits = outputs['pred_logits']
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        prob = torch.sigmoid(logits)
        gt_cnt = torch.as_tensor([t['points'].shape[0] for t in targets], device=prob.device, dtype=torch.long)

        update_count_errors(self.val_abs_err, self.val_sq_err, prob, gt_cnt)

    def on_validation_epoch_end(self):
        results = summarize_count_metrics(self.val_abs_err, self.val_sq_err)
        mae_msg = []
        mse_msg = []
        for thr in THRESHOLDS:
            mae, rmse = results[thr]
            mse = rmse
            self.log(f'val/mae@{thr}', mae)
            self.log(f'val/mse@{thr}', mse)
            mae_msg.append(f'@{thr}:{mae:.2f}')
            mse_msg.append(f'@{thr}:{mse:.2f}')

        mae_05, rmse_05 = results[0.5]
        mse_05 = rmse_05
        standalone_eval = getattr(self.trainer.state, 'fn', None) == TrainerFn.VALIDATING

        self.log('mae', mae_05, prog_bar=True)
        self.log('mse', mse_05, prog_bar=True)
        if not standalone_eval:
            if mae_05 < self.best_mae:
                self.best_mae = mae_05
                self.best_mse = mse_05
            self.log('best_mae', self.best_mae, prog_bar=True)
            self.log('best_mse', self.best_mse)

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        log_path = output_dir / f'{HydraConfig.get().job.name}.log'
        with log_path.open('a', encoding='utf-8') as f:
            if standalone_eval:
                f.write(
                    f'Evaluation\n'
                    f'  Val MAE   : {" | ".join(mae_msg)}\n'
                    f'  Val MSE   : {" | ".join(mse_msg)}\n\n'
                )
            else:
                metrics = self.trainer.callback_metrics
                loss_t = metrics.get('train/loss')
                loss_ce_t = metrics.get('train/loss_ce')
                loss_points_t = metrics.get('train/loss_points')
                loss = loss_t.item() if loss_t is not None else float('nan')
                loss_ce = loss_ce_t.item() if loss_ce_t is not None else float('nan')
                loss_points = loss_points_t.item() if loss_points_t is not None else float('nan')
                f.write(
                    f'Epoch {self.current_epoch}\n'
                    f'  Train Loss: total={loss:.6f}, ce={loss_ce:.6f}, points={loss_points:.6f}\n'
                    f'  Val MAE   : {" | ".join(mae_msg)}\n'
                    f'  Val MSE   : {" | ".join(mse_msg)}\n'
                    f'  Best@0.5  : mae={self.best_mae:.2f}, mse={self.best_mse:.2f}\n\n'
                )

    def configure_optimizers(self):
        non_backbone = [p for n, p in self.model.named_parameters() if 'backbone' not in n and p.requires_grad]
        backbone = [p for n, p in self.model.named_parameters() if 'backbone' in n and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {'params': non_backbone, 'lr': self.cfg.optimizer.lr},
                {'params': backbone, 'lr': self.cfg.optimizer.lr_backbone},
            ],
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.scheduler.lr_drop)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
