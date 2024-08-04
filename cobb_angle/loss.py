from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsnt import dsnt


class WingLoss(nn.Module):
    def __init__(
        self, width: int = 15, curvature: int = 3, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.width = width
        self.curvature = curvature
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff_abs = (targets - outputs).abs()
        loss = diff_abs.clone()

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width

        loss[idx_smaller] = self.width * torch.log(
            1 + diff_abs[idx_smaller] / self.curvature
        )

        c = self.width - self.width * math.log(1 + self.width / self.curvature)
        loss[idx_bigger] = loss[idx_bigger] - c

        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class JSDivLoss2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.clamp(input, min=1e-32)
        target = torch.clamp(target, min=1e-32)
        batch_size, channels, _, _ = input.shape
        mixture_distribution = 0.5 * (input + target)
        term1 = F.kl_div(input.log(), mixture_distribution, reduction="sum") / (
            batch_size * channels
        )
        term2 = F.kl_div(target.log(), mixture_distribution, reduction="sum") / (
            batch_size * channels
        )

        return 0.5 * term1 + 0.5 * term2


class WingLossWithRegularization(nn.Module):
    def __init__(self, regularization_coefficient: float = 1.0) -> None:
        super().__init__()
        self.regularization_coefficient = regularization_coefficient
        self.wing_loss = WingLoss()
        self.js_div = JSDivLoss2D()

    def forward(
        self, input: Tuple[torch.Tensor, ...], target: torch.Tensor
    ) -> torch.Tensor:
        _, _, height, width = input[0].shape

        target_resized = torch.zeros_like(target)
        target_resized[..., 0] = 2 * target[..., 0] / (width - 1) - 1
        target_resized[..., 1] = 2 * target[..., 1] / (height - 1) - 1

        stage1, stage2 = input

        return (
            self.wing_loss(dsnt(stage1), target_resized)
            + self.wing_loss(dsnt(stage2), target_resized)
            # + self.regularization_coefficient * self.js_div(stage2, target_heatmap)
        )
