from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsnt import dsnt, render_gaussian_2d


class WingLoss(nn.Module):
    def __init__(self, w: int = 15, eps: int = 3) -> None:
        super().__init__()
        self.w = w
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        condition_term = torch.abs(diff) - self.w
        one = torch.ones_like(target)
        loss_templete = torch.zeros_like(target)

        loss_templete[condition_term < 0] = (
            self.w * torch.log(one + torch.abs(diff) / self.eps)
        )[condition_term < 0]

        loss_templete[condition_term >= 0] = (
            condition_term + self.w * torch.log(one + self.w / self.eps)
        )[condition_term >= 0]

        return loss_templete.mean()


class JSDivLoss2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mixture_distribution = 0.5 * (input + target)
        term1 = F.kl_div(input.log(), mixture_distribution, reduction="batchmean")
        term2 = F.kl_div(target.log(), mixture_distribution, reduction="batchmean")

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

        target[..., 0] = target[..., 0] / (height / 2) - 1
        target[..., 1] = target[..., 1] / (width / 2) - 1

        stage1 = input[0]
        stage2 = input[1]

        target_heatmap = render_gaussian_2d(target.to("cpu"), size=(height, width)).to(
            dtype=stage1.dtype, device=stage1.device
        )

        return (
            self.regularization_coefficient * self.js_div(stage2, target_heatmap)
            + self.wing_loss(dsnt(stage1), target)
            + self.wing_loss(dsnt(stage2), target)
        )
