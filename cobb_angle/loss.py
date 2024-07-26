import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsnt import dsnt, render_gaussian_2d

__all__ = ["LossAll"]


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        pred = self._tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class WingLoss(nn.Module):
    def __init__(self, w: int = 15, eps: int = 3) -> None:
        super().__init__()
        self.w = w
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        condition_term = torch.clamp(torch.abs(diff) - self.w, min=0)

        return condition_term + self.w * torch.log(1 + self.w / self.eps)


class JSDivLoss2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mixture_distribution = 0.5 * (input + target)
        term1 = F.kl_div(input.log(), mixture_distribution, reduction="none").sum(
            dim=(-1, -2)
        )
        term2 = F.kl_div(target.log(), mixture_distribution, reduction="none").sum(
            dim=(-1, -2)
        )

        return 0.5 * term1 + 0.5 * term2


class WingLossWithRegularization(nn.Module):
    def __init__(self, regularization_coefficient: float = 1.0) -> None:
        super().__init__()
        self.regularization_coefficient = regularization_coefficient

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        wing_loss = WingLoss()
        js_div = JSDivLoss2D()

        target_heatmap = render_gaussian_2d(target)

        return wing_loss(
            dsnt(input), target
        ) + self.regularization_coefficient * js_div(input, target_heatmap)


class LossAll(nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_off = RegL1Loss()
        self.L_wh = RegL1Loss()

    def forward(self, pr_decs, gt_batch):
        hm_loss = self.L_hm(pr_decs["hm"], gt_batch["hm"])
        wh_loss = self.L_wh(
            pr_decs["wh"], gt_batch["reg_mask"], gt_batch["ind"], gt_batch["wh"]
        )
        off_loss = self.L_off(
            pr_decs["reg"], gt_batch["reg_mask"], gt_batch["ind"], gt_batch["reg"]
        )
        loss_dec = hm_loss + off_loss + wh_loss
        return loss_dec
