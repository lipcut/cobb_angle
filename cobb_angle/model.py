import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

__all__ = ["BigBrainNet"]


def make_fc(
    in_channels: int,
    mid_channels: int,
    num_classes: int,
    kernel_size: int,
    final_kernel_size: int,
    padding: int,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            mid_channels,
            num_classes,
            kernel_size=final_kernel_size,
            padding=final_kernel_size // 2,
            bias=True,
        ),
    )


def fill_fc_weight(layers: nn.Sequential):
    for layer in layers:
        if isinstance(layer, nn.Conv2d) and layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class SkipConnection(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, normalization: str = "batch_norm"
    ):
        # vaild normalization methods: batch_norm, group_norm, instance_norm
        super().__init__()
        norm_methods = {
            "batch_norm": nn.BatchNorm2d(out_channels),
            "group_norm": nn.GroupNorm(num_groups=32, num_channels=out_channels),
            "instance_norm": nn.InstanceNorm2d(num_features=out_channels),
        }
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            norm_methods.get(normalization, nn.Identity()),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_main, x_side):
        x_main = self.layer1(F.interpolate(x_main, x_side.shape[2:]))
        outputs = self.layer2(torch.cat((x_side, x_main), dim=1))
        return outputs


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        final_kernel_size: int,
        num_classes: int,
    ):
        super().__init__()
        self.layer1 = SkipConnection(512, 256)
        self.layer2 = SkipConnection(256, 128)
        self.layer3 = SkipConnection(128, 64)
        self.heat_map_fc = make_fc(
            in_channels,
            mid_channels,
            num_classes,
            kernel_size=3,
            final_kernel_size=final_kernel_size,
            padding=1,
        )
        self.center_offset_fc = make_fc(
            in_channels,
            mid_channels,
            2 * num_classes,
            kernel_size=3,
            final_kernel_size=final_kernel_size,
            padding=1,
        )

        self.corner_offset_fc = make_fc(
            in_channels,
            mid_channels,
            4 * 2,
            kernel_size=7,
            final_kernel_size=7,
            padding=7 // 2,
        )

        self.heat_map_fc[-1].bias.data.fill_(-2.19)
        self.heat_map_fc.append(nn.Sigmoid())
        fill_fc_weight(self.center_offset_fc)
        fill_fc_weight(self.corner_offset_fc)

        def forward(self, x):
            combination1 = self.layer1(x[-1], x[-2])
            combination2 = self.layer2(combination1, x[-3])
            combination3 = self.layer3(combination2, x[-4])
            outputs = {
                "hm": self.heat_map_fc(combination3),
                "reg": self.center_offset_fc(combination3),
                "wh": self.corner_offset_fc(combination3),
            }

            return outputs


class BigBrainNet(nn.Module):
    def __init__(
        self,
        weights: str,
        down_ratio: int = 4,
        mid_channels: int = 256,
        num_classes: int = 1,
        final_kernel_size: int = 1,
    ):
        super().__init__()
        assert down_ratio in (2, 4, 6, 8), "The only down ratios supported: 2, 4, 6, 8"
        channels = (3, 64, 64, 128, 256, 512)
        index = int(np.log2(down_ratio))
        self.encoder = models.resnet34(weights=weights)
        self.decoder = Decoder(
            channels[index], mid_channels, final_kernel_size, num_classes
        )

    def forward(self, x):
        x = self.encoder(x)
        outputs = self.decoder(x)
        return outputs
