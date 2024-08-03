from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import BasicBlock
from transformers import AutoBackbone

from cobb_angle.dsnt import spatial_softmax_2d


def make_residual_blocks(
    inplanes: int,
    planes: int,
    blocks: int,
    stride: int = 1,
    dilation: int = 1,
    downsample: Optional[nn.Module] = None,
    norm_layer: Optional[nn.Module] = None,
) -> nn.Sequential:
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
            norm_layer(planes),
        )

    layers = []

    layers.append(
        BasicBlock(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            dilation=dilation,
            norm_layer=norm_layer,
        )
    )

    for _ in range(1, blocks):
        layers.append(
            BasicBlock(
                planes,
                planes,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )

    return nn.Sequential(*layers)


def make_backbone(model: str) -> nn.Module:
    weights_name: dict[str, str] = {
        "resnet18": "microsoft/resnet-18",
        "resnet34": "microsoft/resnet-34",
        "resnet50": "microsoft/resnet-50",
    }

    return AutoBackbone.from_pretrained(
        weights_name[model], out_features=["stage1", "stage2", "stage3", "stage4"]
    )


@dataclass(frozen=True)
class CascadedPyramidNetworkConfig:
    backbone_model: str = "resnet18"
    landmarks_count: int = 68


class CascadedPyramidNetwork(nn.Module):
    def __init__(self, config: CascadedPyramidNetworkConfig):
        super().__init__()
        self.backbone = make_backbone(config.backbone_model)
        self.res1 = make_residual_blocks(512, 512, blocks=1)
        self.upsamplex2 = partial(
            F.interpolate, scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample_to = partial(F.interpolate, mode="bilinear", align_corners=True)

        self.res2 = make_residual_blocks(512, 256, blocks=1)
        self.cascade1 = make_residual_blocks(256, 256, blocks=3)

        self.res3 = make_residual_blocks(256, 128, blocks=1)
        self.cascade2 = make_residual_blocks(128, 128, blocks=2)

        self.res4 = make_residual_blocks(128, 64, blocks=1)
        self.cascade3 = make_residual_blocks(64, 64, blocks=1)

        self.res5 = make_residual_blocks(64, 64, blocks=1)

        self.output1 = make_residual_blocks(64, config.landmarks_count, blocks=3)
        self.output2 = make_residual_blocks(512, config.landmarks_count, blocks=3)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x).feature_maps
        x4 = self.res1(x4)

        x = self.upsamplex2(self.res2(x4)) + x3
        cascaded1 = self.cascade1(x)

        x = self.upsamplex2(self.res3(x)) + x2
        cascaded2 = self.cascade2(x)

        x = self.upsamplex2(self.res4(x)) + x1
        cascaded3 = self.cascade3(x)

        cascaded4 = self.upsamplex2(self.res5(x))
        stage1 = self.output1(cascaded4)

        _, _, height, width = cascaded4.shape

        cascaded1 = self.upsample_to(cascaded1, size=(height, width))
        cascaded2 = self.upsample_to(cascaded2, size=(height, width))
        cascaded3 = self.upsample_to(cascaded3, size=(height, width))

        stage2 = torch.cat((cascaded1, cascaded2, cascaded3, cascaded4), dim=1)
        stage2 = self.output2(stage2)

        stage1 = spatial_softmax_2d(stage1)
        stage2 = spatial_softmax_2d(stage2)

        return stage1, stage2


if __name__ == "__main__":
    fake_image = torch.randn(1, 3, 256, 256)
    config = CascadedPyramidNetworkConfig()
    model = CascadedPyramidNetwork(config)
    output = model(fake_image)
    print(f"{output[0].shape = }")
    print(f"{output[1].shape = }")
