from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import BasicBlock
from transformers import AutoBackbone


def make_residual_blocks(
    inplanes: int,
    planes: int,
    blocks: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    base_width: int = 64,
    norm_layer: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers = []
    layers.append(
        BasicBlock(inplanes, planes, stride, groups, base_width, dilation, norm_layer)
    )
    for _ in range(1, blocks):
        layers.append(
            BasicBlock(
                planes,
                planes,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )

    return nn.Sequential(*layers)


def make_backbone(model: str):
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
    heatmap_regularization: Optional[str] = None


class CascadedPyramidNetwork(nn.Module):
    def __init__(self, config: CascadedPyramidNetworkConfig):
        super().__init__()
        self.backbone = make_backbone(config.backbone_model)
        self.res1 = make_residual_blocks(512, 512, blocks=1)
        self.upsamplex2 = F.interpolate(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample_to = partial(F.interpolate, mode="bilinear", align_corners=True)

        self.res2 = make_residual_blocks(512, 256, blocks=1)
        self.cascade1 = make_residual_blocks(256, 256, blocks=3)

        self.res3 = make_residual_blocks(256, 128, blocks=1)
        self.cascade2 = make_residual_blocks(128, 128, blocks=2)

        self.res4 = make_residual_blocks(128, 64, blocks=1)
        self.cascade3 = make_residual_blocks(64, 4, blocks=1)

        self.res5 = make_residual_blocks(64, 64, blocks=1)

        self.output1 = make_residual_blocks(64, config.landmarks_count, blocks=3)
        self.output2 = make_residual_blocks(512, config.landmarks_count, blocks=3)

        self.dsnt = None  # TODO: Implement this bad boy


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

        _, _, height, width = cascaded4.size()

        cascaded1 = self.upsample_to(cascaded1, size=(height, width))
        cascaded2 = self.upsample_to(cascaded2, size=(height, width))
        cascaded3 = self.upsample_to(cascaded3, size=(height, width))

        stage2 = torch.cat((cascaded1, cascaded2, cascaded3, cascaded4), dim=1)
        stage2 = self.output2(stage2)

        # stage1 = self.dsnt(stage1)
        # stage2 = self.dsnt(stage2)

        return stage1, stage2



if __name__ == "__main__":
    print(f"hello {torch.randn(2, 3)}")
