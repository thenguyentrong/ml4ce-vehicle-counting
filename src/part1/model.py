"""Part 1 detector: a frozen ImageNet backbone + a small single-class detection head.

    input 512x512  --[frozen backbone, stride 32]-->  C x 16 x 16  --[head]-->  5 x 16 x 16

The 5 output channels per cell are exactly what the task sheet asks for:

    channel 0     objectness logit          -> sigmoid -> P(a vehicle center is in this cell)
    channels 1,2  center offset (tx, ty)    -> sigmoid -> position *within* the cell, in [0, 1)
    channels 3,4  box size (tw, th)         -> sigmoid -> size as a fraction of the image

Everything is squashed through a sigmoid, so no anchor boxes are needed. That is a legitimate
simplification for one class of similarly-sized objects; it would not survive on COCO, where a
single cell must predict objects spanning three orders of magnitude in area.

The backbone is frozen (`config.FREEZE_BACKBONE`): we train ~1.2M head parameters, not 11M
backbone parameters, on 801 training images. Unfreezing `layer4` is one of the ablations.

Author: Vinh Nguyen
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

import config


class VehicleDetector(nn.Module):
    """Frozen pretrained backbone + 5-channel convolutional detection head."""

    def __init__(
        self,
        backbone: str = config.BACKBONE,
        freeze: bool = config.FREEZE_BACKBONE,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, out_channels = self._build_backbone(backbone, pretrained)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Freezing the weights is not enough: BatchNorm layers keep updating their
            # running mean/var in train() mode, which quietly shifts the features under the
            # head. eval() on the backbone stops that. This is the classic silent bug in
            # "frozen backbone" setups.
            self.backbone.eval()
        self.frozen = freeze

        self.head = nn.Sequential(
            nn.Conv2d(out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5, kernel_size=1),
        )

        # Bias init for the objectness channel: with ~2 positive cells out of 256, a
        # zero-init head starts by predicting P(object)=0.5 everywhere and the loss spends
        # its first epochs just learning "almost everything is background". Starting the
        # bias at logit(0.01) skips that. (Trick from the RetinaNet/focal-loss paper.)
        nn.init.constant_(self.head[-1].bias[0], -4.6)  # sigmoid(-4.6) ~= 0.01

    @staticmethod
    def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Return (feature extractor with output stride 32, number of output channels)."""
        if name == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.resnet18(weights=weights)
            # Everything except avgpool and fc -> output is [B, 512, H/32, W/32].
            body = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4,
            )
            return body, 512

        if name == "mobilenet_v3_large":
            weights = (
                torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
            net = torchvision.models.mobilenet_v3_large(weights=weights)
            return net.features, 960  # also stride 32

        raise ValueError(f"unknown backbone {name!r} (expected resnet18 or mobilenet_v3_large)")

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode even when the model is set to train()."""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 512, 512) -> raw logits (B, 5, 16, 16). No sigmoid here.

        The sigmoid is deliberately left to the loss (BCEWithLogits is numerically stable)
        and to `infer.decode_predictions`. Applying it twice is a bug that shows up as a
        model that trains fine but predicts boxes squashed toward the image center.
        """
        if self.frozen:
            with torch.no_grad():  # no gradients flow into the backbone; also saves memory
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        return self.head(feats)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
