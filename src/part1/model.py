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
        stride: int = config.STRIDE,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.stride = stride
        self.backbone, out_channels = self._build_backbone(backbone, pretrained, stride)

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
    def _build_backbone(name: str, pretrained: bool, stride: int = 32) -> tuple[nn.Module, int]:
        """Return (feature extractor with the requested output stride, output channel count).

        stride=32 -> 16x16 grid, exactly what the task sheet prescribes.
        stride=16 -> 32x32 grid. Four times as many cells, each covering 16 px instead of 32.
                     This exists because the error analysis showed *every* missed vehicle is a
                     small one (recall 0.77 for boxes under 2.5k px^2, 1.00 for boxes over
                     5k px^2): at stride 32 a distant car spans barely one cell, so there is
                     nothing for the head to localise. This is a resolution problem, and
                     resolution is the only thing that fixes it.
        """
        if stride not in (16, 32):
            raise ValueError(f"stride must be 16 or 32, got {stride}")

        if name == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.resnet18(weights=weights)
            stem = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3]
            if stride == 16:
                return nn.Sequential(*stem), 256  # through layer3 -> stride 16, 256 ch
            return nn.Sequential(*stem, net.layer4), 512  # + layer4 -> stride 32, 512 ch

        if name == "mobilenet_v3_large":
            weights = (
                torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
            net = torchvision.models.mobilenet_v3_large(weights=weights)
            if stride == 32:
                return net.features, 960

            # Find where MobileNet's feature stack reaches stride 16 by probing it, rather
            # than hard-coding a block index that a torchvision update could silently shift.
            with torch.no_grad():
                x = torch.zeros(1, 3, 256, 256)
                cut, channels = None, None
                for k, block in enumerate(net.features):
                    x = block(x)
                    if x.shape[-1] == 256 // 16:  # first block at stride 16
                        cut, channels = k + 1, x.shape[1]
                    elif x.shape[-1] < 256 // 16:  # gone past it, into stride 32
                        break
            if cut is None:
                raise RuntimeError("could not locate a stride-16 stage in mobilenet_v3_large")
            return nn.Sequential(*list(net.features)[:cut]), channels

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
