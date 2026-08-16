"""
Frozen pretrained backbone + a small trainable detection head.

Backbone: MobileNetV3-Small by default (fast; swap in ResNet18/34 if you want).
Head: 1x1 conv stack mapping backbone feature channels -> 5 outputs per grid cell
      (objectness, offset_x, offset_y, w, h).
"""
import torch
import torch.nn as nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self, name="mobilenet_v3_small"):
        super().__init__()
        if name == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            self.features = net.features
            self.out_channels = 576  # last feature map channels for mobilenet_v3_small
        elif name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc
            self.out_channels = 512
        else:
            raise ValueError(f"Unknown backbone {name}")

        # freeze all backbone weights -- we only train the head
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x)  # (B, C, H', W')


class DetectionHead(nn.Module):
    def __init__(self, in_channels, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 5, kernel_size=1),  # objectness + 4 box params
        )

    def forward(self, feat):
        out = self.net(feat)              # (B, 5, H', W')
        return out.permute(0, 2, 3, 1)     # (B, H', W', 5) -- matches target layout


class VehicleDetector(nn.Module):
    def __init__(self, backbone_name="mobilenet_v3_small"):
        super().__init__()
        self.backbone = Backbone(backbone_name)
        self.head = DetectionHead(self.backbone.out_channels)

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.head(feat)


if __name__ == "__main__":
    model = VehicleDetector()
    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    print("output shape:", out.shape)  # expect (2, 16, 16, 5) for stride-32 backbones
