"""Image backbones. Every backbone exposes `.features` (conv feature map) and `.out_channels`."""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """4 conv blocks; 3 poolings -> an (H/8 x W/8) feature map (8x8 for 64px input)."""

    def __init__(self, in_ch: int = 1, widths=(16, 32, 64, 64)):
        super().__init__()
        layers, c = [], in_ch
        for i, w in enumerate(widths):
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=False)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            c = w
        self.features = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.features(x)


class _TorchvisionBackbone(nn.Module):
    def __init__(self, features: nn.Module, out_channels: int):
        super().__init__()
        self.features = features
        self.out_channels = out_channels

    def forward(self, x):
        return self.features(x)


def _one_channel_conv(old: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
    return new


def make_backbone(name: str = "small_cnn", pretrained: bool = False) -> nn.Module:
    """small_cnn | resnet18 | densenet121 | efficientnet_b0.
    pretrained=True downloads ImageNet weights (needs internet) and folds RGB filters into 1 channel."""
    import torchvision.models as tvm

    if name == "small_cnn":
        return SmallCNN()
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        conv1 = _one_channel_conv(m.conv1, pretrained)
        feats = nn.Sequential(conv1, m.bn1, nn.ReLU(inplace=False), m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        return _TorchvisionBackbone(feats, 512)
    if name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.features.conv0 = _one_channel_conv(m.features.conv0, pretrained)
        return _TorchvisionBackbone(nn.Sequential(m.features, nn.ReLU(inplace=False)), 1024)
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.features[0][0] = _one_channel_conv(m.features[0][0], pretrained)
        return _TorchvisionBackbone(m.features, 1280)
    raise ValueError(f"unknown backbone {name!r}")
