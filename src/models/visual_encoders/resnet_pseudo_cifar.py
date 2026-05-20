"""Standard ResNet18 with PseudoNeuron activations, adapted for CIFAR-100.

Differs from SEW-ResNet (sew_resnet_pseudo.py) in two ways:
  1. CIFAR stem: 3×3 conv, stride=1, no max-pool (keeps 32×32 spatial resolution).
  2. Downsample paths are plain SeqToANNContainer(conv1x1, bn) — no extra
     PseudoNeuron after the skip projection (unlike SEW-ResNet).

Residual connections use pad_and_add, identical to SEW-ResNet with connect_f="ADD".

Usage:
    from src.models.visual_encoders.resnet_pseudo_cifar import resnet18_pseudo_cifar
    model = resnet18_pseudo_cifar(num_classes=100, init_T=8.0, min_T=1, max_T=16)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .pseudo_neuron import PseudoNeuron, SeqToANNContainer, pad_and_add

__all__ = ["ResNetPseudo", "resnet18_pseudo_cifar"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        **neuron_kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = SeqToANNContainer(conv3x3(inplanes, planes, stride), nn.BatchNorm2d(planes))
        self.sn1 = PseudoNeuron(**neuron_kwargs)
        self.conv2 = SeqToANNContainer(conv3x3(planes, planes), nn.BatchNorm2d(planes))
        self.sn2 = PseudoNeuron(**neuron_kwargs)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.sn1(self.conv1(x))
        out = self.conv2(out)                  # conv2 receives spikes from sn1, no activation yet
        if self.downsample is not None:
            identity = self.downsample(x)
        out = pad_and_add(out, identity)
        return self.sn2(out)                   # sn2 gates the whole summed signal


class ResNetPseudo(nn.Module):
    """ResNet with PseudoNeuron activations, CIFAR-100 variant.

    Input:  [B, 3, 32, 32]
    Output: [B, num_classes] logits
    """

    def __init__(
        self,
        block: type,
        layers: list[int],
        num_classes: int = 100,
        norm_layer: type = nn.BatchNorm2d,
        **neuron_kwargs,
    ) -> None:
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64

        # CIFAR stem: 3×3, stride=1, no max-pool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.sn1 = PseudoNeuron(**neuron_kwargs)

        self.layer1 = self._make_layer(block, 64,  layers[0], **neuron_kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **neuron_kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **neuron_kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **neuron_kwargs)

        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
        **neuron_kwargs,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Plain conv+bn downsample — no PseudoNeuron (unlike SEW-ResNet)
            downsample = SeqToANNContainer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, **neuron_kwargs)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, **neuron_kwargs))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features [B, C] without the classification head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(0)   # inject T dimension: [1, B, C, H, W]
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)         # [T, B, C, 1, 1]
        x = torch.flatten(x, 2)     # [T, B, C]
        return x.mean(dim=0)        # [B, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))


def resnet18_pseudo_cifar(**kwargs) -> ResNetPseudo:
    return ResNetPseudo(BasicBlock, [2, 2, 2, 2], **kwargs)
