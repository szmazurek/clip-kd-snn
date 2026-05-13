"""SEW-ResNet built on PseudoNeuron for ImageNet classification.

Ported from PseudoSNN/models/noisy/sew_resnet.py (CVPR 2025).
Key changes vs. the original:
  - CostCalculator and FLOP-counting hooks removed; penalty is computed
    externally in the Lightning module by iterating PseudoNeuron.get_timesteps().
  - forward() always returns plain logits (no (logits, cost) tuple).
  - Imports PseudoNeuron / SeqToANNContainer / pad_and_add from pseudo_neuron.py.

Supported architectures:
  sew_resnet18  - BasicBlock [2, 2, 2, 2]
  sew_resnet34  - BasicBlock [3, 4, 6, 3]
  sew_resnet50  - Bottleneck [3, 4, 6, 3]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .pseudo_neuron import PseudoNeuron, SeqToANNContainer, pad_and_add

__all__ = ["SEWResNet", "sew_resnet18", "sew_resnet34", "sew_resnet50"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: type = nn.BatchNorm2d,
        connect_f: str = "ADD",
        **neuron_kwargs,
    ) -> None:
        super().__init__()
        self.connect_f = connect_f

        self.conv1 = SeqToANNContainer(conv3x3(inplanes, planes, stride), norm_layer(planes))
        self.sn1 = PseudoNeuron(**neuron_kwargs)

        self.conv2 = SeqToANNContainer(conv3x3(planes, planes), norm_layer(planes))
        self.sn2 = PseudoNeuron(**neuron_kwargs)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return _connect(out, identity, self.connect_f)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: type = nn.BatchNorm2d,
        connect_f: str = "ADD",
        **neuron_kwargs,
    ) -> None:
        super().__init__()
        self.connect_f = connect_f
        width = planes  # groups=1, base_width=64

        self.conv1 = SeqToANNContainer(conv1x1(inplanes, width), norm_layer(width))
        self.sn1 = PseudoNeuron(**neuron_kwargs)

        self.conv2 = SeqToANNContainer(conv3x3(width, width, stride), norm_layer(width))
        self.sn2 = PseudoNeuron(**neuron_kwargs)

        self.conv3 = SeqToANNContainer(conv1x1(width, planes * self.expansion), norm_layer(planes * self.expansion))
        self.sn3 = PseudoNeuron(**neuron_kwargs)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        out = self.sn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return _connect(out, identity, self.connect_f)


def _connect(out: torch.Tensor, identity: torch.Tensor, connect_f: str) -> torch.Tensor:
    if connect_f == "ADD":
        return pad_and_add(out, identity)
    elif connect_f == "AND":
        return out * identity
    elif connect_f == "IAND":
        return identity * (1.0 - out)
    else:
        raise NotImplementedError(f"connect_f={connect_f!r}")


def _zero_init_residual(net: nn.Module, connect_f: str) -> None:
    """Zero-initialise the last BN in each residual branch for stable early training."""
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3[1].weight, 0)
            if connect_f == "AND":
                nn.init.constant_(m.conv3[1].bias, 1)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2[1].weight, 0)
            if connect_f == "AND":
                nn.init.constant_(m.conv2[1].bias, 1)


# ---------------------------------------------------------------------------
# SEWResNet
# ---------------------------------------------------------------------------

class SEWResNet(nn.Module):
    """Spatially-Effective-Weight ResNet with PseudoNeuron spiking activations.

    Input:  [B, 3, H, W]  (standard ImageNet batch, no T dimension)
    Output: [B, num_classes]  logits
    """

    def __init__(
        self,
        block: type,
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        norm_layer: type = nn.BatchNorm2d,
        connect_f: str = "ADD",
        **neuron_kwargs,
    ) -> None:
        super().__init__()
        self.connect_f = connect_f
        self._norm_layer = norm_layer
        self.inplanes = 64

        # Stem: standard conv (no spiking), then inject T=1 dimension
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.sn1 = PseudoNeuron(**neuron_kwargs)
        self.maxpool = SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64,  layers[0], connect_f=connect_f, **neuron_kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, connect_f=connect_f, **neuron_kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, connect_f=connect_f, **neuron_kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, connect_f=connect_f, **neuron_kwargs)

        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()
        if zero_init_residual:
            _zero_init_residual(self, connect_f)

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
        connect_f: str = "ADD",
        **neuron_kwargs,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                PseudoNeuron(**neuron_kwargs),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample,
                  norm_layer=norm_layer, connect_f=connect_f, **neuron_kwargs)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes,
                      norm_layer=norm_layer, connect_f=connect_f, **neuron_kwargs)
            )
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone features [B, C] without the classification head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(0)    # [1, B, C, H, W]

        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)               # [T, B, C, 1, 1]
        x = torch.flatten(x, 2)           # [T, B, C]
        return x.mean(dim=0)              # [B, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def sew_resnet18(**kwargs) -> SEWResNet:
    return SEWResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs) -> SEWResNet:
    return SEWResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs) -> SEWResNet:
    return SEWResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
