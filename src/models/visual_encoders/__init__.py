"""Visual encoder reference mapping (architecture -> open_clip model name).

All visual encoders are implemented by open_clip and selected via model name.
This package documents the mapping from paper architectures (Table 2) to
open_clip model config names used in Hydra configs.

Architecture           | open_clip name                    | Params
-----------------------|-----------------------------------|---------
ViT-L/14              | ViT-L-14                          | 304M
ViT-B/16              | ViT-B-16                          | 86.2M
ViT-T/16              | timm-vit_tiny_patch16_224         | 5.6M
ResNet-101            | RN101                             | 56.3M
ResNet-50             | RN50                              | 38.3M
ResNet-18             | timm_resnet18                     | 11.4M
Swin-Tiny             | timm-swin_tiny_patch4_window7_224 | 27.9M
MobileViT-S           | timm-mobilevit_s                  | 5.3M
"""
