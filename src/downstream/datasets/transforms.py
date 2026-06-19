"""Image + box transforms for visual grounding.

Port of HiVG/datasets/transforms.py with:
  - CLIP normalization stats instead of ImageNet
  - No horizontal flip (hurts grounding; paper explicitly notes this)
  - Dict-based transform API: each callable receives and returns input_dict
    with keys: 'img' (PIL), 'box' (Tensor x1y1x2y2), 'text' (str),
    'obj_mask' (Tensor [1,H,W] or None).
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageEnhance, ImageFilter

from ..utils.box_utils import xyxy2xywh

# CLIP normalization — used during pretraining, must match here
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _resize_long_side(img, box, size: int):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    img = F.resize(img, (round(h * ratio), round(w * ratio)))
    return img, box * ratio


def _resize_short_side(img, box, size: int):
    h, w = img.height, img.width
    ratio = size / min(h, w)
    img = F.resize(img, (round(h * ratio), round(w * ratio)))
    return img, box * ratio


def _crop(img, box, region, obj_mask=None):
    i, j, h, w = region
    img = F.crop(img, i, j, h, w)
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    box = box - torch.as_tensor([j, i, j, i])
    box = torch.min(box.reshape(2, 2), max_size).clamp(min=0).reshape(-1)
    if obj_mask is not None:
        obj_mask = obj_mask[:, i:i + h, j:j + w]
    return img, box, obj_mask


def _interp_mask(mask: torch.Tensor, new_size: tuple[int, int]) -> torch.Tensor:
    return (
        torch.nn.functional.interpolate(
            mask[:, None].float(), size=new_size, mode="nearest"
        )[:, 0]
        > 0.5
    )


# ---------------------------------------------------------------------------
# Transform classes (dict-based)
# ---------------------------------------------------------------------------

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, d: dict) -> dict:
        for t in self.transforms:
            d = t(d)
        return d


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, d: dict) -> dict:
        if random.random() >= 0.8:
            return d
        img = d["img"]
        ops = list(range(3))
        random.shuffle(ops)
        for op in ops:
            if op == 0:
                f = random.uniform(1 - self.brightness, 1 + self.brightness)
                img = ImageEnhance.Brightness(img).enhance(f)
            elif op == 1:
                f = random.uniform(1 - self.contrast, 1 + self.contrast)
                img = ImageEnhance.Contrast(img).enhance(f)
            else:
                f = random.uniform(1 - self.saturation, 1 + self.saturation)
                img = ImageEnhance.Color(img).enhance(f)
        d["img"] = img
        return d


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0), p=0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, d: dict) -> dict:
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            d["img"] = d["img"].filter(ImageFilter.GaussianBlur(radius=sigma))
        return d


class RandomResize:
    def __init__(self, sizes: list[int], long_side: bool = True):
        self.sizes = sizes
        self.long_side = long_side

    def __call__(self, d: dict) -> dict:
        img, box = d["img"], d["box"]
        size = random.choice(self.sizes)
        if self.long_side:
            img, box = _resize_long_side(img, box, size)
        else:
            img, box = _resize_short_side(img, box, size)
        d["img"], d["box"] = img, box
        if "obj_mask" in d and d["obj_mask"] is not None:
            d["obj_mask"] = _interp_mask(d["obj_mask"], (img.height, img.width))
        return d


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int, max_try: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try = max_try

    def __call__(self, d: dict) -> dict:
        img, box = d["img"], d["box"]
        obj_mask = d.get("obj_mask")
        for _ in range(self.max_try):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])
            # box is xyxy; keep crop only if box center is inside
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            i, j, rh, rw = region
            if j < cx < j + rw and i < cy < i + rh:
                img, box, obj_mask = _crop(img, box, region, obj_mask)
                d["img"], d["box"] = img, box
                if obj_mask is not None:
                    d["obj_mask"] = obj_mask
                return d
        return d


class RandomSelect:
    """Apply transforms1 when query contains directional words, else randomly pick."""
    _DIR_WORDS = {"left", "right", "top", "bottom", "middle"}

    def __init__(self, transforms1, transforms2, p=0.5):
        self.t1 = transforms1
        self.t2 = transforms2
        self.p = p

    def __call__(self, d: dict) -> dict:
        text = d.get("text", "")
        if any(w in text.split() for w in self._DIR_WORDS):
            return self.t1(d)
        return self.t1(d) if random.random() < self.p else self.t2(d)


class ToTensor:
    def __call__(self, d: dict) -> dict:
        d["img"] = F.to_tensor(d["img"])
        return d


class NormalizeAndPad:
    """Normalize with CLIP stats and pad to square target size.

    Converts box from (x1,y1,x2,y2) → normalized (cx,cy,w,h) in [0,1].
    Sets d['mask'] = int tensor, 0 where image exists, 1 where padded.
    """

    def __init__(self, size: int = 224, aug_translate: bool = False):
        self.size = size
        self.aug_translate = aug_translate
        self.mean = _CLIP_MEAN
        self.std = _CLIP_STD

    def __call__(self, d: dict) -> dict:
        img = F.normalize(d["img"], mean=self.mean, std=self.std)
        h, w = img.shape[1:]
        dh, dw = self.size - h, self.size - w

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        out_img = torch.zeros(3, self.size, self.size)
        out_mask = torch.ones(self.size, self.size, dtype=torch.int32)
        out_img[:, top:top + h, left:left + w] = img
        out_mask[top:top + h, left:left + w] = 0

        # shift and normalize box
        box = d["box"].clone()
        box[[0, 2]] += left
        box[[1, 3]] += top
        box = xyxy2xywh(box) / torch.tensor(
            [self.size, self.size, self.size, self.size], dtype=torch.float32
        )

        d["img"] = out_img
        d["mask"] = out_mask
        d["box"] = box

        if "obj_mask" in d and d["obj_mask"] is not None:
            om = torch.zeros(1, self.size, self.size)
            om[:, top:top + h, left:left + w] = d["obj_mask"].float()
            d["obj_mask"] = om

        return d


# ---------------------------------------------------------------------------
# Standard pipelines
# ---------------------------------------------------------------------------

def make_train_transforms(img_size: int = 224) -> Compose:
    # Matches HiVG's published recipe (aug_crop + aug_scale + aug_translate, used by
    # every train_and_eval_script/train_*.sh): the pre-crop resize/crop operates in a
    # fixed absolute pixel range (400-600 / 384-600) independent of img_size, resized
    # by the *short* side so the crop floor is always satisfiable; only the final
    # resize is scaled to img_size's multi-scale ladder.
    scales = [s for s in (img_size - 32 * i for i in range(7)) if s > 0]
    return Compose([
        RandomSelect(
            Compose([RandomResize(scales, long_side=True)]),
            Compose([
                RandomResize([400, 500, 600], long_side=False),
                RandomSizeCrop(384, 600),
                RandomResize(scales, long_side=True),
            ]),
            p=0.5,
        ),
        ColorJitter(),
        GaussianBlur(p=0.0),   # off by default; enable with p>0 if desired
        ToTensor(),
        NormalizeAndPad(size=img_size, aug_translate=True),
    ])


def make_val_transforms(img_size: int = 224) -> Compose:
    return Compose([
        RandomResize([img_size], long_side=True),
        ToTensor(),
        NormalizeAndPad(size=img_size, aug_translate=False),
    ])
