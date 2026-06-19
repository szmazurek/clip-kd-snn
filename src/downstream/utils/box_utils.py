"""Box utility functions for visual grounding.

Port of HiVG/utils/box_utils.py with minor cleanup.
All boxes use (x1, y1, x2, y2) format unless noted.
"""
import torch
from torchvision.ops.boxes import box_area


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) → (x1, y1, x2, y2)."""
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h,
                        x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) → (cx, cy, w, h)."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2.0, (y0 + y1) / 2.0,
                        x1 - x0, y1 - y0], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU loss term. boxes in (x1,y1,x2,y2). Returns [N, M] pairwise matrix."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def bbox_to_mask(bbox_xywh: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Rasterize a single xywh box into a rectangular binary mask.

    Equivalent to HiVG's use_seg_mask=False path (box-as-mask), avoiding a
    pycocotools dependency for polygon decoding.

    Args:
        bbox_xywh: [4] tensor (x, y, w, h) in pixel coordinates.
        height, width: target mask size.

    Returns:
        [1, height, width] float tensor, 1 inside the box, 0 outside.
    """
    mask = torch.zeros(1, height, width)
    x, y, w, h = bbox_xywh.tolist()
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + w)), width), min(int(round(y + h)), height)
    mask[:, y0:y1, x0:x1] = 1.0
    return mask


def acc_at_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
               threshold: float = 0.5) -> torch.Tensor:
    """Fraction of predictions whose IoU with gt exceeds threshold.

    Args:
        pred_boxes: [B, 4] in (cx,cy,w,h) normalized.
        gt_boxes:   [B, 4] in (cx,cy,w,h) normalized.
    """
    pred_xyxy = xywh2xyxy(pred_boxes)
    gt_xyxy = xywh2xyxy(gt_boxes)
    iou, _ = box_iou(pred_xyxy, gt_xyxy)
    diag_iou = iou.diagonal()
    return (diag_iou >= threshold).float().mean()
