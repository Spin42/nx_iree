"""Decode YOLOv5 detection-head outputs (from `cut_head.py`) on the host.

decode(heads) takes the three channels-last head tensors (1 x h x w x C,
C = 255 or a padded 256, strides 8, 16, 32) and returns the N x 85 rows of
the exported Detect layer: x, y, w, h in input pixels, objectness and 80
class scores. The anchors are YOLOv5's COCO defaults.
"""
import numpy as np

ANCHORS = (
    ((10, 13), (16, 30), (33, 23)),
    ((30, 61), (62, 45), (59, 119)),
    ((116, 90), (156, 198), (373, 326)),
)
STRIDES = (8, 16, 32)


def decode(heads):
    rows = []
    for head, anchors, stride in zip(heads, ANCHORS, STRIDES):
        _, h, w, _ = head.shape
        p = 1 / (1 + np.exp(-head[0, :, :, :255].astype(np.float32).reshape(h, w, 3, 85)))
        p = p.transpose(2, 0, 1, 3)  # anchor, y, x, 85 (the export's order)
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        xy = (p[..., :2] * 2 - 0.5 + np.stack([gx, gy], -1)) * stride
        wh = (p[..., 2:4] * 2) ** 2 * np.array(anchors, np.float32)[:, None, None, :]
        rows.append(np.concatenate([xy, wh, p[..., 4:]], -1).reshape(-1, 85))
    return np.concatenate(rows)
