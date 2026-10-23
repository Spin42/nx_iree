"""Letterbox an image into a YOLOv5 input blob.

Usage: blob.py image.jpg blob.bin HEIGHT WIDTH [f32|f16]

The image is scaled to fit HEIGHT x WIDTH keeping its aspect ratio, centred
on a grey (114) canvas, converted to RGB in [0, 1] and written as raw
1 x 3 x HEIGHT x WIDTH values, as ultralytics' letterbox does.
"""
import sys

import numpy as np
from PIL import Image

path, out, height, width = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dtype = np.float16 if (sys.argv[5] if len(sys.argv) > 5 else "f32") == "f16" else np.float32
image = Image.open(path).convert("RGB")
scale = min(width / image.width, height / image.height)
size = (round(image.width * scale), round(image.height * scale))
canvas = Image.new("RGB", (width, height), (114, 114, 114))
canvas.paste(image.resize(size, Image.BILINEAR), ((width - size[0]) // 2, (height - size[1]) // 2))
(np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255).astype(dtype).tofile(out)
