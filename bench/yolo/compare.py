"""Compare YOLOv5 outputs with the reference at the detection level.

Usage: compare.py reference.bin f32|f16 output.bin
       compare.py reference.bin f32|f16 head0.bin head1.bin head2.bin HEIGHT WIDTH

The reference holds N x 85 f32 rows (`reference.py`). The output is either
the full export's N x 85 tensor or the three channels-last head tensors of
a `cut_head.py` model (255 or 256 channels), decoded with `decode.py`.
Passes when every candidate the reference scores above 0.28 (objectness x
class) is found above 0.25 with the same class, and nothing the reference
scores below 0.22 is. Candidates within 0.03 of the 0.25 threshold may fall
either side: fp16 accumulation moves scores by up to about 0.02 and box
coordinates by a few pixels, so those values are reported, not thresholded.
"""
import os
import sys

import numpy as np

from decode import STRIDES, decode

ref_path, elem, *outputs = sys.argv[1:]
dtype = np.float16 if elem == "f16" else np.float32
ref = np.fromfile(ref_path, np.float32).reshape(-1, 85)


def load(path, count):
    raw = open(path, "rb").read()
    return np.frombuffer(raw[-count * np.dtype(dtype).itemsize:], dtype).astype(np.float32)


if len(outputs) == 1:
    out = load(outputs[0], ref.size).reshape(-1, 85)
else:
    *files, height, width = outputs
    heads = []
    for path, stride in zip(files, STRIDES):
        h, w = int(height) // stride, int(width) // stride
        channels = (os.path.getsize(path) // np.dtype(dtype).itemsize) // (h * w)
        heads.append(load(path, h * w * channels).reshape(1, h, w, channels))
    out = decode(heads)


def scores(o):
    s = o[:, 4:5] * o[:, 5:]
    return s.max(1), s.argmax(1)


got_score, got_class = scores(out)
want_score, want_class = scores(ref)
got = set(np.nonzero(got_score > 0.25)[0].tolist())
want = set(np.nonzero(want_score > 0.25)[0].tolist())
missing = [i for i in np.nonzero(want_score > 0.28)[0] if got_score[i] <= 0.25 or got_class[i] != want_class[i]]
extra = [i for i in got if want_score[i] < 0.22]
print(f"finite {np.isfinite(out).all()}, box max diff {np.abs(out[:, :4] - ref[:, :4]).max():.2f} px, "
      f"score max diff {np.abs(out[:, 4:] - ref[:, 4:]).max():.4f}")
print(f"candidates above 0.25: output {len(got)}, reference {len(want)}, common {len(got & want)}; "
      f"clear misses {len(missing)}, clear extras {len(extra)}")
ok = bool(np.isfinite(out).all()) and not missing and not extra
print("DETECTIONS_MATCH" if ok else "DETECTIONS_DIFFER")
sys.exit(0 if ok else 1)
