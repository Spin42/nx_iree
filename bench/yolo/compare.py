"""Compare a YOLOv5 output with the reference at the detection level.

Usage: compare.py output.bin reference.bin [f32|f16]

Both files hold 25200 x 85 values (x, y, w, h, objectness, 80 class scores).
The reference is f32. Passes when the candidates scoring above 0.25
(objectness x class) agree with the reference's, apart from at most 2 %,
and every common candidate keeps its class. fp16 accumulation moves box
coordinates by a few pixels, so values are reported, not thresholded.
"""
import sys

import numpy as np

out_path, ref_path = sys.argv[1:3]
dtype = np.float16 if (sys.argv[3] if len(sys.argv) > 3 else "f32") == "f16" else np.float32
count = 25200 * 85
raw = open(out_path, "rb").read()
out = np.frombuffer(raw[-count * np.dtype(dtype).itemsize:], dtype).astype(np.float32).reshape(25200, 85)
ref = np.fromfile(ref_path, np.float32).reshape(25200, 85)


def candidates(o):
    scores = o[:, 4:5] * o[:, 5:]
    return set(np.nonzero(scores.max(1) > 0.25)[0].tolist()), scores.argmax(1)


got, got_class = candidates(out)
want, want_class = candidates(ref)
common = got & want
same_class = all(got_class[i] == want_class[i] for i in common)
print(f"finite {np.isfinite(out).all()}, box max diff {np.abs(out[:, :4] - ref[:, :4]).max():.2f} px, "
      f"score max diff {np.abs(out[:, 4:] - ref[:, 4:]).max():.4f}")
print(f"candidates: output {len(got)}, reference {len(want)}, common {len(common)}, same class {same_class}")
ok = np.isfinite(out).all() and same_class and len(got ^ want) <= max(1, len(want) // 50)
print("DETECTIONS_MATCH" if ok else "DETECTIONS_DIFFER")
sys.exit(0 if ok else 1)
