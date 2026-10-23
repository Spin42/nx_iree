"""Write the reference detections of a YOLOv5 model for an input blob.

Usage: reference.py model.onnx blob.bin HEIGHT WIDTH out.bin

Runs the fp32 model with onnxruntime on the CPU. A model cut by
`cut_head.py` is decoded with `decode.py`; a full export is written as is.
The output holds N x 85 f32 rows.
"""
import sys

import numpy as np
import onnxruntime as ort

from decode import decode

model, blob, height, width, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
x = np.fromfile(blob, np.float32).reshape(1, 3, height, width)
results = session.run(None, {session.get_inputs()[0].name: x})
rows = decode(results) if len(results) == 3 else results[0].reshape(-1, 85)
rows.astype(np.float32).tofile(out)
