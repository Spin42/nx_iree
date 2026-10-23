"""Pad a YOLOv5 ONNX model so IREE's Vulkan matmuls get tileable shapes.

The stem convolution gets a fourth, all-zero input channel (im2col K becomes
6*6*4 = 144 instead of 108) and the three detect-head convolutions get a 256th
zero output channel, sliced back to 255 behind a reshape so the detect
reshapes stay as exported. The outputs are bit-identical to the original
model's.

Usage: pad_yolov5_onnx.py yolov5s.onnx yolov5s-padded.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1:3]
model = onnx.shape_inference.infer_shapes(onnx.load(src))
graph = model.graph
shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.value_info}
inits = {t.name: t for t in graph.initializer}
image = graph.input[0].name
n_in, _, h_in, w_in = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
nodes = []

for node in list(graph.node):
    if node.op_type == "Conv" and node.input[0] == image:
        w = numpy_helper.to_array(inits[node.input[1]])
        w4 = np.zeros((w.shape[0], 4) + w.shape[2:], w.dtype)
        w4[:, :3] = w
        inits[node.input[1]].CopyFrom(numpy_helper.from_array(w4, node.input[1]))
        graph.initializer.append(numpy_helper.from_array(np.zeros((n_in, 1, h_in, w_in), np.float32), "pad_zero_channel"))
        nodes.append(helper.make_node("Concat", [image, "pad_zero_channel"], ["images_padded"], axis=1, name="pad_concat"))
        node.input[0] = "images_padded"
        nodes.append(node)
    elif node.op_type == "Conv" and inits[node.input[1]].dims[0] == 255:
        w = numpy_helper.to_array(inits[node.input[1]])
        b = numpy_helper.to_array(inits[node.input[2]])
        inits[node.input[1]].CopyFrom(numpy_helper.from_array(np.concatenate([w, np.zeros((1,) + w.shape[1:], w.dtype)]), node.input[1]))
        inits[node.input[2]].CopyFrom(numpy_helper.from_array(np.concatenate([b, np.zeros((1,), b.dtype)]), node.input[2]))
        out = node.output[0]
        _, _, ny, nx = shapes[out]
        tag = node.name.replace("/", "_")
        node.output[0] = out + "_padded"
        for name, value in (("start", [0]), ("end", [255]), ("axes", [1]), ("flat", [1, 256, ny * nx]), ("back", [1, 255, ny, nx])):
            graph.initializer.append(numpy_helper.from_array(np.array(value, np.int64), f"{tag}_{name}"))
        nodes.append(node)
        nodes.append(helper.make_node("Reshape", [out + "_padded", f"{tag}_flat"], [out + "_flat"], name=f"{tag}_flat"))
        nodes.append(helper.make_node("Slice", [out + "_flat", f"{tag}_start", f"{tag}_end", f"{tag}_axes"], [out + "_sliced"], name=f"{tag}_slice"))
        nodes.append(helper.make_node("Reshape", [out + "_sliced", f"{tag}_back"], [out], name=f"{tag}_back"))
    else:
        nodes.append(node)

del graph.node[:]
graph.node.extend(nodes)
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, dst)
