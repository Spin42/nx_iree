"""Rewrite YOLOv5's 6x6 stride-2 stem as space-to-depth plus a 3x3 convolution.

Usage: stem_space_to_depth.py model.onnx out.onnx

A KxK convolution with stride 2 and padding K/2 - 1 (K even) equals a 2x2
space-to-depth of its input followed by a (K/2)x(K/2) stride-1 convolution
with 4x the input channels and padding K/4 rounded down, with the weights
rearranged to match. For YOLOv5's stem (6x6, stride 2, padding 2) that is the
original Focus layer: a 3x3 stride-1 convolution, which Winograd can lower
instead of a 6x6 im2col. Values are unchanged. Run it after
pad_yolov5_onnx.py so the space-to-depth sees four input channels.
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

model = onnx.shape_inference.infer_shapes(onnx.load(sys.argv[1]))
graph = model.graph
shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
          for v in list(graph.value_info) + list(graph.input)}
inits = {t.name: t for t in graph.initializer}
nodes, done = [], False
for node in graph.node:
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    k = list(attrs.get("kernel_shape", []))
    if (not done and node.op_type == "Conv" and k == [6, 6] and list(attrs.get("strides", [])) == [2, 2]
            and list(attrs.get("pads", [])) == [2, 2, 2, 2] and attrs.get("group", 1) == 1):
        n, c, h, w = shapes[node.input[0]]
        weight = numpy_helper.to_array(inits[node.input[1]])  # F, C, 6, 6
        f = weight.shape[0]
        # w2[f, c*4 + rh*2 + rw, a, b] = w[f, c, 2a + rh, 2b + rw]
        w2 = weight.reshape(f, c, 3, 2, 3, 2).transpose(0, 1, 3, 5, 2, 4).reshape(f, c * 4, 3, 3)
        tag = node.name.replace("/", "_")
        graph.initializer.extend([
            numpy_helper.from_array(np.ascontiguousarray(w2), f"{tag}_w3x3"),
            numpy_helper.from_array(np.array([n, c, h // 2, 2, w // 2, 2], np.int64), f"{tag}_s6"),
            numpy_helper.from_array(np.array([n, c * 4, h // 2, w // 2], np.int64), f"{tag}_s4"),
        ])
        nodes += [
            helper.make_node("Reshape", [node.input[0], f"{tag}_s6"], [f"{tag}_6d"]),
            helper.make_node("Transpose", [f"{tag}_6d"], [f"{tag}_t"], perm=[0, 1, 3, 5, 2, 4]),
            helper.make_node("Reshape", [f"{tag}_t", f"{tag}_s4"], [f"{tag}_s2d"]),
            helper.make_node("Conv", [f"{tag}_s2d", f"{tag}_w3x3"] + list(node.input[2:]), list(node.output),
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1], name=node.name),
        ]
        done = True
    else:
        nodes.append(node)
assert done, "no 6x6 stride-2 stem convolution found"
del graph.node[:]
graph.node.extend(nodes)
used = {i for n in graph.node for i in n.input}
kept = [t for t in graph.initializer if t.name in used]
del graph.initializer[:]
graph.initializer.extend(kept)
del graph.value_info[:]
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, sys.argv[2])
