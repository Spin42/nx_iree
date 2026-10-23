"""Move the bias of 3x3 stride-1 convolutions into a separate Add.

Usage: split_conv_bias.py model.onnx out.onnx

IREE 3.13's Winograd rewrite (`iree-linalg-ext-convert-conv2d-to-winograd`)
produces wrong results for a convolution that carries a bias, while the same
convolution without one is exact. The Add keeps the model's values unchanged
and fuses into the following elementwise ops. Only the convolutions that
Winograd converts are touched.
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

model = onnx.load(sys.argv[1])
graph = model.graph
inits = {t.name: t for t in graph.initializer}
nodes = []
for node in graph.node:
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if (node.op_type == "Conv" and len(node.input) == 3 and list(attrs.get("kernel_shape", [])) == [3, 3]
            and list(attrs.get("strides", [1, 1])) == [1, 1] and attrs.get("group", 1) == 1
            and list(attrs.get("dilations", [1, 1])) == [1, 1]):
        bias = numpy_helper.to_array(inits[node.input[2]])
        name = node.input[2] + "_nchw"
        graph.initializer.append(numpy_helper.from_array(bias.reshape(1, -1, 1, 1), name))
        out = node.output[0]
        node.output[0] = out + "_nobias"
        del node.input[2]
        nodes += [node, helper.make_node("Add", [out + "_nobias", name], [out], name=node.name + "_bias")]
    else:
        nodes.append(node)
del graph.node[:]
graph.node.extend(nodes)
del graph.value_info[:]
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, sys.argv[2])
