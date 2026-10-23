"""Convert an fp16 ONNX export to fp32 (initializers, graph I/O, value infos,
constant attributes and Cast targets).

Usage: to_fp32.py model-fp16.onnx model-fp32.onnx
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

model = onnx.load(sys.argv[1])
graph = model.graph
for tensor in graph.initializer:
    if tensor.data_type == TensorProto.FLOAT16:
        tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name))
for value in list(graph.input) + list(graph.output) + list(graph.value_info):
    if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
        value.type.tensor_type.elem_type = TensorProto.FLOAT
for node in graph.node:
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT16:
            attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t).astype(np.float32), attr.t.name))
        if node.op_type == "Cast" and attr.name == "to" and attr.i == TensorProto.FLOAT16:
            attr.i = TensorProto.FLOAT
onnx.checker.check_model(model)
onnx.save(model, sys.argv[2])
