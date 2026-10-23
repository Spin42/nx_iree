"""Replace 1x1 convolutions of channel concatenations by sums of convolutions.

Usage: split_concat_conv.py model.onnx out.onnx

conv1x1(concat(a, b, ...), W) equals conv1x1(a, W_a) + conv1x1(b, W_b) + ...
with W split along its input channels. YOLOv5's C3 blocks, SPPF and neck all
concatenate branches along channels and feed the result to 1x1
convolutions; IREE keeps those concatenations channel-major and transposes
the result back for every consumer. After the rewrite no concatenation is
left for them and the arithmetic is the same. A concatenation is rewritten
only when every consumer is an ungrouped 1x1 convolution.
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
outputs = {o.name for o in graph.output}
consumers = {}
for node in graph.node:
    for i in node.input:
        consumers.setdefault(i, []).append(node)


def is_1x1(node):
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    return (node.op_type == "Conv" and node.input[1] in inits and list(attrs.get("kernel_shape", [])) == [1, 1]
            and attrs.get("group", 1) == 1 and list(attrs.get("strides", [1, 1])) == [1, 1])


replaced, removed = {}, set()
new_inits = []
for node in graph.node:
    if node.op_type != "Concat" or helper.get_attribute_value(node.attribute[0]) not in (1, -3):
        continue
    users = consumers.get(node.output[0], [])
    if not users or node.output[0] in outputs or not all(is_1x1(u) for u in users):
        continue
    widths = [shapes[i][1] for i in node.input]
    for conv in users:
        w = numpy_helper.to_array(inits[conv.input[1]])
        tag = conv.name.replace("/", "_")
        parts, start = [], 0
        for j, (src, width) in enumerate(zip(node.input, widths)):
            wname = f"{tag}_w{j}"
            new_inits.append(numpy_helper.from_array(np.ascontiguousarray(w[:, start:start + width]), wname))
            ins = [src, wname] + ([conv.input[2]] if j == 0 and len(conv.input) > 2 else [])
            parts.append(helper.make_node("Conv", ins, [f"{tag}_part{j}"], kernel_shape=[1, 1], name=f"{tag}_part{j}"))
            start += width
        count = len(parts)
        total = parts[0].output[0]
        for j in range(1, count):
            out = conv.output[0] if j == count - 1 else f"{tag}_sum{j}"
            parts.append(helper.make_node("Add", [total, parts[j].output[0]], [out], name=f"{tag}_sum{j}"))
            total = out
        replaced[id(conv)] = parts
    removed.add(id(node))

nodes = []
for node in graph.node:
    if id(node) in removed:
        continue
    nodes.extend(replaced.get(id(node), [node]))
del graph.node[:]
graph.node.extend(nodes)
graph.initializer.extend(new_inits)
used = {i for n in graph.node for i in n.input}
kept = [t for t in graph.initializer if t.name in used]
del graph.initializer[:]
graph.initializer.extend(kept)
del graph.value_info[:]
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, sys.argv[2])
print(f"rewrote {len(removed)} concatenations feeding {len(replaced)} convolutions")
