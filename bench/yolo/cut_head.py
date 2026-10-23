"""Cut a YOLOv5 ONNX export after its three detection-head convolutions and
set its input size.

Usage: cut_head.py model.onnx out.onnx HEIGHT WIDTH

The exported Detect layer (sigmoid, grid, anchors, concatenation into one
N x 85 tensor) is replaced by three channels-last outputs of shape
1 x H/s x W/s x 255 for strides s = 8, 16, 32, holding the raw logits;
`decode.py` turns them into boxes on the host. Without the Detect reshapes the
graph is fully convolutional, so HEIGHT and WIDTH may be any multiples of 32.
"""
import sys

import onnx
from onnx import TensorProto, helper

src, dst, height, width = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
model = onnx.load(src)
graph = model.graph
inits = {t.name for t in graph.initializer}
weights = {t.name: t for t in graph.initializer}
heads = [n for n in graph.node if n.op_type == "Conv" and weights[n.input[1]].dims[0] == 255]
assert len(heads) == 3, "expected three 255-channel detection convolutions"

outputs = []
for i, head in enumerate(heads):
    name = f"head{i}"
    graph.node.append(helper.make_node("Transpose", [head.output[0]], [name], perm=[0, 2, 3, 1], name=f"{name}_nhwc"))
    elem = graph.input[0].type.tensor_type.elem_type
    outputs.append(helper.make_tensor_value_info(name, elem, None))

# Keep only the nodes the new outputs depend on.
producers = {o: n for n in graph.node for o in n.output}
needed, stack = set(), [o.name for o in outputs]
while stack:
    tensor = stack.pop()
    node = producers.get(tensor)
    if node is None or id(node) in needed:
        continue
    needed.add(id(node))
    stack.extend(i for i in node.input if i and i not in inits)
kept = [n for n in graph.node if id(n) in needed]
used = {i for n in kept for i in n.input}
del graph.node[:]
graph.node.extend(kept)
kept_inits = [t for t in graph.initializer if t.name in used]
del graph.initializer[:]
graph.initializer.extend(kept_inits)
del graph.output[:]
graph.output.extend(outputs)
del graph.value_info[:]

dims = graph.input[0].type.tensor_type.shape.dim
dims[2].dim_value, dims[3].dim_value = height, width
model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.checker.check_model(model)
onnx.save(model, dst)
