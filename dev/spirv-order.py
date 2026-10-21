"""Show the order of global loads, shared stores/loads, FMAs and barriers in
the loop body of an IREE .spirv.mlir dump (compressed into runs).

Usage: spirv-order.py <file.spirv.mlir>
"""
import re
import sys

text = open(sys.argv[1]).read().splitlines()
seq = []
for line in text:
    if "spirv.Load" in line and "StorageBuffer" in line:
        seq.append("gLD")
    elif "spirv.Load" in line and "Workgroup" in line:
        seq.append("sLD")
    elif "spirv.Store" in line and "Workgroup" in line:
        seq.append("sST")
    elif "spirv.Store" in line and "StorageBuffer" in line:
        seq.append("gST")
    elif "spirv.GL.Fma" in line:
        seq.append("FMA")
    elif "spirv.ControlBarrier" in line:
        seq.append("BAR")
    elif "spirv.mlir.loop" in line or "spirv.Branch" in line or "spirv.BranchConditional" in line or "^bb" in line:
        seq.append("|" + line.strip()[:30])
runs = []
for s in seq:
    if runs and runs[-1][0] == s:
        runs[-1][1] += 1
    else:
        runs.append([s, 1])
print(" ".join(f"{s}x{n}" if n > 1 else s for s, n in runs))
