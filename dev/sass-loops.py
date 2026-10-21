"""Find loops (backward branches) in an nvdisasm SASS file and summarise each
loop body: instruction mix, memory ops, barriers.

Usage: sass-loops.py <file.sass>
"""
import collections
import re
import sys

pat = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?P[0-9T]\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?([^;]*);")
instrs = []
for line in open(sys.argv[1]):
    m = pat.search(line)
    if m:
        instrs.append((int(m.group(1), 16), m.group(3), (m.group(4) or ""), m.group(5)))

for addr, op, mods, args in instrs:
    if op != "BRA":
        continue
    t = re.search(r"0x([0-9a-f]+)", args)
    if not t:
        continue
    target = int(t.group(1), 16)
    if target >= addr:
        continue
    body = [i for i in instrs if target <= i[0] <= addr]
    c = collections.Counter(i[1] for i in body)
    mem = collections.Counter(i[1] + i[2] for i in body if i[1] in ("LDS", "LDG", "STS", "LD", "ST", "TLD", "TEX"))
    total = len(body)
    print(f"loop {target:05x}..{addr:05x}: {total} instrs, FFMA {c['FFMA']} ({100*c['FFMA']/total:.0f}%), "
          f"BAR {c['BAR']}, mem {dict(mem)}, int/addr {c['IADD']+c['XMAD']+c['SHL']+c['SHR']+c['LOP']+c['IADD3']+c['ISCADD']+c['LEA']}, "
          f"MOV {c['MOV']+c['MOV32I']}, other {total - sum(c[k] for k in ['FFMA','BAR','IADD','XMAD','SHL','SHR','LOP','IADD3','ISCADD','LEA','MOV','MOV32I']) - sum(mem.values())}")
