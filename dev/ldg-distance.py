"""For each global load (LDG) in an nvdisasm SASS file, count the instructions
issued before the first instruction that reads its destination register(s),
and whether a barrier lies in between.

Usage: ldg-distance.py <file.sass>
"""
import re
import sys

pat = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?P[0-9T]\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?\s*([^;]*);")
instrs = []
for line in open(sys.argv[1]):
    m = pat.search(line)
    if m:
        instrs.append((int(m.group(1), 16), m.group(3), m.group(4) or "", m.group(5)))


def regs(text):
    return [int(r) for r in re.findall(r"\bR(\d+)\b", text)]


for i, (addr, op, mods, args) in enumerate(instrs):
    if op != "LDG":
        continue
    parts = [p.strip() for p in args.split(",", 1)]
    base = regs(parts[0])[0]
    width = 4 if ".128" in mods else 2 if ".64" in mods else 1
    dst = set(range(base, base + width))
    dist, bar, user = None, False, None
    for j in range(i + 1, len(instrs)):
        a2, op2, m2, args2 = instrs[j]
        if op2 == "BAR":
            bar = True
        split = args2.split(",", 1)
        if op2 in ("STS", "STG", "ST"):
            srcs = args2
        else:
            srcs = split[1] if len(split) > 1 else ""
        if set(regs(srcs)) & dst:
            dist, user = j - i, f"{op2}{m2} @{a2:05x}"
            break
        if op2 in ("BRA", "EXIT"):
            break
    print(f"LDG{mods} R{base} @{addr:05x}: first use after {dist} instrs by {user}{' (barrier between)' if bar else ''}")
