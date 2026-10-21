"""Summarise a region of nvdisasm output: opcode counts and where memory
operations and barriers sit.

Usage: sass-region.py <file.sass> [start-addr end-addr]   (hex, e.g. 04a8 51a8)
"""
import collections
import re
import sys

path = sys.argv[1]
lo = int(sys.argv[2], 16) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3], 16) if len(sys.argv) > 3 else 1 << 30
pat = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?P[0-9T]\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?(.*?);")
counts = collections.Counter()
events = []
for line in open(path):
    m = pat.search(line)
    if not m:
        continue
    addr = int(m.group(1), 16)
    if not lo <= addr < hi:
        continue
    op = m.group(3)
    counts[op] += 1
    if op in ("LDS", "LDG", "STS", "STG", "BAR", "MEMBAR", "BRA", "DEPBAR"):
        events.append(f"{addr:05x} {op}{m.group(4) or ''}{m.group(5)[:40]}")
total = sum(counts.values())
print(f"{total} instructions in [{lo:x}, {hi:x})")
for op, n in counts.most_common(14):
    print(f"  {n:5d} {op}")
print("memory/barrier events:")
for e in events[:60]:
    print("  " + e)
