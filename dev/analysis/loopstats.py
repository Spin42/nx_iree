#!/usr/bin/env python3
"""Steady-state loop statistics for nvdisasm SM53 output.

Usage: loopstats.py <file.sass> <lo-hi>[,<lo-hi>...]   (hex address ranges, hi exclusive)

Prints opcode mix, FFMA register-bank conflicts (Maxwell: bank = reg % 4,
.reuse operands are served from the operand reuse cache), the list of LDS
instructions with base register and immediate, and a per-category summary.
"""
import collections
import re
import sys

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?P[0-9T]\s+)?([A-Z0-9_]+)((?:\.[A-Z0-9_]+)*)\s*(.*?)\s*;")
REG = re.compile(r"(-?\|?)R(\d+)(\.reuse)?(\.H[01])?(\|?)")


def parse(path):
    out = []
    for line in open(path):
        m = LINE.search(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        pred = (m.group(2) or "").strip()
        op = m.group(3)
        mods = m.group(4)
        operands = m.group(5)
        out.append((addr, pred, op, mods, operands))
    return out


def in_ranges(addr, ranges):
    return any(lo <= addr < hi for lo, hi in ranges)


def ffma_bank_conflict(operands):
    """Return (conflict, detail) for FFMA operand string 'Rd, Ra, Rb, Rc'."""
    parts = [p.strip() for p in operands.split(",")]
    if len(parts) != 4:
        return False, "nonreg"
    srcs = []
    for p in parts[1:]:
        m = REG.fullmatch(p.replace(" ", ""))
        if not m:
            srcs.append(None)  # constant / immediate: no bank
            continue
        srcs.append((int(m.group(2)), bool(m.group(3))))
    regs = [s for s in srcs if s is not None]
    # operands served by the reuse cache do not touch the banks
    live = [r for r, reuse in regs if not reuse]
    banks = collections.Counter(r % 4 for r in set(live))
    worst = max(banks.values()) if banks else 0
    return worst >= 2, f"banks={dict(banks)} srcs={srcs}"


def main():
    path = sys.argv[1]
    ranges = []
    for r in sys.argv[2].split(","):
        lo, hi = r.split("-")
        ranges.append((int(lo, 16), int(hi, 16)))
    ins = [i for i in parse(path) if in_ranges(i[0], ranges)]
    counts = collections.Counter(i[2] + ("" if i[2] not in ("LDS", "STS", "LDG", "STG") else i[3]) for i in ins)
    total = len(ins)
    nops = counts.get("NOP", 0)
    print(f"{total} instructions ({total - nops} excl. NOP) in loop ranges {sys.argv[2]}")
    for op, n in counts.most_common():
        print(f"  {n:5d} {op}")
    # categories
    cat = collections.Counter()
    conflicts = 0
    conflict_lines = []
    ffma_reuse = 0
    lds = []
    for addr, pred, op, mods, operands in ins:
        if op == "FFMA":
            cat["FFMA"] += 1
            c, d = ffma_bank_conflict(operands)
            if ".reuse" in operands:
                ffma_reuse += 1
            if c:
                conflicts += 1
                conflict_lines.append(f"{addr:05x} FFMA {operands} {d}")
        elif op == "LDS":
            cat["LDS"] += 1
            lds.append((addr, mods, operands))
        elif op == "STS":
            cat["STS"] += 1
        elif op == "LDG":
            cat["LDG"] += 1
        elif op == "STG":
            cat["STG"] += 1
        elif op in ("BAR", "MEMBAR"):
            cat["BAR/MEMBAR"] += 1
        elif op == "NOP":
            cat["NOP"] += 1
        elif op in ("BRA", "EXIT", "SSY", "SYNC", "PBK", "PCNT", "BRK", "CONT", "ISETP", "PSETP"):
            cat["control"] += 1
        elif op == "MOV":
            cat["MOV"] += 1
        else:
            cat["int/other"] += 1
    print("categories:", dict(cat))
    issued = total  # NOPs occupy issue slots too
    print(f"FFMA fraction of issued (incl NOP): {cat['FFMA']/issued:.3f}; excl NOP: {cat['FFMA']/max(1,total-nops):.3f}")
    print(f"FFMA with unresolved register-bank conflict: {conflicts} / {cat['FFMA']}  (FFMAs using .reuse: {ffma_reuse})")
    for l in conflict_lines[:12]:
        print("   ", l)
    print("LDS instructions (addr, width, operands):")
    for addr, mods, operands in lds:
        print(f"   {addr:05x} LDS{mods} {operands}")


if __name__ == "__main__":
    main()
