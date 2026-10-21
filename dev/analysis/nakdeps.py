#!/usr/bin/env python3
"""Parse the last 'NAK IR after calc_instr_deps' section of a NAK log and
report, per basic block: instruction count, summed 'delay=' stall cycles,
number of scoreboard waits (wt=), yields, and load->first-use distances.

Usage: nakdeps.py <file.nak.log> [block ids to detail, e.g. 1,4]
"""
import re
import sys

SECTION = "NAK IR after calc_instr_deps:"
BLOCK = re.compile(r"^block\.(\S+) (\d+) (\S+) \[([^\]]*)\] -> \{")
END = re.compile(r"^\} -> \[([^\]]*)\]")
INS = re.compile(r"^\s+(?:(\S+(?:\.\.\S+)?(?:\s+c\d)?(?:\s+p\d)?)\s+=\s+)?(\S+)(.*?)\s*//\s*(.*)$")
DELAY = re.compile(r"delay=(\d+)")
WT = re.compile(r"wt=([01]+)")
RD = re.compile(r"rd:(\d)")
WR = re.compile(r"wr:(\d)")
REGS = re.compile(r"\br(\d+)(?:\.\.(\d+))?")


def regset(tok):
    out = set()
    for m in REGS.finditer(tok):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a + 1
        out.update(range(a, b))
    return out


def main():
    path = sys.argv[1]
    detail = set(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
    text = open(path).read()
    idx = text.rfind(SECTION)
    body = text[idx + len(SECTION):]
    lines = body.splitlines()
    blocks = []
    cur = None
    for line in lines:
        m = BLOCK.match(line)
        if m:
            cur = {"id": int(m.group(2)), "label": m.group(3), "preds": m.group(4), "ins": []}
            continue
        m = END.match(line)
        if m and cur is not None:
            cur["succ"] = m.group(1)
            blocks.append(cur)
            cur = None
            continue
        if cur is None:
            if line.startswith("Num GPRs") or line.startswith("Stage:"):
                break
            continue
        m = INS.match(line)
        if not m:
            continue
        dst, op, srcs, ann = m.group(1) or "", m.group(2), m.group(3), m.group(4)
        d = DELAY.search(ann)
        cur["ins"].append({
            "dst": dst, "op": op, "srcs": srcs, "ann": ann,
            "delay": int(d.group(1)) if d else 0,
            "wt": WT.search(ann).group(1) if WT.search(ann) else None,
            "rd": RD.search(ann).group(1) if RD.search(ann) else None,
            "wr": WR.search(ann).group(1) if WR.search(ann) else None,
            "yld": "yld" in ann,
        })
    print(f"{len(blocks)} blocks")
    for b in blocks:
        n = len(b["ins"])
        if n == 0:
            print(f"block {b['id']:3d} preds=[{b['preds']}] succ=[{b['succ']}] EMPTY")
            continue
        delay = sum(i["delay"] for i in b["ins"])
        waits = sum(1 for i in b["ins"] if i["wt"])
        ffma = sum(1 for i in b["ins"] if i["op"].startswith("ffma"))
        nop = sum(1 for i in b["ins"] if i["op"] == "nop")
        bars = sum(1 for i in b["ins"] if i["op"].startswith("bar") or i["op"].startswith("membar"))
        lds = sum(1 for i in b["ins"] if i["op"].startswith("ld.shared"))
        ldg = sum(1 for i in b["ins"] if i["op"].startswith("ld.global"))
        d1 = sum(1 for i in b["ins"] if i["delay"] <= 1)
        print(f"block {b['id']:3d} preds=[{b['preds']}] succ=[{b['succ']}] n={n} nop={nop} ffma={ffma} lds={lds} ldg={ldg} bar={bars} "
              f"sum_delay={delay} (avg {delay/n:.2f}/ins) waits={waits} delay<=1:{d1}")
        if b["id"] in detail:
            # load -> first-use distance: instructions and accumulated delay
            ins = b["ins"]
            for k, i in enumerate(ins):
                if i["op"].startswith("ld.shared") or i["op"].startswith("ld.global"):
                    dregs = regset(i["dst"])
                    acc = 0
                    found = None
                    for j in range(k + 1, len(ins)):
                        acc += ins[j - 1]["delay"]
                        if regset(ins[j]["srcs"]) & dregs or regset(ins[j]["dst"]) & dregs:
                            found = (j - k, acc, ins[j]["op"], ins[j]["wt"])
                            break
                    print(f"    {k:4d} {i['op']:28s} {i['dst']:10s} wr:{i['wr']} -> first use after {found}")
            # histogram of delays
            from collections import Counter
            print("    delay histogram:", sorted(Counter(i["delay"] for i in ins).items()))
            # wait instructions listing
            for k, i in enumerate(ins):
                if i["wt"] or i["yld"]:
                    print(f"    wait {k:4d} {i['op']:28s} {i['dst']:10s} {i['srcs'][:40]:40s} // {i['ann']}")


if __name__ == "__main__":
    main()
