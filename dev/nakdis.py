#!/usr/bin/env python3
"""Disassemble the Nth encoded shader of a NAK_DEBUG=print log with nvdisasm.

Usage: nakdis.py <log> <shader-index from 1> [extra nvdisasm args...]
"""
import struct
import subprocess
import sys

NVDISASM = "/var/home/marc/.claude/jobs/7dc3895b/tmp/nvdisasm/x/cuda_nvdisasm-linux-x86_64-12.6.77-archive/bin/nvdisasm"
OUT = "/var/home/marc/.claude/jobs/7dc3895b/tmp/nakdis.bin"

log, idx = sys.argv[1], int(sys.argv[2])
k, words, grab = 0, [], False
for line in open(log, errors="replace"):
    if line.startswith("Encoded shader:"):
        k += 1
        grab = k == idx
        continue
    if grab:
        parts = line.split()
        if parts and all(len(p) == 8 for p in parts):
            words += [int(p, 16) for p in parts]
        else:
            grab = False
with open(OUT, "wb") as f:
    f.write(struct.pack(f"<{len(words)}I", *words))
subprocess.run([NVDISASM, "-b", "SM53", *sys.argv[3:], OUT], check=False)
