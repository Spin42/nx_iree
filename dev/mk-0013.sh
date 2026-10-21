#!/usr/bin/env bash
# Generate patches/mesa3d-style 0013 from the NAK scheduler change.
T=/var/home/marc/.claude/jobs/7dc3895b/tmp
NAK=src/nouveau/compiler/nak
OUT=$T/0013-nak-Only-order-memory-ops-that-may-conflict-when-scheduling.patch
cat > $OUT <<'EOF'
From: Marc Lainez <marc.lainez@gmail.com>
Date: Wed, 21 Oct 2026 12:00:00 +0200
Subject: [PATCH] nak: Only order memory ops that may conflict when scheduling

Both schedulers chained every memory instruction to the previous one, so
a global load could not move above an unrelated shared load.  In tiled
matmul shaders that left the next tile's global loads after all FMAs
with their latency exposed, or forced every shared-memory fragment to be
loaded up front.  Order memory instructions only when they access the
same space and at least one writes, or both are strong at GPU or system
scope; anything else stays a full fence.

An IREE 64x64x16 f32 matmul on GM20B drops from 240 to 128 GPRs and
issues its global loads ahead of the FMAs.

Assisted-by: Claude:claude-opus-5-5
---
EOF
for f in opt_instr_sched_common.rs opt_instr_sched_prepass.rs opt_instr_sched_postpass.rs; do
  diff -u --label a/$NAK/$f --label b/$NAK/$f $T/nak-orig/$f $T/mesa-host-src/$NAK/$f | sed '1s/^/diff --git a\/'"${NAK//\//\\/}"'\/'"$f"' b\/'"${NAK//\//\\/}"'\/'"$f"'\n/' >> $OUT
done
grep -c "^@@" $OUT
echo $OUT
