#!/usr/bin/env bash
# For each matmul config "name tm tn tk wx wy depth", build a tuning spec, AOT
# the Jetson modules with it, compile the 1024 matmul through NAK on the GM20B
# shim and print GPR count plus loop shape.
# Usage: sweep-configs.sh <configs file>
set -e
T=/var/home/marc/.claude/jobs/7dc3895b/tmp
B=/var/home/marc/Projects/nx_iree/bench
OUT=$T/sweep
mkdir -p $OUT
while read -r name tm tn tk wx wy depth; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  spec=$OUT/$name.mlir
  python3 $B/tuning/matmul_spec.py $spec $tm $tn $tk $wx $wy $depth
  (cd $B && NXIREE_VULKAN_TARGET=rdna3 NXIREE_FLAGS="--iree-codegen-transform-dialect-library=$spec" \
    AOT_OUT=$OUT/$name.bin mix run aot.exs > $OUT/$name.aot.log 2>&1) || { echo "$name: AOT failed"; continue; }
  bash $T/scripts/shim-nak.sh $OUT/$name.bin "matmul 1024" $OUT/$name.nak.log > /dev/null
  gprs=$(grep "Num GPRs" $OUT/$name.nak.log | tail -1 | awk '{print $3}')
  python3 $T/scripts/nakdis.py $OUT/$name.nak.log 3 > $OUT/$name.sass 2>&1
  ffma=$(grep -c "FFMA" $OUT/$name.sass || true)
  lds=$(grep -c "LDS" $OUT/$name.sass || true)
  # position of the first global load relative to the first FFMA in the loop
  first_ldg=$(grep -n "LDG" $OUT/$name.sass | sed -n 2p | cut -d: -f1)
  first_ffma=$(grep -n "FFMA" $OUT/$name.sass | head -1 | cut -d: -f1)
  echo "$name tile=${tm}x${tn}x${tk} wg=${wx}x${wy} depth=$depth gprs=$gprs ffma=$ffma lds=$lds 2nd_ldg_line=$first_ldg 1st_ffma_line=$first_ffma"
done < "$1"
