#!/usr/bin/env bash
# Rebuild the four AOT module sets that were timed on the board (preset
# targets, no tuning spec) and compile their 2048 matmul through the
# unpatched NAK on the GM20B shim. Output: $T/measured/<target>.{bin,nak.log,sass}
set -e
T=/var/home/marc/.claude/jobs/7dc3895b/tmp
OUT=$T/measured
mkdir -p $OUT
cd /var/home/marc/Projects/nx_iree/bench
for t in pascal rdna3 valhall adreno; do
  [ -f $OUT/$t.bin ] || NXIREE_VULKAN_TARGET=$t AOT_OUT=$OUT/$t.bin mix run aot.exs > $OUT/$t.aot.log 2>&1
  SHIM_BUILD=$T/mesa-host-build-orig bash $T/scripts/shim-nak.sh $OUT/$t.bin "matmul 2048" $OUT/$t.nak.log > /dev/null
  n=$(grep -c "Encoded shader" $OUT/$t.nak.log)
  python3 $T/scripts/nakdis.py $OUT/$t.nak.log $n > $OUT/$t.sass 2>&1
  echo "$t: shaders=$n gprs=$(grep 'Num GPRs' $OUT/$t.nak.log | tail -1 | awk '{print $3}') ffma=$(grep -c FFMA $OUT/$t.sass) lds=$(grep -c 'LDS' $OUT/$t.sass) ldg=$(grep -c 'LDG' $OUT/$t.sass) instrs=$(grep -cE '^\s+/\*[0-9a-f]+\*/' $OUT/$t.sass)"
done
