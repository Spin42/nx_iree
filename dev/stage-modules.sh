#!/usr/bin/env bash
# Put the given sweep module files (by config name, or "default" for the
# rdna3 preset) into the board bundle.
set -e
T=/var/home/marc/.claude/jobs/7dc3895b/tmp
rm -f $T/nxiree-board/jetson-modules*.bin
for n in "$@"; do
  if [ "$n" = default ]; then
    (cd /var/home/marc/Projects/nx_iree/bench && NXIREE_VULKAN_TARGET=rdna3 AOT_OUT=$T/nxiree-board/jetson-modules-default.bin mix run aot.exs > /dev/null 2>&1)
  else
    cp $T/sweep/$n.bin $T/nxiree-board/jetson-modules-$n.bin
  fi
done
ls $T/nxiree-board/
