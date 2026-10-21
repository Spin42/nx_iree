#!/usr/bin/env bash
# AOT-compile the Jetson test modules for each given IREE Vulkan target into
# the board bundle (one jetson-modules-<target>.bin each).
set -e
B=/var/home/marc/Projects/nx_iree/bench
OUT=/var/home/marc/.claude/jobs/7dc3895b/tmp/nxiree-board
rm -f $OUT/jetson-modules*.bin
cd $B
for t in "$@"; do
  NXIREE_VULKAN_TARGET=$t AOT_OUT=$OUT/jetson-modules-$t.bin mix run aot.exs 2>&1 | tail -1
done
