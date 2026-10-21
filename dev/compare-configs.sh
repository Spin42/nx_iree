#!/usr/bin/env bash
# Dump the 1024 matmul for each target and print IREE's chosen config.
S=/var/home/marc/.claude/jobs/7dc3895b/tmp/scripts
D=/var/home/marc/.claude/jobs/7dc3895b/tmp/irdump
for t in "$@"; do
  [ -d $D/$t-1024 ] || bash $S/iree-dump.sh 1024 $t > /dev/null 2>&1
  cfg=$(grep -ohE 'tile_sizes = [^>]*>|workgroup_size = \[[^]]*\]|spirv_pipeline<[A-Za-z]+>|subgroup_size = [0-9]+|pipeline_depth = [0-9]+' $D/$t-1024/files/configured_*.mlir | sort -u | tr '\n' ' ')
  echo "$t: $cfg"
done
