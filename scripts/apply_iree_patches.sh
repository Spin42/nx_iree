#!/usr/bin/env bash
set -euo pipefail

source_dir=$1
patch_dir=$2
shopt -s nullglob

for patch_file in "$patch_dir"/*.patch; do
  if git -C "$source_dir" apply --check "$patch_file" 2>/dev/null; then
    git -C "$source_dir" apply "$patch_file"
    printf 'Applied %s\n' "${patch_file##*/}"
  elif git -C "$source_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    printf 'Already applied %s\n' "${patch_file##*/}"
  else
    printf 'Cannot apply %s to %s; inspect the source version and local changes.\n' \
      "$patch_file" "$source_dir" >&2
    exit 1
  fi
done
