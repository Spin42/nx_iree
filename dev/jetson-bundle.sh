#!/usr/bin/env bash
# Assemble the board bundle used by bench/jetson/board_run.exs:
#   lib/<app>/ebin   BEAM files from the bench host build (portable)
#   lib/nx_iree/priv aarch64 NIF and runtime from a Nerves build of an app
#                    that depends on nx_iree
#   run.exs          bench/jetson/run.exs
#   jetson-modules-*.bin  given module files (bench/aot.exs output)
#
# Usage: jetson-bundle.sh <out dir> <nerves app _build/<target>/lib dir> <modules.bin>...
set -e
OUT=$1
NERVES_LIB=$2
shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
HOST_LIB=$REPO/bench/_build/dev/lib

rm -rf "$OUT"
mkdir -p "$OUT/lib"
for app in complex telemetry nx nx_iree; do
  mkdir -p "$OUT/lib/$app"
  cp -r "$HOST_LIB/$app/ebin" "$OUT/lib/$app/"
done
cp -rL "$NERVES_LIB/nx_iree/priv" "$OUT/lib/nx_iree/priv"
file -L "$OUT/lib/nx_iree/priv/libnx_iree.so" | grep -q aarch64 || { echo "priv is not aarch64"; exit 1; }
cp "$REPO/bench/jetson/run.exs" "$OUT/run.exs"
for m in "$@"; do cp "$m" "$OUT/"; done
find "$OUT" \( -name "*.so" -o -name "*.bin" \) -printf "%s %p\n" | sort -k2
