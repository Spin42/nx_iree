#!/usr/bin/env bash
# Check a compiled YOLOv5 module against a reference, then time it.
# Usage: run.sh module.vmfb blob.bin reference.bin [device]
#   ELEM=f32|f16   element type of the module's input and outputs (f32)
#   SIZE=HxW       input size (640x640)
#   HEADS=1        the module comes from cut_head.py (three head outputs)
# The reference holds N x 85 f32 rows (reference.py, or OpenCV DNN's output
# for a full export). The default device is the first Vulkan device; pass
# vulkan://<path> from `iree-run-module --dump_devices=vulkan` for another.
set -euo pipefail
module=$1 blob=$2 reference=$3 device=${4:-vulkan}
tools=${IREE_TOOLS:-} elem=${ELEM:-f32} size=${SIZE:-640x640}
here=$(cd "$(dirname "$0")" && pwd)
common=("--device=$device" "--module=$module" "--function=${IREE_FUNCTION:-torch_jit}"
        "--input=1x3x${size}x$elem=@$blob")
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
if [[ ${HEADS:-0} == 1 ]]; then
  "${tools}iree-run-module" "${common[@]}" "--output=@$out/0.bin" "--output=@$out/1.bin" "--output=@$out/2.bin" > /dev/null
  python3 "$here/compare.py" "$reference" "$elem" "$out/0.bin" "$out/1.bin" "$out/2.bin" "${size%x*}" "${size#*x}"
else
  "${tools}iree-run-module" "${common[@]}" "--output=@$out/0.bin" > /dev/null
  python3 "$here/compare.py" "$reference" "$elem" "$out/0.bin"
fi
"${tools}iree-benchmark-module" "${common[@]}" \
  --batch_size=1 --benchmark_min_time=2s --benchmark_repetitions=3
