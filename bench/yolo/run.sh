#!/usr/bin/env bash
# Check a compiled YOLOv5 module against a reference output, then time it.
# Usage: run.sh module.vmfb blob.bin expected.bin [device]
# The default device is the first Vulkan device; pass vulkan://<path> from
# `iree-run-module --dump_devices=vulkan` to choose another. ELEM=f16 runs an
# F16=1 module with an f16 blob and checks its detections against the f32
# reference with compare.py.
set -euo pipefail
module=$1 blob=$2 expected=$3 device=${4:-vulkan}
tools=${IREE_TOOLS:-} elem=${ELEM:-f32}
common=("--device=$device" "--module=$module" "--function=${IREE_FUNCTION:-torch_jit}"
        "--input=1x3x640x640x$elem=@$blob")
if [[ $elem == f32 ]]; then
  "${tools}iree-run-module" "${common[@]}" \
    "--expected_output=1x25200x85xf32=@$expected" "--expected_f32_threshold=${THRESHOLD:-0.05}"
else
  out=$(mktemp)
  trap 'rm -f "$out"' EXIT
  "${tools}iree-run-module" "${common[@]}" "--output=@$out" > /dev/null
  python3 "$(dirname "$0")/compare.py" "$out" "$expected" "$elem"
fi
"${tools}iree-benchmark-module" "${common[@]}" \
  --batch_size=1 --benchmark_min_time=2s --benchmark_repetitions=3
