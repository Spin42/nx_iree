#!/usr/bin/env bash
# Check a compiled YOLOv5 module against a reference output, then time it.
# Usage: run.sh module.vmfb blob.bin expected.bin [device]
# The default device is the first Vulkan device; pass vulkan://<path> from
# `iree-run-module --dump_devices=vulkan` to choose another.
set -euo pipefail
module=$1 blob=$2 expected=$3 device=${4:-vulkan}
tools=${IREE_TOOLS:-}
common=("--device=$device" "--module=$module" "--function=${IREE_FUNCTION:-torch_jit}"
        "--input=1x3x640x640xf32=@$blob")
"${tools}iree-run-module" "${common[@]}" \
  "--expected_output=1x25200x85xf32=@$expected" "--expected_f32_threshold=${THRESHOLD:-0.05}"
"${tools}iree-benchmark-module" "${common[@]}" \
  --batch_size=1 --benchmark_min_time=2s --benchmark_repetitions=3
