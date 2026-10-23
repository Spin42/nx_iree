#!/usr/bin/env bash
# Compile an imported YOLOv5 ONNX module for IREE's Vulkan backend.
# Usage: compile.sh model.mlir out.vmfb <vulkan-target> [tuning-spec.mlir]
set -euo pipefail
input=$1 out=$2 target=$3 spec=${4:-}
args=(
  "$input" --iree-input-type=onnx --iree-hal-target-backends=vulkan-spirv
  "--iree-vulkan-target=$target"
  --iree-global-opt-use-im2col-for-convs
  '--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-preprocessing-convert-conv-to-channels-last))'
  -o "$out"
)
[[ -n $spec ]] && args+=("--iree-codegen-transform-dialect-library=$(realpath "$spec")")
exec "${IREE_COMPILE:-iree-compile}" "${args[@]}"
