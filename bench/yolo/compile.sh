#!/usr/bin/env bash
# Compile an imported YOLOv5 ONNX module for IREE's Vulkan backend.
# Usage: compile.sh model.mlir out.vmfb <vulkan-target> [tuning-spec.mlir]
# F16=1 demotes every f32 value, accumulators included, to f16. Use it with
# an fp16 export on GPUs with fast packed fp16 arithmetic.
# WINOGRAD=1 lowers 3x3 stride-1 convolutions with Winograd F(6x6, 3x3); the
# model must have gone through split_conv_bias.py.
set -euo pipefail
input=$1 out=$2 target=$3 spec=${4:-}
args=(
  "$input" --iree-input-type=onnx --iree-hal-target-backends=vulkan-spirv
  "--iree-vulkan-target=$target"
  --iree-global-opt-use-im2col-for-convs
  -o "$out"
)
passes=iree-preprocessing-convert-conv-to-channels-last
if [[ ${WINOGRAD:-0} == 1 ]]; then
  passes+=',iree-linalg-ext-convert-conv2d-to-winograd{replace-all-convs=true}'
  # Keeps the transformed filters as precomputed constants.
  args+=(--iree-opt-const-expr-max-size-increase-threshold=1000000000)
fi
args+=("--iree-preprocessing-pass-pipeline=builtin.module(util.func($passes))")
[[ ${F16:-0} == 1 ]] && args+=(--iree-input-demote-f32-to-f16)
[[ -n $spec ]] && args+=("--iree-codegen-transform-dialect-library=$(realpath "$spec")")
exec "${IREE_COMPILE:-iree-compile}" "${args[@]}"
