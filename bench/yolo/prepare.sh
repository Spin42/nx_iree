#!/usr/bin/env bash
# Turn an ultralytics YOLOv5 ONNX export into the IREE input used on GPUs.
# Usage: prepare.sh yolov5s.onnx out-prefix HEIGHT WIDTH
# Writes <out-prefix>.onnx (the rewritten model) and <out-prefix>.mlir.
# HEIGHT and WIDTH must be multiples of 32; choose sizes whose feature maps
# (H/8 x W/8, H/16 x W/16, H/32 x W/32) divide evenly into the tuning
# spec's tiles, e.g. 512x640 for 4:3 frames rather than 480x640.
set -euo pipefail
src=$1 out=$2 height=$3 width=$4
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"$py" "$here/cut_head.py" "$src" "$tmp/1.onnx" "$height" "$width"
"$py" "$here/split_concat_conv.py" "$tmp/1.onnx" "$tmp/2.onnx" > /dev/null
"$py" "$here/pad_yolov5_onnx.py" "$tmp/2.onnx" "$tmp/3.onnx"
"$py" "$here/stem_space_to_depth.py" "$tmp/3.onnx" "$tmp/4.onnx"
"$py" "$here/split_conv_bias.py" "$tmp/4.onnx" "$out.onnx"
"$py" -m iree.compiler.tools.import_onnx "$out.onnx" -o "$out.mlir"
