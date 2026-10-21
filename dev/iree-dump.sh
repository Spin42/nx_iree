#!/usr/bin/env bash
# Compile an NxN f32 matmul for an IREE Vulkan target and dump the executables.
# Usage: iree-dump.sh <n> <target> [extra iree-compile flags...]
set -e
N=$1
TGT=$2
shift 2
T=/var/home/marc/.claude/jobs/7dc3895b/tmp
C=$T/iree-compiler-3.13/iree/compiler/_mlir_libs/iree-compile
OUT=$T/irdump/$TGT-$N${DUMP_TAG:+-$DUMP_TAG}
mkdir -p "$OUT"
cat > "$OUT/mm.mlir" <<EOF
func.func @main(%a: tensor<${N}x${N}xf32>, %b: tensor<${N}x${N}xf32>) -> tensor<${N}x${N}xf32> {
  %0 = "stablehlo.dot"(%a, %b) : (tensor<${N}x${N}xf32>, tensor<${N}x${N}xf32>) -> tensor<${N}x${N}xf32>
  return %0 : tensor<${N}x${N}xf32>
}
EOF
$C --iree-hal-target-backends=vulkan-spirv --iree-input-type=stablehlo \
  --iree-vulkan-target=$TGT --iree-hal-dump-executable-files-to="$OUT/files" "$@" \
  "$OUT/mm.mlir" -o "$OUT/mm.vmfb"
ls "$OUT/files"
