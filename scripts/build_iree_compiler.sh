#!/usr/bin/env bash
set -euo pipefail

if [[ -n ${CROSSCOMPILE:-} ]]; then
  printf 'Build the IREE compiler in a host environment without CROSSCOMPILE.\n' >&2
  exit 1
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
revision=${NX_IREE_GIT_REV:-v3.13.0}
source_dir=${NX_IREE_SOURCE_DIR:-${HOME}/.cache/nx_iree/iree-${revision}}
build_dir=${NX_IREE_COMPILER_BUILD_DIR:-${repo_root}/iree-compiler-build}

if [[ ! -d $source_dir ]]; then
  bash "$repo_root/scripts/clone_iree.sh" "$revision" "$source_dir"
fi

git -C "$source_dir" submodule update --init --depth 1 \
  third_party/llvm-project third_party/stablehlo
bash "$repo_root/scripts/apply_iree_patches.sh" "$source_dir" "$repo_root/patches/iree"

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC:-clang}" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DIREE_USE_LINKER="${IREE_LINKER:-lld}" \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_VULKAN_SPIRV=ON \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_TARGET_BACKEND_LLVM_CPU_WASM=OFF \
  "-DIREE_DEFAULT_CPU_LLVM_TARGETS=${IREE_CPU_TARGETS:-X86;AArch64}" \
  -DIREE_INPUT_TORCH=OFF \
  -DIREE_INPUT_TOSA=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON \
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  "$@"

cmake --build "$build_dir" --target iree-opt iree-compile FileCheck \
  --parallel "${IREE_BUILD_JOBS:-8}"
printf 'Select this compiler with NX_IREE_COMPILER_DIR=%s/tools\n' "$build_dir"
