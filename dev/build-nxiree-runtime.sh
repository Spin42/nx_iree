#!/usr/bin/env bash
# Build libnx_iree_runtime.so from the nx_iree checkout against an IREE source tree.
# Usage: build-nxiree-runtime.sh <nx_iree dir> <iree source dir>
set -e
NX=$1
SRC=$2
cd "$NX"
cmake -G Ninja -B iree-runtime/iree-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_RUNTIME_BUILD_DIR="$NX/iree-runtime/build" \
  -DIREE_RUNTIME_INCLUDE_PATH="$SRC/runtime/src/iree" \
  -DNX_IREE_SOURCE_DIR="$SRC" \
  -S cmake
cmake --build iree-runtime/iree-build --config Release
cmake --install iree-runtime/iree-build --config Release --prefix "$NX/iree-runtime/host/install"
