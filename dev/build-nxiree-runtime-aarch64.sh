#!/usr/bin/env bash
# Cross-build libnx_iree_runtime.so for the Jetson with the Nerves toolchain.
# Usage: build-nxiree-runtime-aarch64.sh <nx_iree dir> <iree source dir>
set -e
NX=$1
SRC=$2
TC=/var/home/marc/.local/share/nerves/artifacts/nerves_toolchain_aarch64_nerves_linux_gnu-linux_x86_64-15.3.1
SYSROOT=$TC/aarch64-nerves-linux-gnu/sysroot
cd "$NX"
cmake -G Ninja -B iree-runtime/aarch64/iree-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$TC/bin/aarch64-nerves-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$TC/bin/aarch64-nerves-linux-gnu-g++ \
  -DCMAKE_SYSROOT=$SYSROOT \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DIREE_HOST_BIN_DIR="$NX/iree-runtime/build/tools" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_RUNTIME_BUILD_DIR="$NX/iree-runtime/aarch64/build" \
  -DIREE_RUNTIME_INCLUDE_PATH="$SRC/runtime/src/iree" \
  -DNX_IREE_SOURCE_DIR="$SRC" \
  -S cmake
cmake --build iree-runtime/aarch64/iree-build --config Release
cmake --install iree-runtime/aarch64/iree-build --config Release --prefix "$NX/iree-runtime/aarch64/install"
