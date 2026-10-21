#!/usr/bin/env bash
# Build the patched NVK natively on the host with the nouveau drm-shim, so NAK
# can compile shaders for a GM20B without the board.
set -e
A=/var/home/marc/Projects/nerves_system_okdo_c100/.claude/worktrees/base-system/.nerves/artifacts/nerves_system_okdo_c100-portable-0.1.0
T=/var/home/marc/.claude/jobs/7dc3895b/tmp
SRC=$T/mesa-host-src
B=$T/mesa-host-build
# Only the code generators come from Buildroot; its pkg-config targets aarch64.
TOOLS=$T/mesa-host-tools
mkdir -p $TOOLS
for t in mesa_clc vtn_bindgen2 cbindgen bindgen python3 python; do ln -sf $A/host/bin/$t $TOOLS/$t; done
# Buildroot's meson must run under Buildroot's Python to import its mesonbuild
rm -f $TOOLS/meson
printf '#!/bin/sh\nexec %s %s "$@"\n' "$A/host/bin/python3" "$A/host/bin/meson" > $TOOLS/meson
chmod +x $TOOLS/meson
export PATH=$TOOLS:$PATH
export LIBCLANG_PATH=$A/host/lib
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $A/host/lib/clang/22/include"
export LD_LIBRARY_PATH=$A/host/lib
if [ ! -d $SRC ]; then
  mkdir -p $SRC
  rsync -a --exclude 'build*/' $A/build/mesa3d-26.1.8/ $SRC/
fi
ls $SRC/src/nouveau/drm-shim > /dev/null
if [ ! -d $B ]; then
  meson setup $B $SRC \
    -Dbuildtype=release \
    -Dvulkan-drivers=nouveau \
    -Dgallium-drivers= \
    -Dplatforms= \
    -Dglx=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled -Dopengl=false \
    -Dtools=drm-shim \
    -Dmesa-clc=system -Dprecomp-compiler=system \
    -Dllvm=disabled \
    -Dvulkan-layers= \
    -Dbuild-tests=false
fi
ninja -C $B
find $B -name "libvulkan_nouveau.so" -o -name "libnouveau_noop_drm_shim.so" | head
