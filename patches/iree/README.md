# IREE source fixes

These patches target IREE revision
`89ece87b89852100c0a4ac3d515e14842d090967` (`iree-3.13.0rc20261016`).

* `0001-reuse-all-free-tlsf-slabs.patch` searches uncached slabs before pool
  growth. An eight-slab regression reproduces the previous failure to reuse
  released memory. The fix is shared HAL allocator code.
* `0002-elide-proven-same-device-transfers.patch` uses inferred device equality
  only when all incoming producers agree, preserving unknown/mixed affinities
  and copies that protect borrowed storage. It includes loops, mixed import
  ownership, returned aliases, opaque forwarders, zero-trip loop results
  and interprocedural clone-order regressions. All four copy-elision suites
  and thirteen added cases pass.
* `0003-flush-generated-buffer-mappings.patch` flushes successful in-situ
  buffer generation before unmapping. Non-coherent device memory otherwise
  receives stale CLI inputs. Identical C100 edge tests fail with the original
  native tool and pass with the flush-only tool; NumPy and literal parsers are
  covered. NxIREE's upload path already uses a flushing transfer API.
* `0004-serialize-profile-metadata.patch` uses the local profile recorder's
  existing sink lock for metadata writes as well as event flushes. Concurrent
  Vulkan contexts otherwise interleave file-record headers and payloads.
  The native regression fails before the fix, passes after it and survives
  100 repetitions. C100's corrected two-context capture contains all 250
  dispatches with no invalid or dropped records.
* `0005-invalidate-mapped-transfer-sources.patch` invalidates non-coherent
  source mappings before the generic host transfer reads them. Mappable
  C100 outputs otherwise return stale data. Four native cases fail before
  the fix; all six pass afterwards, including error cleanup and unchanged
  coherent/host paths. The same C100 fixtures fail with the old runtime
  and pass with the corrected runtime.

Runtime source builds apply the patches automatically through the Makefile.
The apply helper accepts an already-applied patch and rejects conflicting
source changes. Precompiled runtime artifacts require rebuilding.

The downloaded IREE compiler wheel does **not** contain patch 0002. Build
and explicitly select the source compiler to use that optimization:

```sh
bash scripts/build_iree_compiler.sh
export NX_IREE_COMPILER_DIR="$PWD/iree-compiler-build/tools"
mix test
```

The script initializes LLVM and StableHLO submodules, applies the patches,
and builds `iree-compile`, `iree-opt` and `FileCheck`. It enables portable
Vulkan, VMVX and LLVM CPU code generation for X86/AArch64, including the
local runtime required for compile-time constant evaluation. `CC`, `CXX`
and `IREE_LINKER` select the host toolchain; Clang/lld 19.1.7 was used here.
`IREE_CPU_TARGETS` changes the CPU architectures, `IREE_BUILD_JOBS` defaults
to eight, and `NX_IREE_COMPILER_BUILD_DIR` changes the build directory.
Additional arguments are passed to CMake. Build outside the Nerves cross
environment; the resulting compiler runs on the host.

Unsetting `NX_IREE_COMPILER_DIR` restores the wheel on the next Mix build.
Use distinct module-cache directories when comparing compiler versions.

See [the recurrence investigation](../../docs/stereo-path.md) for real GPU
evidence, performance limits and links to the regression logs.
