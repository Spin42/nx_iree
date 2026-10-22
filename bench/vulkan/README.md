# Portable Vulkan streaming measurement

This benchmark measures completed-runtime throughput of a coalesced
storage-buffer transform: one u32 load, XOR, and one u32 store per element.
It uses the generic `NxIREE.Vulkan.export/4` interface and the same runtime
path as application kernels. No device identity or clock control is embedded.

From `bench/`, set `GLSLANG` and `SPIRV_VAL` to their executable paths:

```sh
VULKAN_BUNDLE=/tmp/stream.bin mix run vulkan/stream_export.exs
VULKAN_BUNDLE=/tmp/stream.bin NXIREE_DEVICE_URI='vulkan://GPU-…' \
  mix run --no-compile vulkan/stream_run.exs
```

The input is resident before timing. Tests use 4 MiB and 16 MiB input/output
buffers, three discarded timing warmups and twenty measured invocations.
Reported logical bandwidth is `(input bytes + output bytes) / median time`,
with nearest-rank median. Timing includes runtime output allocation and
dispatch completion, but excludes upload, readback verification and release.
Full first/last outputs and caller input preservation are checked. A 70°C
thermal guard runs outside timing.

This is not a hardware DRAM counter or a universal memory-throughput ceiling.
Caches, access patterns, runtime overhead and other work affect the result.
It provides a reference for estimating the cost of large streaming buffers
through this particular Vulkan stack.

The runner supports the existing runtime-only `STEREO_RUNTIME_ROOT` loader
convention. For the C100 child-process harness, select this runner through
`STEREO_SCRIPT` and set `STEREO_SUCCESS_MARKER=VULKAN_STREAM_PASS`.
Device controls remain in the dedicated `c100_gpu` library.
