# SGM path correctness and Vulkan repeat benchmark

The new [stereo sequence benchmark](video/README.md) adds image matching,
eight-direction aggregation, disparity selection and consistency checking.
Use it for image-pair FPS and quality; this directory's original path benchmark
remains useful for isolated recurrence work.

This is one directional SGM cost recurrence, not a complete StereoSGBM
matcher. It has no matching-cost generation, direction combination,
disparity selection, subpixel refinement, or consistency/speckle filtering.
The integer scalar reference is independent of the Nx implementation.
Inputs use `{steps, independent_lines, disparities}` with nonnegative s32
costs and penalties in `[0, 1_000_000_000)`. Normalizing and clamping before
adding penalties keeps intermediates within s32, including the large-penalty
regression cases.

The same exported SPIR-V module runs on NVK and RADV. No board identity,
clock setting, or vendor extension is used by the algorithm.

From `bench/`:

```sh
mix run stereo/check.exs
STEREO_OUT=/tmp/stereo.bin mix run stereo/export.exs
mix run -e 'IO.inspect(NxIREE.list_devices("vulkan"))'
STEREO_BUNDLE=/tmp/stereo.bin NXIREE_DEVICE_URI='vulkan://GPU-…' \
  mix run stereo/run.exs
```

Select an exact URI when more than one Vulkan device is available.
`STEREO_REPEATS` defaults to ten. Each result is checked in full, then its
device buffer is released; resident inputs are released after each case.
Repeats run continuously. `STEREO_PAUSE_MS=100` reproduces the earlier idle
pauses, which let normal devfreq ramp down between invocations on the C100.
`PATH_BENCH` includes invocation and output readback. `PATH_PHASES` reports
them separately; the compiled function waits for GPU completion before
returning, so invocation is more than queue submission. Tests stop at the first
incorrect output, thermal limit (70°C), or configured memory guard.
`STEREO_MIN_AVAILABLE_KB` enables Linux system-memory checks between calls;
it cannot limit memory allocated inside a single invocation.

## Whole-line Vulkan kernels

The optimized implementation keeps the recurrence inside one GPU dispatch,
with independent lines assigned to workgroups and disparities distributed
across threads. The application shaders are imported through the general
[`NxIREE.Vulkan.export/4`](../../docs/vulkan-kernels.md) API. This is an explicit
kernel path; ordinary Nx `while` expressions are not automatically fused.

Install glslang and SPIRV-Tools on the build host, or set `GLSLANG` and
`SPIRV_VAL` to their executable paths. Then, from `bench/`:

```sh
STEREO_KERNEL=subgroup STEREO_OUT=/tmp/stereo-vulkan.bin \
  mix run stereo/export_vulkan.exs
STEREO_BUNDLE=/tmp/stereo-vulkan.bin NXIREE_DEVICE_URI='vulkan://GPU-…' \
  mix run stereo/run.exs
```

Three variants use the same inputs, reference and runtime runner:

- `shared`: SPIR-V 1.3 Shader only, with shared-memory reductions. Workgroup
  size is the next power of two covering the disparities, up to 128.
- `subgroup` (default): additionally requires subgroup arithmetic and shuffle.
  Uses 32 invocations with multiple disparity values per thread. It checks
  that the workgroup occupies one subgroup and that lane IDs match before
  using shuffles; otherwise it executes its shared-memory fallback.
- `subgroup_shared`: forces that fallback, for correctness testing.

A device lacking subgroup capabilities must use `shared`; the internal
fallback does not remove the module's declared capability requirements.
Both GPU implementations test the same arithmetic, including counts 1, 2,
31, 32, 33, 63, 64, 65, 97 and 128, minima on chunk boundaries, changed inputs
and penalties approaching one billion. Every output is checked against the
independent scalar implementation and inputs must remain unchanged.

The compiled modules embed their shaders and run on the existing target
runtime; no target GLSL compiler, new NIF, firmware upload or reboot is needed.

For runtime-only deployment, set `STEREO_RUNTIME_ROOT` to a directory with
`lib/{complex,telemetry,nx,nx_iree}/ebin`, and set `STEREO_BUNDLE` to the
exported bundle. This needs neither EXLA nor the IREE compiler on the device.

The dedicated `c100_gpu` library owns `scripts/stereo_board_run.exs`, the
C100 harness. It launches a child BEAM with
a 120-second deadline, checks dmesg, and leaves GPU devfreq unchanged.
It expects scripts and bundle under `/root/nxiree`; `STEREO_RUNTIME_ROOT`
can select a separately staged runtime. From the board's Elixir SSH shell:

```elixir
System.put_env("STEREO_RUNTIME_ROOT", "/root/nxiree-stereo")
Code.eval_file("/root/nxiree/stereo-board-run.exs")
```

See [measurements and limitations](../../docs/stereo-path.md).
