# Portable Vulkan stereo work, 2026-10-22

The stereo matcher is now the [`nx_stereo`](../../nx_stereo) library, an
`NxIREE.Vulkan.Pipeline` of the shaders this work produced. This page records
the work that led there; `bench/stereo` and `NxIREE.Vulkan.export/4`, which
it refers to, are in nx_iree's history up to commit cae4115.

The target is stereo SGBM on the OKdo C100 through Elixir and Vulkan,
without CUDA, approaching the board's NVIDIA performance. Runtime and
compiler improvements must apply to general Vulkan workloads. The host's
AMD discrete GPU is a second validation target. Resolution, disparity
range, FPS and exact OpenCV compatibility remain application choices. CPU
parity is not success: the GPU must provide a worthwhile end-to-end
throughput advantage at comparable quality while leaving CPU capacity for
Elixir. The current full-pipeline reference is 320×136/D64 on two Sintel
sequences, compared with optimized four-thread OpenCV.

The latest [CPU-coexistence test](reference/stereo-concurrency-20261022/README.md)
processes 1,000 pairs per run with normal CPU/GPU governors. GPU throughput
is 147–148 FPS without background work and 150–152 FPS with two CPU workers;
CPU 3WAY gives 84–85 and 51–52 FPS respectively. Under contention the GPU
therefore has about a 2.9× lead, using about 0.71 CPU cores while the other
workers use about two. All repeated copies agree and all 400 downloaded
maps are unchanged. Camera capture/pacing, sustained driver acceptance and
standard Evision GPU SGBM integration remain open.

The latest [fusion and readback measurements](reference/stereo-fused-20261022/README.md)
reach 121–140 whole-loop GPU FPS versus 85–89 CPU FPS with matched temporary
CPU-performance policy. Normal CPU schedutil gives about 101–104 GPU FPS.
The GPU uses much less CPU time and preserves all 100 earlier predictions.
This establishes a conditional prototype throughput lead, not production
acceptance or NVIDIA-equivalent performance. The earlier native submission
failure remains unresolved, and no default board policy has changed.

The latest [device-timestamp and saturation measurements](reference/stereo-saturation-20261022/README.md)
identify aggregation and selection as about 88% of warmed GPU execution.
The existing mapping takes about 7.26 ms of device work in that capture;
this describes the implementation and measured conditions, not a hardware
limit. Useful large gains require reducing those stages and pipeline costs.

Device-specific code belongs in a dedicated Buildroot package or Elixir
library. `~/Projects/c100_gpu` owns the Maxwell tuning provider, NVK target
capabilities, and C100 board harness; nx_iree exposes generic tuning providers.

## Latest result: one dispatch per direction

`NxIREE.Vulkan.export/4` now imports arbitrary compatible SPIR-V compute
shaders into the existing Elixir export/load/run interface. It supports
static s32/u32/f32 tensor bindings and one or multiple outputs, without
device-specific behavior. The stereo shaders live in `bench/stereo`, using
this general API. This is an explicit application kernel path, not automatic
fusion of arbitrary Nx loops.

The earlier compiler emitted about 480 dispatches for a 160-step direction,
and assigned each line's entire reduction serially to one invocation. The
new shader runs the whole line in one dispatch and distributes disparities
across threads. A basic shared-memory version needs only SPIR-V 1.3 Shader.
A faster variant uses subgroup arithmetic/shuffle with a checked lane
mapping and a shared-memory fallback. The separate basic shader remains
necessary for devices without those optional subgroup capabilities.

Final controlled C100 comparison, accepted 921.6 MHz cap, ten calls per case:

| 160 × 120 × 64 direction | Invocation median | Total median | Total range |
| --- | --- | --- | --- |
| Previous compiler-optimized Nx path | 96.739 ms | 109.407 ms | 103.945–113.175 ms |
| Whole-line shared shader | 5.072 ms | 13.148 ms | 12.738–16.064 ms |
| Whole-line subgroup shader | **3.402 ms** | **11.138 ms** | **11.004–14.838 ms** |
| Forced shared fallback | 4.501 ms | 12.561 ms | 12.103–15.748 ms |

The subgroup path reduces median total time by **9.82×**, and invocation
time by **28.44×**. Invocation includes runtime overhead, allocation and
completion wait; it is not a GPU timestamp measurement. Final subgroup
invocations ranged 3.356–3.469 ms. Readback of the 4,915,200-byte cost volume
took 7.595–11.364 ms (median 7.753 ms). The generated HAL function contains
`hal.fence.await` before returning, so the invocation measurement includes
finished GPU work.

Normal devfreq remains variable: one final subgroup run took 12.469–38.136 ms
including readback. The minimum was restored to 76.8 MHz after controlled
testing; governor, voltage policy, firmware and runtime binaries were not
changed. The same exports pass on host RADV, where readback similarly
dominates the now short invocation. See the captured logs for every sample.

Sixteen scalar-reference fixtures plus two benchmark volumes pass on both
GPUs for all three shader variants, including forced fallback. Each run
also checks ten repeated calls per benchmark and unchanged caller inputs.
Counts around 32/64 boundaries and up to 128 disparities, changed inputs,
and large penalties are covered. The Nx evaluator and IREE CPU pass all
32 fixture checks. Extending the range tests found two earlier arithmetic
defects: adding penalties could overflow before taking a minimum, and the
finite boundary sentinel could win incorrectly. Normalizing and clamping
to P2 before addition fixes both in the Nx baseline and shaders. A captured
GPU run fails the old arithmetic; the same case passes the final kernels.
The supported arithmetic domain is costs/P1/P2 in `[0, 1_000_000_000)`.

Final regression suites pass: **805 CPU checks** (744 doctests, 61 tests;
one Vulkan-specific test excluded) and **741 Vulkan checks** (679 doctests,
62 tests), including independent signed integer and float/unsigned custom
shaders, tuple outputs, changed inputs and resident-input preservation.
An additional `glslang -Os` comparison reduced shader size but did not improve
board invocation time, so that compiler option is not enabled.

This is still one SGM direction, not a complete matcher or a measured frame
rate. Four identical-cost invocations would use about 13.6 ms of the current
invocation budget, and eight about 27.2 ms, before cost generation, combining
directions, disparity selection and filtering. These are arithmetic budgets,
not forecasts: horizontal/vertical/diagonal access patterns and occupancy
must be measured in the actual pipeline. Keep intermediate tensors on the
GPU and read back the final disparity image instead of each full cost volume.
No NVIDIA performance-parity claim is established.

Reproduce using [the whole-line kernel instructions](../bench/stereo/README.md).
Logs, the failing arithmetic case and final code hashes are under
[reference/stereo-kernels-20261022](reference/stereo-kernels-20261022/).

## Generic allocator fix

IREE source revision `89ece87b89852100c0a4ac3d515e14842d090967` (v3.13.0)
can overlook reusable TLSF slabs: it searches one preferred slab and four
cached candidates, then grows without searching the remaining slabs.
Released memory becomes stranded even when live reservations are stable.

`patches/iree/0001-reuse-all-free-tlsf-slabs.patch` adds a fallback search
through unvisited slabs, preserving the fast paths and frontier checks.
This is shared allocator code, with no Vulkan-vendor or SGM special case.
An eight-slab regression fails before the fix and passes after it; all
15 TLSF pool tests pass. Source builds apply the patch through the Makefile;
both fresh and repeated application were tested. Existing precompiled
runtime artifacts must be rebuilt to contain the fix.

The same AOT module and inputs reproduce the defect on the C100 and the
host's AMD Navi32 discrete GPU (16 GiB VRAM, RADV). The host device URI is
`vulkan://GPU-00000000030000000000000000000000` on this machine.

| Measurement | Before | After |
| --- | --- | --- |
| Host live reservations after each large call | 76,940,032 bytes | 76,940,032 bytes |
| Host backing slabs across repeated calls | 31, 48, 64, 78, 91 | 12, 13, 13, 13, 13, 13 |
| Host backing memory | grows to 6.11 GB | settles at 872 MB |
| C100 large recurrence repeats | child OOM kill | ten repeats pass |
| C100 available memory after warmup | exhausted | about 2.5 GB |

Allocator statistics were temporary instrumentation, removed from the
validated runtime. Explicit output release and Elixir GC did not fix the
unpatched allocator. No GPU trap accompanied the original OOM.

## Earlier correctness and timings

Seven edge/reference fixtures pass exactly on the Nx evaluator, IREE CPU,
RADV and NVK: first pixel, one disparity, boundary exclusion, zero penalties,
large common offset, multiple independent lines, and changed inputs for a
reused model. Both benchmark volumes also match the scalar reference in full.

The clean runtime passed ten measured repeats per size on both GPUs:

| Shape (steps × lines × disparities) | Values | Host RADV | C100 NVK |
| --- | --- | --- | --- |
| 32 × 24 × 32 | 24,576 | 5.5–7.8 ms | 38.7–79.5 ms |
| 160 × 120 × 64 | 1,228,800 | 33.4–36.4 ms | 214–512 ms |

These earlier runs inserted 100 ms idle pauses between measured calls.
Timings include readback and use ordinary device power management. They
are not frame rates or full-SGBM performance. No NVIDIA SGBM baseline was
measured. The original compiler allocates/copies its 4,915,200-byte
accumulated output at every scan step: 811,202,560 bytes of cumulative
allocation requests per invocation. That is not peak live memory.

## Generic compiler fix

`0002-elide-proven-same-device-transfers.patch` makes inferred device
equality usable for copy elision only when all producer paths agree and
ownership permits it. Mixed and unknown devices, borrowed aliases, opaque
forwarders and zero-trip loop results remain protected. All four existing
copy-elision suites and thirteen added scenarios pass. The owned-loop case
fails with the original compiler; additional failing cases caught unsafe
intermediate versions during development.

The final SGM IR has one initial full-volume allocation, no full-volume
copies, and the necessary 30,720-byte row updates. Ordinary NxIREE exports
pass with default and forced VMVX constant evaluation. Every exported shape
reports `written_inputs=[]`; the real GPU checks still verify all caller
inputs after repeated execution. No topology annotation or special dump
flags are needed. Build and select this compiler explicitly using
[the source-build recipe](../patches/iree/README.md); the downloaded wheel
does not contain the compiler patch.

The full NxIREE suites also pass with the patched compiler: 798 CPU cases
(744 doctests, 54 tests) and 733 Vulkan cases (679 doctests, 54 tests).

Both GPUs pass all fixtures and ten continuous repeats per benchmark size:

| Large recurrence, invocation plus readback | Original compiler | Patched compiler |
| --- | --- | --- |
| Host RADV, continuous calls | 32.6–34.6 ms | 25.6–28.5 ms |
| C100, controlled comparison at 921.6 MHz | 198–223 ms | 107–122 ms |
| C100 available memory after warmup | about 2.5 GB | about 3.3 GB |

Normal C100 devfreq produces variable latency with both compilers. Recorded
frequencies span 76.8–921.6 MHz; the patched large loop often runs at
153.6–384 MHz. One normal patched run took 144–475 ms. The controlled
comparison temporarily set the minimum to the accepted maximum through
`c100_gpu/scripts/compare_stereo_at_max.exs`, then restored the original
76.8 MHz minimum in `after`. The governor remained `simple_ondemand`.
The fixed-frequency numbers establish a compiler benefit, not a new default
power policy or a claim of NVIDIA-equivalent SGBM performance.

An experimental single-device topology annotation removes the full-volume
copy and allocation without changing arithmetic or consuming inputs. Both
GPUs pass exact results and input-preservation checks with it. On the C100,
available memory then settles around 3.3 GB; latency remains about 295–505 ms.
Host latency is about 28–29 ms. This annotation is not enabled in production:
the compiler repair above replaces it with affinity and ownership proofs.
That experiment used custom phase dumps, causing NxIREE's conservative
input-write fallback to copy caller inputs. Its input-preservation result
therefore does not independently prove compiler ownership safety; dedicated
compiler regressions cover borrowed aliases and mixed ownership.

The remaining compiled loop has two `slow_memcpy` dispatches that materialize
padded previous rows, then one reduction dispatch. Constant padding feeds
`tensor.extract_slice`, so IREE's direct-pad-to-Linalg fusion exception does
not apply; padding lowers to fill plus insert before dispatch fusion.
Generic fusion across pad, slice and arithmetic is a subsequent compiler
task. The loop induction value already lives on the host and only the final
fence is awaited, so the existing device-indexing rewrite addresses no
observed per-step readback here. Forced indirect command buffers compile,
but have not yet been measured and are not enabled by default.

An isolated experiment with IREE's existing pad/slice swap pass and
consumer-padding fusion removed the two fills but retained three dispatches.
Both GPUs passed ten large-volume repeats with exact results, empty input
write sets and unchanged caller inputs. C100 latency was 305–419 ms, with
about 3.27 GB available; host latency was 33.5–37.8 ms during a concurrent
compiler build. This does not establish a speed gain, so these pass changes
are not integrated.

## C100 baseline and reproduction

Firmware `1ffb5cb3-94bd-54a7-f1e2-ec9661ce6746` remains on validated slot B,
with normal `simple_ondemand`, 76.8 MHz idle and 921.6 MHz maximum. It had
already passed 30 consecutive warm reboots with the kernel fixes. These
stereo tests did not reboot or flash it. The patched test runtime is staged
at `/root/nxiree-stereo`; the earlier `/root/nxiree` runtime is preserved.

Cross runtime SHA256:
`30c77510f56f7d915a8432da1838e0f39160cd040dfe6d88527c38b6dd7bb1dc`.

The system's `package/c100-gpu` now installs the unchanged `c100-limits`
script. Buildroot installation and the running board script both hash to
`40215f4427b6915cfff27a7e6701b9f75b1770c5932361989fba8c29e5d775a1`.
The packaging move has not been flashed. Live `vulkaninfo` confirms the
8/16-bit arithmetic/storage features declared by `C100GPU.Target`.

Reproduce with [bench/stereo](../bench/stereo/README.md). Captured evidence
is in [reference/stereo-path-20261022](reference/stereo-path-20261022/).
