# Reusing stereo and GPU libraries, 2026-10-22

Follow-up: the [Vulkan libSGM port experiment](../stereo-census-reuse-20261022/README.md)
now runs the complete eight-direction GPU matcher at 28–29 FPS on C100.
Its accuracy is close to the fast CPU reference across two sequences, but
CPU 3WAY still reaches 94–101 FPS. The performance goal remains open.

## Recommendation

The production target is a **GPU stereo pipeline that beats optimized CPU
OpenCV at comparable output quality**, while leaving CPU capacity for Elixir
and the application. The measured CPU results below are the baseline to beat.
They do not satisfy the GPU goal, even where they exceed 20 FPS. The current
10.39 FPS Vulkan prototype has not met that performance target.

Prefer established GPU libraries and upstream implementations, packaged through
Buildroot or a dedicated Elixir library, with thin integration code. Keep IREE
for generic Vulkan execution and reuse its command-buffer, memory-management
and profiling facilities. Evaluate supported OpenCL kernels through clspv/clvk
before undertaking a larger port. Clearly licensed libSGM components remain a
candidate when no compatible implementation can be reused directly. Adding
another Rust/C++ Vulkan wrapper alone will not remove SGM work.

Evision/OpenCV can supply the Elixir vision interface, calibration and an
independent accuracy/performance reference. Its tested StereoSGBM path is CPU
code. Only operations with verified GPU execution belong in the accelerated
frame pipeline; enabling a build option is insufficient. No OpenCV, clvk or
SGM production package has been installed by this survey.

Acceptance requires named CPU and GPU configurations with comparable
directions, consistency checks and measured accuracy, including frame transfers
and synchronization in latency. Keep frame intermediates on the GPU. Verify
GPU execution and distinguish any CPU fallback explicitly. Compare throughput,
p95 latency and CPU use as well as disparity quality. Test the generic GPU
implementation on host RADV and C100 NVK; board policy remains in `c100_gpu`
or the system's dedicated Buildroot package. A lower-quality mode is a separate
tradeoff, not proof of a faster equivalent implementation.

## Measured Evision performance and quality

Published Evision 1.0.0, non-contrib ARM64 Linux GNU NIF, OpenCV 5.0.0.
The archive's SHA256 is
`5ab2cce1871f2e9f827998e0af18fec4e2a9f772829bc753d96f6c83f45f7c68`.
The [build information](evision-build-info-board.log) reports NEON, pthreads
and no GUI. It was loaded into an isolated child BEAM on the same Nerves
firmware, with normal CPU/GPU policy and four OpenCV threads.

All 50 Sintel `alley_1` clean pairs were processed after three warmups per
mode. Timings include Elixir Mat creation, the native matcher, f32 output
conversion and binary copying. Decode, resizing, disk I/O and startup are
excluded. P1=72, P2=288, D=64, block=3, uniqueness=10, disp12MaxDiff=1,
prefilter cap=31, speckle filtering off. Native penalty values are verified.

| Implementation | FPS | p50 ms | p95 ms | Common-region bad1 or rejected | Common-region bad3 or rejected |
|---|---:|---:|---:|---:|---:|
| Evision default SGBM, five directions | 25.58 | 38.177 | 42.402 | 4.377% | 2.486% |
| Evision HH, eight directions | 15.64 | 65.662 | 65.954 | 4.085% | 2.511% |
| Evision 3WAY, three directions | 100.78 | 9.367 | 13.388 | 4.434% | 2.744% |
| Evision HH4, four directions | 63.65 | 15.252 | 18.486 | 4.389% | 2.751% |
| Current Vulkan block-SAD, eight directions per view | 10.39 | 94.837 | 101.344 | 4.269% | 3.171% |

Sources: [correctly configured benchmark](evision-board.log),
[quality metrics](evision-quality.json),
[Vulkan report](../stereo-efficiency-20261022/README.md).

The common region is x≥64 over non-occluded/in-frame ground truth. Full-frame
coverage is also in the JSON; OpenCV's invalid left border makes full-frame
scores unsuitable for a direct comparison without that common region.
Common coverage is 98.57% for 3WAY versus 97.34% for the Vulkan reference.
Their accepted-pixel MAEs are 0.438 and 0.307 pixels respectively. Thus no
single metric establishes a universal winner. This is one resized synthetic
sequence, not a broad quality or temporal-stability evaluation.

OpenCV uses its Birchfield–Tomasi/block cost, internal derived-right check,
median filtering and fixed-point subpixel output. Our Vulkan reference uses
block SAD and independently aggregates both views. The table compares useful
pipelines, not equivalent workloads. It demonstrates the value of reusing an
optimized existing implementation; it does not demonstrate CPU superiority
over an optimized GPU algorithm.

## Evision integration findings

[Evision exposes StereoSGBM directly](https://evision.hexdocs.pm/Evision.StereoSGBM.html).
Its 1.0.0 release targets OpenCV 5.0.0 and declares Nx ~>0.12.1:
[release source](https://github.com/cocoa-xu/evision/blob/v1.0.0/mix.exs).
The CPU SGBM implementation reads Mats and uses CPU vector operations:
[pinned OpenCV source](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/stereo/src/stereosgbm.cpp).
The separate [CUDA StereoSGM source](https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudastereo/src/stereosgm.cpp)
does not supply a Vulkan implementation. Enabling OpenCL in an OpenCV build
does not turn this SGBM path into a GPU matcher.

The first Evision run exposed a real constructor binding problem: accepted
`P1: 72, P2: 288` options produced native `getP1=0`, `getP2=0` while the other
requested parameters matched. [Observed values](evision-parameters-observed.log).
The benchmark's parameter assertion then failed with
`getP1: expected 72, got 0`: [red proof](evision-parameters-red.log).
Positional `setP1(matcher,72)` and `setP2(matcher,288)` make the same assertion
pass and restore the expected quality. The final table uses this corrected
run. [Initial zero-penalty metrics](evision-quality-initial-zero-penalties.json)
are retained solely as superseded diagnostic evidence. No upstream source
was patched or issue published.

The benchmark project is isolated to preserve the NxIREE dependency graph.
A separate child successfully used this Evision build with the staged
**Nx 1.0.0**: Nx→Mat→Nx BinaryBackend roundtrip, plus Evision.Backend Nx.add.
[Interop proof](evision-nx1-interop-board.log). This supports feasibility but
does not validate every backend callback or resolve the declared Mix version
constraint. Production integration must explicitly align/override and test
that dependency, or use an isolated worker.

The precompiled runtime is staged at `/root/evision-survey`; the main Nerves
application, original NxIREE runtime and firmware are unchanged. The initial
staging archive's hardlink aliases needed restoring after Erlang extraction;
this was packaging of the temporary runtime, not an SGBM failure.

## Reusable libraries and tools

The following are source-backed assessments. Only Evision/OpenCV was run on
C100 in this survey; the other performance effects remain unmeasured.

| Candidate | What can be reused | Fit and decision |
|---|---|---|
| **Evision + OpenCV C++** | Existing Elixir API, CPU SGBM, calibration/rectification, filtering, Nx conversion | Useful interface/reference. Tested SGBM is CPU-only and cannot meet GPU acceptance. Verify the backend of any proposed per-frame operation. |
| **libSGM, C++/CUDA** | Census, compact directional states, simultaneous left/right selection, filtering, consistency and test cases | Primary algorithm port source. Apache-2.0. CUDA kernels require adaptation to Vulkan; it is not a native Vulkan runtime. |
| **IREE C HAL + profiling** | Command buffers, dispatches, barriers, queue submission, resident buffers and profiling | First choice for batching because it is already our runtime. Expose a thin generic graph interface instead of replacing this infrastructure. |
| **Kompute, C++ Vulkan** | Recorded sequences, asynchronous submit/wait, timestamp queries, reusable tensors | Good independent comparison harness or fallback executor. Apache-2.0. A second production runtime would add buffer/lifetime interop work. |
| **clspv + clvk, C++** | Compile supported OpenCL C to Vulkan SPIR-V; run OpenCL APIs over Vulkan | Practical route to evaluate existing OpenCL kernels. Compilation/ABI/capability compatibility still requires tests; neither tool supplies SGM. |
| **stereo-sgm-opencl, C++/OpenCL** | Existing SGM host code and kernels | Not a first production choice: its README explicitly reports broken oblique/eight-path optimization; no root LICENSE or per-file license header was found in the inspected core files. Resolve licensing and correctness before code reuse. |
| **wgpu-native, Rust with C API** | Portable compute pipeline, command encoding, buffers and queue abstraction | Credible runtime alternative; no SGM implementation supplied. MIT OR Apache-2.0. Replacing IREE has no established performance benefit here. |
| **Vulkano, Rust** | Typed Vulkan command-buffer construction and resource management | Credible for a separate executor; no turnkey SGM found. Adds a second resource ecosystem if combined with NxIREE. |
| **sgm-rs, Rust** | Small Census/SGM CPU implementation | Source uses scalar loops with image/ndarray dependencies, not a Vulkan compute backend. Useful educational reference; not an identified route to beat the tested OpenCV CPU implementation. Cargo declares MIT; no root license file was observed. |

Primary source anchors:

- [libSGM selection code](https://github.com/fixstars/libSGM/blob/e4c669bb9e039b60f04178b080ac7d6077a4d4bd/src/winner_takes_all.cu)
  produces left/right maps from the same aggregated input. Preserve upstream
  notices when adapting code; its CUDA implementation cannot simply be linked
  into a CUDA-free Vulkan deployment.
- [Kompute Sequence](https://github.com/KomputeProject/kompute/blob/267e019ec419ae57f634b341983b60972924b58f/src/include/kompute/Sequence.hpp)
  provides recording, async evaluation/wait and timestamp retrieval.
- [IREE profiling](https://iree.dev/developers/performance/profiling/) and
  [Tracy support](https://iree.dev/developers/performance/profiling-with-tracy/)
  provide existing instrumentation. The installed C HAL headers also expose
  `iree_hal_command_buffer_create/dispatch/execution_barrier`,
  `iree_hal_device_queue_execute`, and device profiling begin/flush/end.
- [clspv](https://github.com/google/clspv/tree/20a0f004cf7c81879a906eabacef48855ebe8195)
  and [clvk](https://github.com/kpet/clvk/tree/3c155df22f9c30ef2c096469e9fc382d023aa899)
  supply the compiler/runtime bridge. clspv's descriptor ABI is not automatically
  identical to our initial flat-buffer custom-shader API.
- [OpenCL SGM known issues](https://github.com/siposcsaba89/stereo-sgm-opencl/blob/8cf8b251dbe2d4ebdc37faa20e7457205d03af63/README.md).
- [wgpu-native](https://github.com/gfx-rs/wgpu-native/tree/589e6caa4e9fa5596f601e82ca5d3937a1afc829),
  [Vulkano command builder](https://github.com/vulkano-rs/vulkano/blob/9a80c29e7f8a90d7f39564b31769d6a9c85f6cc4/vulkano/src/command_buffer/auto/builder.rs),
  [sgm-rs implementation](https://github.com/neka-nat/sgm-rs/blob/85292ea0f0a109bfd907f40c55dd2ee8917b834c/src/sgm.rs).

Repo identities and inspected revisions are saved in [repositories.jsonl](repositories.jsonl).
No complete, verified, ready-to-use Vulkan SGM library was identified in this
search. That is a search result, not a claim that none exists.

## Consequences for the GPU work

1. Use the tested Evision/OpenCV results as the CPU performance and accuracy
   baseline. The target is faster GPU processing at comparable quality, rather
   than adopting CPU SGBM as the delivered stereo implementation.
2. Evaluate existing GPU code before expanding the handwritten matcher.
   OpenCV/Evision can supply interfaces and offline calibration; per-frame
   rectification/filtering still needs a verified GPU backend. Benchmark actual
   capture/rectification/output separately before quoting live-video FPS.
3. Keep generic graph/buffer execution in NxIREE, using IREE's existing HAL.
   Keep board policy in `c100_gpu`/the dedicated Buildroot package.
4. Evaluate a reusable, clearly licensed OpenCL matcher with clspv/clvk, or
   adapt licensed libSGM components to portable SPIR-V where necessary.
   Shared-volume left/right selection is one substantial candidate; test quality
   explicitly because it changes the current two-independent-view algorithm.
5. Require the GPU version to beat the optimized CPU configuration at comparable
   quality, including transfer/synchronization overhead. Do not promise a 2.9×
   speedup from changing wrapper language or importing CUDA source unchanged.

## Artifacts and validation

The reusable benchmark is [bench/stereo/opencv](../../../bench/stereo/opencv/README.md).
Its evaluator reuses the existing independent accuracy accumulator rather
than duplicating metric formulas. Python lint/strict type checks and Elixir
format checks pass. The parameter assertion is a runtime regression check
that was observed failing before the workaround and passing after it.
The final C100 run reports all modes complete, child exit 0 and GPU errors 0.
No complete upstream Evision test suite was run in this targeted assessment.
