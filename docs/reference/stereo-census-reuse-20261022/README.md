# Census SGM: library reuse and second-sequence validation

The complete eight-direction Vulkan matcher reaches **28.03 FPS on C100 for
Sintel alley_1 and 29.00 FPS for temple_2**, at 320×136 / D=64 with normal
devfreq. Accuracy is close to the fast CPU reference across these two
sequences. It still does **not** beat CPU OpenCV 3WAY, which measures
100.78 and 93.71 FPS respectively. This remains a benchmark candidate, not
an accepted production package or a completed performance goal.

A three-direction comparison reaches **38.57 / 36.08 FPS**, with somewhat
worse alley_1 accuracy. Reducing work improves throughput, but does not
establish a faster equivalent GPU implementation.

The subsequent [projected-right-map experiment and runtime investigation](../stereo-projection-20261022/README.md)
did not improve throughput. It also records a confirmed generic IREE input
flush fix and an unresolved Nouveau timeout during rapid native submissions.
The native profiling run that encountered that timeout is rejected.

## Measured comparison

All configurations process all 50 prepared rectified pairs per sequence,
after three warmups. Timings include input transfers, complete matching,
output transfer and synchronization. Loading executables, decode, resize
and disk I/O are excluded. CPU uses four OpenCV threads through Evision;
every matching stage of the Vulkan candidate runs on GPU.

The common region is x≥64, excluding occluded/out-of-frame ground truth.
Rejected pixels count as errors. This avoids rewarding OpenCV for excluding
the first 64 columns. Matching costs, path counts, right-map construction
and cleanup differ; these are quality/performance comparisons, not identical
algorithms or bitwise OpenCV reproductions.

| Sequence | Pipeline | FPS | p50 / p95 ms | Bad1 or rejected | Bad3 or rejected | Accepted MAE px |
|---|---|---:|---:|---:|---:|---:|
| alley_1 | CPU OpenCV 3WAY | 100.78 | 9.367 / 13.388 | 4.434% | 2.744% | 0.4381 |
| alley_1 | CPU OpenCV HH4 | 63.65 | 15.252 / 18.486 | 4.389% | 2.751% | 0.4259 |
| alley_1 | Vulkan, eight directions | 28.03 | 33.963 / 43.368 | 4.360% | 2.274% | 0.4793 |
| alley_1 | Vulkan, three directions | 38.57 | 26.847 / 33.029 | 4.849% | 2.585% | 0.5187 |
| temple_2 | CPU OpenCV 3WAY | 93.71 | 9.352 / 15.779 | 2.284% | 0.789% | 0.2502 |
| temple_2 | CPU OpenCV HH4 | 66.87 | 14.411 / 17.230 | 2.450% | 0.771% | 0.2501 |
| temple_2 | Vulkan, eight directions | 29.00 | 33.090 / 39.049 | 2.397% | 0.631% | 0.2341 |
| temple_2 | Vulkan, three directions | 36.08 | 26.586 / 37.824 | 2.353% | 0.765% | 0.2462 |

Alley CPU results are from the [parameter-verified library survey](../stereo-library-survey-20261022/README.md).
Held-out CPU runs and the preceding GPU configuration are in
[this log](cpu-heldout-and-previous-gpu.log). Final GPU runs are
[eight-direction](board.log) and [three-direction](three-board.log).
Downloaded board predictions produced the quality files:
[eight/alley](quality-alley.json), [eight/temple](quality-temple.json),
[three/alley](quality-three-alley.json), [three/temple](quality-three-temple.json)
and [CPU/temple](quality-cpu-temple.json).

Eight-direction coverage is 99.075% / 99.687%. Its alley accepted-pixel MAE
is worse than CPU 3WAY despite better bad1/bad3-or-rejected rates; no single
configuration wins every accuracy metric. No visible ground-truth disparity
exceeds the configured search range on either sequence (maxima 42.637 /
48.457 pixels). These two resized scenes do not establish general accuracy.

P1=8, P2=64 and uniqueness=1/0.95 were selected from three classic-Census
configurations on alley_1: [training results](training-parameters.json).
They were frozen before evaluating temple_2, and remain unchanged for the
three-direction comparison. Temple_2 is a second scene from the same dataset,
not an independent sensor/domain validation. The [fetch log](heldout-fetch.log)
records the exact HTTP ranges used to retrieve its 50 clean stereo pairs,
disparities and masks; archive CRC checking and the existing preparation
script were retained. The interrupted full ZIP is not used.

## What was reused

The [portable benchmark source](../../../bench/stereo/census/README.md)
ports selected components of Apache-2.0
[Fixstars libSGM e4c669b](https://github.com/fixstars/libSGM/tree/e4c669bb9e039b60f04178b080ac7d6077a4d4bd).
Upstream itself remains CUDA-only. The GLSL adaptation preserves license
notices and runs through the existing NxIREE/IREE Vulkan runtime on NVK and
RADV. There is no new target executor, allocator, NIF, firmware, board policy
or production dependency in this change.

The follow-up reuses:

- The classic 9×7 Census transform: 48 comparisons, excluding the center
  row and column. Two u32 words avoid requiring GPU integer-64 arithmetic.
  The initial symmetric 31-bit transform remains available.
- Horizontal register-window caching and transposed shared right-feature
  windows for vertical/oblique paths, adapted from
  [cost_aggregation.cu](https://github.com/fixstars/libSGM/blob/e4c669bb9e039b60f04178b080ac7d6077a4d4bd/src/cost_aggregation.cu).
- The upstream shared-memory accumulation tile in
  [winner_takes_all.cu](https://github.com/fixstars/libSGM/blob/e4c669bb9e039b60f04178b080ac7d6077a4d4bd/src/winner_takes_all.cu),
  combining summation and both selections without a global summed-volume
  buffer. The portable tile uses packed pairs of u16 scores in u32 storage.

The initial symmetric matcher reached 19.11 FPS with substantially worse
accuracy: [historical report](../stereo-census-20261022/README.md). Switching
to classic Census, reusing the path caches, and freezing the selected
parameters produced 26.58 / 26.04 FPS. Fusing selection produced 27.48 /
27.72 FPS. Constant-index register selection then produced the final
28.03 / 29.00 FPS. The latter are individual 50-frame measurements under
normal devfreq, not confidence intervals proving every small change faster.

The fused and separate-stage versions produce byte-identical board outputs
on both sequences. The register-index change also preserves every output:
[agreement](agreement.log). Host and board validity masks match, with maximum
absolute float difference 3.815e-6 pixels over all 100 final frames.
Host eight-direction runs measure 590.41 / 620.99 FPS:
[alley](host-alley.log), [temple](host-temple.log). Those numbers are not a
C100 performance forecast. [Three-direction agreement](three-agreement.log)
likewise checks all 100 host/board predictions.

## Why CPU is still faster

This is not a demonstration that dense stereo inherently favors CPU.
OpenCV already uses SIMD and threaded stripes; its 3WAY implementation
combines left, right and top paths and fuses accumulation with selection.
Its implementation is in pinned upstream
[stereosgbm.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/stereo/src/stereosgbm.cpp).
Our eight-direction candidate does more aggregation, and each GPU scan
still has sequential dependencies along its line. The three-direction
experiment controls that path-count difference without reproducing the CPU
cost function or all its scheduling optimizations.

Native upstream `iree-benchmark-module`, cross-built from the existing
IREE runtime tree, independently measured the initial path shader at
**29.9 ms**: [native baseline](native-paths-baseline.log). This matches
29.6 ms measured through Elixir. Specializing direction branches alone gave
29.7 ms; horizontal caching reduced it to **18.3 ms**:
[native schedule comparison](native-schedules.log). These are coarse native
invocation timings, not pure GPU timestamps. The tool's dispatch-statistics
flag produced no usable records for these opaque shaders. The comparison
isolates Elixir as a minor contributor to that initial dominant stage.

Read-only host GM20B drm-shim compiler inspection exposed eight bytes of
per-thread local memory for an indirect two-element register lookup in
selection. Explicit constant-index choices remove it:
[before](nak-before-register-change.log), [after](nak-after-register-change.log).
Selection changes from 919 to 916 instructions, local storage 8 to 0 bytes;
path code uses 48 GPRs and no spills. This is host compiler evidence, not a
hardware trace or a claim that local-memory removal alone explains the
whole speed difference. Disparity blocks of 4 and 16 were slower than 8:
[board trials](disparity-blocks-board.log). The default remains 8.

[Final synchronous stage means](stage-means.log), alley_1:

| Stage | Eight directions, ms | Three directions, ms |
|---|---:|---:|
| Paths | 16.294 | 6.628 |
| Fused sum and both selections | 9.604 | 10.246 |
| Census | 2.537 | 2.455 |
| Median and consistency | 1.956 | 1.778 |
| Upload | 3.425 | 3.220 |
| Readback | 1.645 | 1.419 |

These include allocation, native runtime work and completion waits. They
are not GPU timestamp measurements. Intermediates remain device-resident;
there is no hidden CPU matcher or intermediate CPU copy. Small images also
leave fixed invocation/transfer costs significant. The
[three-direction stage means](three-stage-means.log) identify selection as
the largest remaining stage after reducing aggregation work.

The libSGM-style rolling right minimum serializes an entire row. The
subsequent projected-right-map experiment implements another construction,
but its gather version is slower; the follow-up above records the results.
Composing dispatches into a single IREE function remains unimplemented.
More CPU-side Elixir tuning cannot by itself remove the measured GPU-stage
costs.

## Correctness and reproduction

Independent fixtures validate Census features, directional costs, sums
where separately materialized, both disparity maps, median/consistency
output and preservation of caller inputs. Four cases cover known shift,
unrelated textures, ambiguity and opposite constant images. Widths 73/137
and height 17 exercise incomplete tiles, right-map tails and partial groups.

- D=64 / eight paths: [host](fixtures-host.log), first child in [board.log](board.log).
- D=128 / eight paths: [host](d128-fixtures-host.log), second board child.
- D=64 / four paths: [host](four-fixtures-host.log), third board child.
- D=64 / three paths: [host](three-fixtures-host.log), first child in [three-board.log](three-board.log).
- The shared scalar selector gained optional uniqueness; all 28 original
  SAD fixture files remain byte-identical: [regression](sad-oracle-regression.log).

SPIR-V validation occurs for every shader during export. Ruff, strict
basedpyright and Elixir formatting pass. All actual fixture/video GPU runs
completed with GPU_ERRORS=0. A rejected invocation initially pointed at a
nonexistent input directory, stopped before processing, and was corrected;
its child crash dump was retained separately. It was not a GPU failure.

Use the compiler/toolchain setup described in the benchmark README. From
`bench/`, export the final eight-direction candidate with:

```sh
VIDEO_CENSUS=classic VIDEO_PATH_SCHEDULE=tiled VIDEO_SELECT=fused \
VIDEO_P1=8 VIDEO_P2=64 VIDEO_UNIQUENESS=1.0526315789473684 \
STEREO_OUT=/tmp/census.bin mix run --no-compile stereo/census/export.exs

STEREO_BUNDLE=/tmp/census.bin VIDEO_DATA=/path/to/alley_1 \
VIDEO_OUTPUT=/tmp/census-result NXIREE_DEVICE_URI='<chosen Vulkan URI>' \
mix run --no-compile stereo/census/run.exs
```

Add `VIDEO_PATHS=3` at export for the measured tradeoff. From the repository
root, generate matching fixtures with:

```sh
PYTHONPATH=bench/stereo/video uv run bench/stereo/census/fixtures.py \
  /tmp/census-fixtures --classic --matcher 8 64 1.0526315789473684
```

Export those at VIDEO_WIDTH=73 / VIDEO_HEIGHT=17. `--disparities 128`
uses width 137; `--paths 3|4` changes path count. For C100, use the existing
guarded child harness in the dedicated `c100_gpu` library, with
STEREO_SUCCESS_MARKER=CENSUS_STEREO_PASS and the staged runtime. Restore
temporary environment overrides with `try/after`; no clock pinning is needed.

[Source and bundle hashes](sha256.txt) identify the measured exports. The
final source also re-exports the eight-direction matcher with identical
module payloads apart from embedded temporary MLIR filenames:
[re-export check](final-source-agreement.log).
The
final eight-direction bundle is staged as `/root/nxiree/census-register-320.bin`,
the three-direction bundle as `/root/nxiree/census-three-320.bin`.
The accepted boot ID, validated slot B and normal idle policy remain:
[final board state](final-board-state.log). No firmware was flashed, no reboot
was performed, and voltage/governor limits were not changed.
