# Census SGM Vulkan port experiment

This adapts the classic and symmetric 9×7 Census, normalized directional recurrence,
byte-sized directional costs, shared-volume left/right selection and median
network from [Fixstars libSGM](https://github.com/fixstars/libSGM/tree/e4c669bb9e039b60f04178b080ac7d6077a4d4bd).
Upstream is CUDA-only. This is a Vulkan port experiment using the existing
NxIREE/IREE runtime, not a claim that upstream libSGM supports Vulkan.
The upstream Apache-2.0 license is preserved in LICENSE and source headers.

Source mappings: `census.comp` ← `src/census_transform.cu`;
`paths.comp` ← `DynamicProgramming`/directional scans in `src/cost_aggregation.cu`;
`sum.comp`/`select.comp` ← `src/winner_takes_all.cu`;
`filter.comp` ← `src/median_filter.cu`/`src/check_consistency.cu`.
The port uses GLSL storage buffers and subgroup shuffles instead of CUDA.
It defines Census windows with replicated image borders, explicitly
excludes negative-coordinate matches and retains float subpixel output.
The default `select_rolling.comp` adapts the upstream rolling right minimum;
`VIDEO_SELECT=gather` retains the initial independent-gather comparison.
`VIDEO_SELECT=fused` also ports the upstream shared-memory accumulation tile,
eliminating the separate summed-volume buffer and dispatch. Constant-index
register choices avoid indirect arrays in the rolling shuffle.
`VIDEO_SELECT=projected` adapts OpenCV's cheaper pseudo-right-map semantics
with parallel left selection, gather projection, consistency, then a
median-only pass. Its BSD-3-Clause attribution is retained in OPENCV-LICENSE.
`VIDEO_SELECT_LANES=4|8|16|32` controls that selector (default 32).
The initial gather projection experiment was slower on C100; see the
[results and runtime findings](../../../docs/reference/stereo-projection-20261022/README.md).
It does not apply upstream's zero-grayscale mask or fixed-point rounding.
`VIDEO_PROJECTION=shared` replaces the projected-consistency gather with
OpenCV's winner scatter, using a row-local shared atomic minimum. It keeps
the same cost/tie rule and output while reducing projection to one scatter
per accepted winner and two shared lookups per pixel. This requires width
at most 4096; the default `gather` remains available for comparison.
The [shared-projection measurements](../../../docs/reference/stereo-shared-20261022/README.md)
reach 61–69 FPS on C100, with paired gains of 9–20% under normal devfreq.
This still does not beat optimized CPU OpenCV.
The subpixel consistency check accepts either adjacent integer correspondence,
as in OpenCV StereoSGBM, rather than truncating disparity before the check.

The complete frame pipeline runs on GPU. CPU code loads inputs, submits work
through IREE and records timing. Stage intermediates remain on GPU; final
disparity is copied back. All temporary buffers have bounded explicit lifetimes.
The implementation is generic; no board controls occur here.
`VIDEO_EXECUTION=graph` exports and measures one composed IREE invocation
while retaining individual stages for intermediate oracle checks. The
[composed results](../../../docs/reference/stereo-graph-20261022/README.md)
reach 49–53 FPS in bounded C100 diagnostics; CPU remains faster.

`throughput.exs` measures 50 prepared pairs with `VIDEO_CONCURRENCY=1|2|4`
independent existing IREE contexts and Elixir `Task.async_stream`. It reports
whole-sequence throughput and per-frame processing latency separately.
`VIDEO_REPEATS=1..20` repeats that complete clip in the same measured loop
(default 1). All copies of each frame must return identical output; agreement
checks and writing one result per distinct input occur after timing. Summary
latencies cover every invocation, while per-frame lines show the first copy.
The C100 child also enforces its configured available-memory floor.
That wall boundary also includes task scheduling and thermal checks, unlike
the older sum of individual frame-processing times; do not compare those
figures as if their accounting were identical. The heartbeat line measures
regular BEAM scheduling during the run. These experiments did not provide
a worthwhile CPU speed advantage; see the
[device profiling and saturation report](../../../docs/reference/stereo-saturation-20261022/README.md).

Export with the same GLSLANG, SPIRV_VAL, NX_IREE_COMPILER_DIR setup as
`../video`, and `STEREO_OUT=... mix run stereo/census/export.exs` from `bench/`.
Run with `STEREO_BUNDLE`, `VIDEO_DATA`, `VIDEO_OUTPUT`, `NXIREE_DEVICE_URI`.
`VIDEO_PATHS=3|4|8`, `VIDEO_DISPARITIES=64|128`, `VIDEO_P1`, `VIDEO_P2` and
`VIDEO_UNIQUENESS` specialize exports. Defaults are 8, 64, 10, 120, 1.1.
Three paths retain left, right and top, as in the direction set used by
OpenCV 3WAY; four add bottom, eight also add the diagonals. This does not
reproduce OpenCV's cost function, striped processing or output.
`VIDEO_DP_BLOCK=4|8|16` controls disparity candidates per thread (default 8).
`VIDEO_PATH_SCHEDULE=generic|specialized|cached|tiled` selects scheduling.
The default is `cached`, using libSGM's horizontal register window. `tiled`
also reuses its transposed shared right-feature window for other directions.
`VIDEO_CENSUS=symmetric|classic` defaults to the symmetric 31-bit transform.
Classic stores the upstream 48 comparisons in two u32 words, without GPU
64-bit arithmetic. P2≤224 (symmetric) or P2≤207 (classic) guarantees every
normalized path fits in a byte. Scores use regular u32 storage, without
8/16-bit GPU extensions.
The shaders require shuffle support with subgroup sizes of at least 32.

`VIDEO_FUSION=vertical` fuses the vertical libSGM recurrence with projected
selection. It requires three tiled paths, the projected selector and
`VIDEO_SELECT_LANES=VIDEO_DISPARITIES/VIDEO_DP_BLOCK`. The horizontal paths
remain a separate dispatch; the vertical-only recurrence stays in registers
while selection adds the two horizontal costs. This removes one full byte
cost volume write/read without changing output. Classic features and packed
costs use explicit wide accesses with the same raw tensor layout.
The default fusion remains `none`.

`VIDEO_FUSION=cost` is the fastest configuration on the C100. It keeps the
same classic Census, three paths, projected selection, consistency and median
output, but restructures the graph into six dispatches: `census_pair.comp`
emits both images' features from one packed comparison per neighbour,
`cost_first.comp` computes the matching costs inside the forward horizontal
scan and stores them as a byte volume, `cost_scan.comp` runs the reverse
scan over that volume and, with `FUSE_VERTICAL`, the vertical recurrence with
`cost_select.glsl`. Features and byte volumes use a row-tiled layout
(`volume_tile` rows per column, equal to the scan lines per workgroup), so
each horizontal workgroup reads and writes contiguous spans; the oracle
runner un-tiles intermediates before comparing them. It requires the same
three tiled projected paths as `vertical` plus P2 + cost bound <= 127, since
both horizontal path bytes are summed in packed form. On the C100 it reaches
49.6 FPS VGA / 17.3 FPS 720p for the resident graph and 47.1 FPS for the
complete four-context Elixir loop, above the NVIDIA L4T driver's 45.8 / 15.6
FPS for the same graph; see
[the parity report](../../../docs/reference/stereo-nvidia-20261022/optimization-round.md).

`VIDEO_MAPPABLE=true` explicitly requests IREE's existing mappable external
buffer policy. With the source-invalidation runtime fix it removes the final
GPU staging copy on the C100. The default is false; measure this policy on
each device, particularly discrete GPUs. `STEREO_PROFILE=path.ireeprof` on
`throughput.exs` captures existing IREE events around the measured loop;
profiled timings are separate from reported FPS.

The [fusion and mapped-readback follow-up](../../../docs/reference/stereo-fused-20261022/README.md)
reaches 121–140 whole-loop FPS with a bounded CPU-performance diagnostic,
versus 85–89 for four-thread CPU 3WAY under the same policy. Normal CPU
schedutil gives about 101–104 GPU FPS. Resolution is 320×136/D64; all 100
downloaded predictions are unchanged. These are short diagnostic runs with
normal GPU devfreq, not a production/stability acceptance or video-decoding
measurement. C100 power controls remain in its dedicated library.

The [concurrent CPU-work test](../../../docs/reference/stereo-concurrency-20261022/README.md)
uses 1,000 pairs per run with normal CPU/GPU governors. Two supervised
OpenSSL-backed CPU workers run in the main Elixir VM while the isolated
GPU matcher sustains about 151–152 FPS; CPU 3WAY with the same background
work reaches about 51–52 FPS. This controlled workload demonstrates CPU
offload capacity, not live-camera or application-specific acceptance.

`serving.exs` replays prepared pairs through `NxIREE.Serving` and the existing
`Nx.Serving` input/output streams. Set `VIDEO_FPS=100 VIDEO_REPEATS=20` with
the same `STEREO_BUNDLE`, `VIDEO_DATA`, and `VIDEO_OUTPUT` paths. It uses one
loaded context and an absolute arrival schedule, records arrival lateness
and completion latency in `timing.csv`, checks every repeated output, and
writes one disparity map per distinct input. `VIDEO_REFERENCE` optionally
checks those maps against an earlier output directory. Timing includes
upload, serving, graph execution, readback, guards and repeat comparison;
input preparation, warmup and disk output are excluded. A success marker
establishes correctness; inspect lateness and elapsed time to assess pacing.

Use the existing `../video/evaluate.py` for Sintel accuracy and the existing
`c100_gpu` guarded child harness (`CENSUS_STEREO_PASS` success marker) on C100.
The [follow-up measurements](../../../docs/reference/stereo-census-reuse-20261022/README.md)
reach 28–29 FPS on C100 across two sequences with the eight-direction
candidate below. It remains slower than optimized CPU OpenCV. Defaults
preserve the initial symmetric experiment; select the measured configuration
explicitly when exporting:

```sh
VIDEO_CENSUS=classic VIDEO_PATH_SCHEDULE=tiled VIDEO_SELECT=fused \
VIDEO_P1=8 VIDEO_P2=64 VIDEO_UNIQUENESS=1.0526315789473684 \
STEREO_OUT=/tmp/census.bin mix run --no-compile stereo/census/export.exs
```

Generate independent fixtures from the repository root with
`PYTHONPATH=bench/stereo/video uv run bench/stereo/census/fixtures.py OUTPUT`.
Defaults use 73×17 / D=64 / eight paths; `--disparities 128` uses width 137,
and `--paths 3|4` selects fewer directions. `--classic --matcher 8 64
1.0526315789473684` matches the measured candidate. Export matching
dimensions/options; these fixture options do not modify exporter defaults.
The runner automatically checks all intermediate oracle files before warmup
and excludes those checks from the performance timing.

The [native VGA follow-up](../../../docs/reference/stereo-vga30-20261022/README.md)
reaches 31.3 prepared-pair FPS at 640×480 per eye/D64, and sustains a paced
30 Hz rawvideo network replay using four standard Nx.Serving contexts with
normal board governors. All repeated disparity maps remain unchanged.
Four contexts trade about 124 ms median processing latency for throughput;
two contexts only sustain this replay rate in the measured loaded-CPU case.
No camera capture, rectification or compressed decoding is included.

`network.exs` accepts side-by-side gray frames over TCP. Set VIDEO_FRAMES,
VIDEO_FPS, VIDEO_BIND, VIDEO_PORT, VIDEO_OUTPUT, VIDEO_EXPECTED and
STEREO_BUNDLE; VIDEO_CONCURRENCY=1..4 selects independent standard serving
processes (default1). VIDEO_EXPECTED has 50 lines of `frame input_sha256
output_sha256`, checked cyclically; `-` skips an output hash. The runner
checks every input, reads back every result, and writes one disparity map
per unique input plus `timing.csv`. VIDEO_MATCH=false checks transport
without loading GPU models. Set EVISION_RUNTIME_ROOT and STEREO_RUNTIME_ROOT
when using isolated staged applications on a runtime-only target.

`NETWORK_STEREO_PASS` means correctness completed. Inspect wall FPS,
completion cadence, arrival lateness and scheduled completion delay to
judge pacing; processing latency alone hides TCP backpressure. The source
report records a two-context idle-CPU case that passes correctness while
falling behind. Keep device-specific guards and policy in `c100_gpu`.
