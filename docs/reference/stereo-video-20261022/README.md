# First complete stereo sequence measurement — 2026-10-22

The same 50 outputs now run at about 10.3 FPS on C100, with exact output
agreement and normal devfreq. See the
[optimization report](../stereo-efficiency-20261022/README.md). The baseline
measurements below remain the original implementation's results.

Tested all 50 consecutive frames of MPI Sintel `alley_1`, clean pass, at
320×136 and 64 disparities. This measures an actual stereo pair producing
a checked disparity image, using the same exported Vulkan modules on RADV
and NVK. It is an eight-direction block-SAD SGM baseline, with subpixel
selection, uniqueness and left–right consistency. See the
[reproduction instructions and exact algorithm](../../../bench/stereo/video/README.md).

| Device / policy | Processing FPS | Median pair latency | p95 | p99 |
|---|---:|---:|---:|---:|
| Host AMD Navi32 / RADV | 155.20 | 6.429 ms | 6.668 ms | 6.804 ms |
| C100 / normal simple_ondemand | 5.928 | 167.449 ms | 176.699 ms | 200.082 ms |
| C100 / accepted 921.6 MHz cap held | 5.900 | 169.374 ms | 174.941 ms | 176.371 ms |

FPS includes Elixir input upload, both cost volumes, sixteen path passes,
both disparity selections, consistency, final readback and buffer release.
It excludes decoding/resizing, disk access, quality checks, guards, startup
and three warmup pairs. Percentiles use nearest rank over 50 frames.
These are short sequential processing measurements, not live-camera FPS,
long thermal soak results, or proof of NVIDIA-equivalent performance.

The fixed-clock comparison does not improve this workload. The normal
minimum was restored to 76.8 MHz, governor to `simple_ondemand`; cap remains
921.6 MHz. No firmware, voltage policy, target runtime or boot change was
made. The final board temperatures were 37–39.5°C with about 3.6 GB available
memory and no Nouveau errors. See [final state](final-board-state.log).

## Accuracy

Across 1,886,631 visible, in-frame ground-truth pixels:

- Accepted output coverage: **97.369%**.
- Mean absolute disparity error on accepted pixels: **0.29470 pixels**.
- Accepted pixels with error >1 pixel: **1.493%**.
- Error >1 pixel **or rejected output**: **4.084%** of the evaluation population.
- Error >3 pixels **or rejected output**: **3.095%**.
- Ground truth outside the 0–63 search range: **0%**; largest evaluated
  disparity is 42.637 pixels.

The C100 and host differ by at most 0.000003815 pixels; rejected-pixel masks
are identical across all 50 frames, including the fixed-clock run.
See [device agreement](device-agreement.log).

OpenCV MODE_HH is also evaluated as a quality reference. It uses a different
matching cost and postprocessing. Its leftmost 64 columns are rejected by
design; compare on the common x≥64 region rather than attributing that border
to poor matching:

| Common visible region | Coverage | Accepted MAE | >1 px or rejected | >3 px or rejected |
|---|---:|---:|---:|---:|
| Vulkan block-SAD SGM | 97.341% | 0.3066 px | 4.269% | 3.171% |
| OpenCV 4.12 SGBM MODE_HH | 98.694% | 0.4121 px | 4.085% | 2.511% |

Thus the Vulkan baseline has lower accepted-pixel error here, but OpenCV
retains more matches and has better combined error/rejection rates. This
single clean rendered scene does not establish general accuracy. Ground
truth is nearest-neighbor resized and scaled; these are not official
Sintel leaderboard scores. Occluded pixels are excluded explicitly.

## Where time goes

Mean completed-invocation wall times on the C100 with normal devfreq:

| Stage, both views combined | Mean per pair |
|---|---:|
| Eight-direction aggregation per view | 114.272 ms |
| Disparity selection / subpixel / uniqueness | 28.128 ms |
| 3×3 SAD cost construction | 19.231 ms |
| Input upload | 3.467 ms |
| Final readback | 1.863 ms |
| Consistency check | 1.550 ms |

These timings include runtime/driver work, not just GPU timestamps. The
remaining fraction is orchestration and release. Aggregation is about 68%
of pair time; selection about 17%. The next optimization should examine
selection's strided cost-volume reads, aggregation occupancy/memory access,
and reducing synchronous submissions while keeping intermediates resident.
The current eight-direction result cannot yet meet 30 FPS: 168.7 ms mean
must fall below 33.3 ms, about a **5.1× improvement at this resolution**.
Larger images/disparity ranges need separate measurements.

## Evidence and artifacts

- [Preview video](stereo-preview.mp4): 50 frames at an intentional 10 FPS
  playback rate, five seconds; this playback rate is independent of inference.
- [Preview image](stereo-preview.png): frame 25 from the C100 result.
- Logs: [host](alley-host.log), [C100 normal](alley-board.log),
  [C100 fixed cap](alley-board-max.log).
- Structured metrics: [host](host/metrics.json), [C100](board/metrics.json),
  [fixed cap](board-max/metrics.json).
- Independent scalar oracle: [host](fixture-host.log),
  [C100](fixture-board.log). All three synthetic pairs pass exact costs,
  exact eight-path sums, final disparity tolerance and unchanged inputs.
- All generated SPIR-V modules passed `spirv-val` before export.
- Python helper checks: Ruff, the programming-rule audit and BasedPyright
  pass. The documented NumPy `Any`/script-import exceptions apply. Source
  files remain below 200 nonblank/noncomment lines each.

All changes remain uncommitted. The portable algorithm lives in
`nx_iree/bench/stereo/video`. The only board helper change is allowing
`STEREO_COMPARE_SCRIPT` in `c100_gpu/scripts/compare_stereo_at_max.exs`;
its previous default is preserved and the script is validated before
changing clocks.

## Local data provenance

The source is the official [MPI Sintel stereo training archive](https://sintel.is.tuebingen.mpg.de/stereo),
`MPI-Sintel-stereo-training-20150305.zip` (2,612,604,481 bytes). To avoid
downloading unrelated sequences, HTTP byte ranges fetched the archive index,
SDK and five complete `alley_1` directories. The sparse archive at
`~/.cache/stereo-bench/sintel-index.zip` contains only these selected entries;
it is not a complete dataset ZIP. The interrupted full download is retained
separately as `sintel.zip` and must be resumed before normal full-archive use.

Prepared inputs/ground truth are in `~/.cache/stereo-bench/alley_1`;
raw outputs are in `alley-host`, `alley-board` and `alley-board-max` beside it.
Full preview PNG sequences are also in that cache. The exported bundles
are `/tmp/stereo-video-320.bin` and `/tmp/stereo-video-fixture.bin`, staged
on the board as `/root/nxiree/video-320.bin` and `video-fixture.bin`.
The current working cache paths are evidence locations, not required paths
for reproduction.

Dataset attribution: Butler, Wulff, Stanley and Black, *A naturalistic open
source movie for optical flow evaluation*, ECCV 2012. Dataset copyright
2015 Jonas Wulff and Michael Black, Max Planck Institute for Intelligent Systems.
