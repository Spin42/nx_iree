# Stereo sequence benchmark

This is a runnable, portable image-pair-to-disparity baseline built with
`NxIREE.Vulkan.export/4`. The same exported modules run on the host RADV GPU
and C100 NVK GPU. Board controls belong to the separate `c100_gpu` library.

The matcher computes grayscale 3×3 block SAD, aggregates eight directions
for each camera, selects the minimum with a 10% uniqueness margin, refines
it with a parabola and applies a one-pixel left–right consistency check.
It uses P1=72, P2=288, disparities 0…63, replicated image borders and -1 for
rejected output. Intermediate costs remain on the GPU and buffers are freed
between passes. Each runtime invocation waits for completion.

This is an SGM block-matching baseline, not bit-identical OpenCV SGBM.
It lacks OpenCV's derivative/Birchfield–Tomasi cost, median postprocessing
and speckle filtering. It also has no temporal smoothing, camera calibration,
rectification or live-video decoding. Inputs must already be rectified.
Subgroup arithmetic/shuffle support is required; the aggregation shader has
a shared-memory fallback for incompatible subgroup sizes/lane layouts.

The default selector distributes each pixel's disparity candidates across
32 threads for contiguous reads. `VIDEO_SELECT=serial` reproduces the first
benchmark; `coalesced_shared` exercises the selector's shared-memory fallback.
`VIDEO_LINES_PER_GROUP` accepts 1 (default), 2 or 4 independent scan lines.
Four lines improve C100 throughput but regress the tested AMD configuration,
which uses the aggregation fallback for that grouping. This is a portable
tuning parameter, not a change in the number of directions or matching quality.
See the [optimization measurements and bandwidth model](../../../docs/reference/stereo-efficiency-20261022/README.md).

`VIDEO_STORAGE=packed16 VIDEO_FUSE_SELECT=1` enables the faster packed
pipeline. It stores two 16-bit scores in each u32 word and selects disparity
inside the last directional pass. It uses ordinary 32-bit Vulkan storage,
without requiring 16-bit arithmetic or storage extensions. These options
preserve all eight directions, both camera views and the matching parameters.
Use `VIDEO_LINES_PER_GROUP=4` on the tested C100; the tested AMD GPU performs
better with the default 1. `VIDEO_PIXELS_PER_GROUP=1|2|4` separately tunes the
unfused selector; the fused pipeline does not run that selector.

Packing requires grayscale input values 0…255, checked by the runner. Valid
3×3 SAD is at most 2295; one normalized path is at most 2583 and eight sum
to at most 20664. Invalid costs use 65535 in storage. Invalid sums use bit
15 plus their exact excess over `path_count * 1000000`; decoding reproduces
the original full s32 sums. Odd disparity counts have one padded halfword,
excluded from all reductions. No cost precision is discarded.

## Download and prepare

[MPI Sintel stereo](https://sintel.is.tuebingen.mpg.de/stereo) provides
consecutive stereo pairs, disparity ground truth and visibility masks.
Download its [20150305 training ZIP](https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip)
outside the repository:

```sh
curl -fL -C - -o /tmp/sintel.zip \
  https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip
uv run stereo/video/prepare.py /tmp/sintel.zip /tmp/alley --sequence alley_1
```

Commands here run from `bench/`. Preparation uses all 50 consecutive clean
frames, resized from 1024×436 to 320×136. Images use Lanczos; decoded
ground-truth disparity and masks use nearest-neighbor sampling. Horizontal
disparity is multiplied by 320/1024. The SDK's positive disparity magnitude
maps `x_right = x_left - disparity`, despite the dataset prose saying
“negative disparities.” The RGB decoding includes all three SDK channels.
This resized test is not an official Sintel leaderboard evaluation.

Python scripts declare pinned dependencies and Python 3.13 through `uv`.
They produce s32 grayscale input buffers and f32 disparity ground truth.

## Export and run

Set `GLSLANG` and `SPIRV_VAL` to installed executable paths. Select the
patched source IREE compiler described in `docs/stereo-path.md` through
`NX_IREE_COMPILER_DIR` when compiling the Mix project.

```sh
STEREO_OUT=/tmp/video.bin mix run stereo/video/export.exs
NXIREE_DEVICE_URI='vulkan://GPU-…' STEREO_BUNDLE=/tmp/video.bin \
  VIDEO_DATA=/tmp/alley VIDEO_OUTPUT=/tmp/alley-result \
  mix run --no-compile stereo/video/run.exs
uv run stereo/video/evaluate.py /tmp/alley /tmp/alley-result /tmp/alley-report
ffmpeg -framerate 10 -i /tmp/alley-report/preview_frames/%04d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart /tmp/stereo-preview.mp4
```

For the optimized export, prefix the export command with
`VIDEO_STORAGE=packed16 VIDEO_FUSE_SELECT=1`. Add
`VIDEO_LINES_PER_GROUP=4` for the C100 measurement. Defaults retain the s32
pipeline so earlier comparisons remain reproducible.

The preview deliberately plays at 10 FPS for inspection; that is not the
measured inference rate. It shows left/right images, ground truth, Vulkan
disparity, absolute error and an OpenCV 4.12 MODE_HH quality baseline.
Color scales stay fixed: disparity 0–32 pixels (larger values saturate),
error 0–3 pixels, rejected/masked pixels black.

`VIDEO_WIDTH`, `VIDEO_HEIGHT` and `VIDEO_DISPARITIES` specialize exports;
their defaults match preparation. The exporter supports 3–128 disparities
subject to the generic API's dispatch limits. The preparation/evaluation
scripts currently define this particular 320×136, 64-disparity experiment.

The runner performs three warmup pairs, then processes every frame once.
FPS is frame count divided by summed processing latency. Timing includes
input upload, cost generation, 16 directional passes, both selections,
consistency checking, readback and buffer release. It excludes startup,
warmup, image decoding/resizing, disk input/output, quality evaluation and
resource guards. Percentiles use nearest rank. Thus it measures serial
matcher throughput, not a live camera-to-display system or pipelined rate.

Quality is pixel-weighted over non-occluded, in-frame ground truth. Reports
include accepted-pixel error, coverage, and bad-or-rejected rates, so rejecting
pixels cannot silently improve the result. They also report out-of-search
ground truth and a common x≥64 region for the OpenCV comparison, since
OpenCV rejects the left search-width border by design. The algorithms differ;
their scores are a quality reference, not an equivalence assertion.

## Correctness and board execution

```sh
uv run stereo/video/fixtures.py /tmp/video-fixtures
VIDEO_WIDTH=37 VIDEO_HEIGHT=13 VIDEO_DISPARITIES=33 \
  STEREO_OUT=/tmp/video-fixtures.bin mix run --no-compile stereo/video/export.exs
NXIREE_DEVICE_URI='vulkan://GPU-…' STEREO_BUNDLE=/tmp/video-fixtures.bin \
  VIDEO_DATA=/tmp/video-fixtures VIDEO_OUTPUT=/tmp/fixture-result \
  mix run --no-compile stereo/video/run.exs
```

The independent scalar oracle checks full left/right cost volumes,
eight-direction sums and final disparity, including a known four-pixel
shift, unrelated images, an ambiguous constant pair and maximum-SAD inputs.
It uses 33
disparities to exercise the cross-lane/chunk boundary. Caller inputs must
remain unchanged. A mismatch stops the run immediately.
`fixtures.py --disparities D` supports 3…128; width is 37 through D=33,
otherwise D+5, with height 13. Match the export dimensions to the fixtures.
The packed/fused pipeline was also checked at D=3,65,128 on both GPUs.
In fused exports, an additional `path7_sum` module verifies the complete
eight-direction sum outside the timed pipeline. The reported `select` phase
then includes the final directional pass, so compare combined path/selection
time across fused and unfused versions.

For runtime-only targets, copy the bundle, runner and raw input files;
set `STEREO_RUNTIME_ROOT`, `STEREO_BUNDLE`, `VIDEO_DATA`, `VIDEO_OUTPUT`.
The C100 harness additionally uses `STEREO_SCRIPT` to select `video/run.exs`.
It runs in an isolated child BEAM with a 120-second deadline and checks
GPU errors. The runner guards temperature at 70°C and supports
`STEREO_MIN_AVAILABLE_KB`. All clock/power control remains in `c100_gpu`.
Its controlled comparison accepts `STEREO_COMPARE_SCRIPT` and restores
the original minimum frequency even on failure.

`ruff check stereo/video` and `basedpyright --project stereo/video` check
Python helpers with their declared dependencies installed. The type-check
configuration uses `all`, except NumPy's scalar-indexing/reduction `Any`
annotations and script-local imports; no application `Any` signatures or
type-ignore escapes are used. Each helper owns dataset preparation, the
scalar oracle, fixture generation, evaluation, or preview rendering.

See [measured results and evidence](../../../docs/reference/stereo-video-20261022/README.md).

Dataset attribution: Butler, Wulff, Stanley and Black, *A naturalistic open
source movie for optical flow evaluation*, ECCV 2012. Dataset copyright
2015 Jonas Wulff and Michael Black, Max Planck Institute for Intelligent Systems.
