# Census SGM GPU port, 2026-10-22

Historical result: the [subsequent reuse and validation work](../stereo-census-reuse-20261022/README.md)
reaches 28–29 FPS with improved accuracy on two sequences. The measurements
below preserve the initial symmetric-Census candidate.

The initial Vulkan candidate processes all 50 Sintel `alley_1` pairs at
**19.11 FPS on C100**, with normal devfreq, at 320×136 / D=64. Its complete
matching pipeline runs on the GPU through the existing NxIREE/IREE runtime.
**It does not meet the goal of beating optimized CPU OpenCV at comparable
quality.** The Census matching cost currently produces substantially more
bad/rejected pixels. This remains an experiment, not a production replacement.

## Reuse and implementation

The [benchmark source](../../../bench/stereo/census/README.md) adapts
Fixstars libSGM revision `e4c669bb9e039b60f04178b080ac7d6077a4d4bd`:

- Symmetric 9×7 Census, including the upstream eight-row shared-memory ring.
- Normalized SGM recurrence with multiple disparity candidates per thread.
- Byte-sized directional state; four scores packed in ordinary u32 storage.
- Both disparity maps derived from one aggregated volume, including the
  upstream rolling right-map minimum rather than a second full aggregation.
- The upstream median selection network and a GPU consistency check.

Upstream libSGM is CUDA-only. This is a GLSL/Vulkan port of those components,
not direct linking or a claim of upstream Vulkan support. Apache-2.0 notices
and the full license are retained. IREE supplies buffer allocation, executable
loading, dispatch, submission and synchronization; no new native executor,
allocator, NIF, board policy or firmware was introduced.

Differences are explicit in the source README: replicated Census image
borders, exclusion of negative-coordinate candidates, float subpixel output,
no zero-grayscale mask, and an OpenCV-style floor/ceil consistency check.
This is not an exact libSGM or OpenCV output reproduction. The source remains
generic and has run on host RADV and C100 NVK.

## Measured comparison

All rows use the same 50 prepared rectified pairs, image size, search range
and ground truth. Directions, matching costs and right-view construction
differ, so these are complete-pipeline comparisons rather than identical work.
The common region is x≥64 over non-occluded/in-frame ground truth; rejected
pixels count as errors. No capture, decoding, resizing or disk I/O is timed.

| Pipeline | C100 FPS | Common bad1 or rejected | Common bad3 or rejected |
|---|---:|---:|---:|
| OpenCV 3WAY, three directions | 100.78 | 4.434% | 2.744% |
| OpenCV HH4, four directions | 63.65 | 4.389% | 2.751% |
| OpenCV HH, eight directions | 15.64 | 4.085% | 2.511% |
| Previous Vulkan SAD, eight directions per view | 10.39 | 4.269% | 3.171% |
| Census Vulkan candidate, eight shared directions | 19.11 | 7.129% | 4.554% |

CPU measurements and configuration proofs are in the
[library survey](../stereo-library-survey-20261022/README.md). The GPU candidate
uses P1=10, P2=120 and a 10% uniqueness threshold. Its common coverage is
97.129%, accepted-pixel MAE 0.5596 pixels. Thus its throughput above CPU HH
does not establish the requested win; it is both slower and less accurate
than the fast CPU modes on this sequence.

The C100 candidate's median is **51.504 ms**, p95 **56.223 ms**. Upload,
Census, all directional paths, summation, both selections, medians,
consistency, final readback and buffer release are included. There are three
warmup pairs, and executable loading/startup is excluded. Every dispatch
waits for completion. Host RADV measured **512.29 FPS** for the same bundle;
that is a host measurement, not a C100 forecast.

Sources: [C100 run](board.log), [host run](host.log), [board quality](quality.json).
All 50 board predictions were downloaded and evaluated. They have identical
validity masks to host predictions and maximum absolute difference
`3.814697265625e-6` pixels: [agreement](host-board-agreement.log).

## What the experiments changed

The initial straightforward port measured 16.56 FPS (17.02 on repeat) but
rejected incomplete Census borders, giving 13.21% common bad1-or-rejected.
Replicated borders removed that avoidable penalty. Exploratory
[penalty](penalty-sweep.json) and [uniqueness](uniqueness-sweep.json) sweeps
still failed to match CPU accuracy. These use the same sequence; they are
not a held-out quality validation or general parameter recommendation.

Increasing the disparity candidates per thread from 8 to 16 made the
revised C100 pipeline slower, at 15.17 FPS. The final source defaults to 8.
The shared-memory Census and rolling selector then reached the reported
19.11 FPS. The initial shader [compiler summary](nak-initial-summary.log)
reported no spills and no scratch storage; it came from the host GM20B
drm-shim compiler and predates the tiled/rolling changes. It is not a hardware
GPU timestamp trace.

Read-only [clock samples](normal-devfreq-samples.log) during the initial
repeated run included 71 samples at 921.6 MHz, one at 614.4 MHz and 88 at
76.8 MHz, covering startup, processing and idle. The normal governor already
reaches the accepted cap; no frequency or voltage policy was changed.

[Final stage means](phase-means.log), over exactly the 50 video pairs:

| Stage | Mean ms |
|---|---:|
| Directional paths | 29.585 |
| Selection of both maps | 7.681 |
| Sum of directional costs | 5.071 |
| Census | 2.253 |
| Median and consistency | 1.955 |
| Upload | 3.681 |
| Readback | 1.864 |

These are synchronous invocation timings including allocation/runtime work,
not pure GPU timestamps. Paths remain the largest cost. Further work must
retain comparable accuracy, improve the dominant GPU stages and compare with
the fast CPU modes. Passing the slower HH timing alone is insufficient.

## Correctness and regression evidence

Independent scalar fixtures validate every Census feature, each directional
cost byte, the aggregate, both disparity maps, final output and preservation
of caller inputs. The median oracle uses existing OpenCV code. Four cases
cover a known four-pixel shift, unrelated textures, ambiguity and opposing
constant images, with partial workgroups/tiles and incomplete right-map tails.

- D=64 / eight paths: [host](fixtures-host.log) and the first child in
  [board.log](board.log).
- D=128 / eight paths: [host](d128-fixtures-host.log) and
  [board](extra-fixtures-board.log).
- D=64 / four paths: [host](four-fixtures-host.log) and the same board log.
- The shared scalar recurrence gained optional penalties/directions; all
  28 original SAD fixture files remain byte-identical:
  [regression](sad-oracle-regression.log).

The first oracle exposed a CPU-reference bug, not a GPU recurrence error:
at index 301 (x=9), rounding an actual 8.5 subpixel result falsely identified
it as an integer winner at x=9, overwriting it with 9. The boundary rule now
uses the integer argmin directly. [Red](oracle-red.log) and
[green](oracle-green.log) runs are retained; all integer stages already matched.

SPIR-V validation, Python lint/strict typing and Elixir formatting pass.
All guarded board children exited successfully with GPU_ERRORS=0. The
accepted firmware, boot ID, validated slot and normal idle policy are retained:
[final state](final-board-state.log). Source/bundle [hashes](sha256.txt) identify
the measured candidate. No firmware was flashed or reboot requested.
