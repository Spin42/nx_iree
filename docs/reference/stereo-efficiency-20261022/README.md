# Stereo optimization and efficiency measurements, 2026-10-22

The C100 now processes the same 50 Sintel `alley_1` stereo pairs at about
**10.3 FPS**, using ordinary devfreq, with **bit-identical disparity maps and
rejection masks** compared with the first video baseline. This is about 30%
faster than the previous 7.9 FPS version and 73% faster than the original
5.93 FPS. The portable host implementation improves from 172 to 217 FPS.

This is 320×136, D=64, grayscale 3×3 SAD, eight directions **per camera**,
P1=72/P2=288, 10% uniqueness, subpixel interpolation and left–right checking.
Resolution, disparity range, matching parameters and direction count are
unchanged. This remains the SGM block-SAD pipeline described in the
[video baseline](../stereo-video-20261022/README.md), rather than exact OpenCV
SGBM. Timing includes uploads, completed GPU calls, readback and buffer
release; it excludes decode, file I/O, validation, warmup and startup.

## Measured throughput

Every video run processes all 50 pairs after three warmups. FPS is
50 / summed processing time, with nearest-rank percentiles.

| C100 variant, normal devfreq | FPS | p50 ms | p95 ms |
|---|---:|---:|---:|
| Original serial selector, s32 volumes | 5.928 | 167.449 | 176.699 |
| Coalesced selector, four scan lines/group, s32 | 7.876 | 126.655 | 129.565 |
| Packed volumes, four scan lines/group | 9.562 | 103.894 | 108.597 |
| Also four pixels/selection group | 9.873 | 101.226 | 109.791 |
| Packed volumes, selection fused with last path | 10.131 | 97.943 | 105.837 |
| Fused repeat 1 | 10.452 | 94.240 | 101.683 |
| Fused repeat 2 | 10.153 | 97.485 | 105.816 |
| Fresh final-source export | 10.391 | 94.837 | 101.344 |

The previous s32 implementation repeated at 8.052 FPS immediately before
the two fused repeats. The observed gain is 26–30% in that comparison.
Sources: [packed run](packed-alley-board.log),
[grouped selector](packed-pixels4-board.log),
[fused run](fused-alley-board.log),
[normal-policy repeats](normal-policy-repeats-board.log).
[Final-source rerun](optimized-final-board.log) also passes the four D=33
fixtures, including maximum SAD, before processing the sequence.
These are serial matcher rates, not live capture/decode/display FPS.

On the host AMD/RADV GPU, one scan line/group is preferable. The final
packed/fused version measures **216.942 FPS**, p50 4.526 ms, p95 5.249 ms,
versus 172.446 FPS for the previous coalesced/s32 version. See
[host run](fused-group1-320-host.log) and
[previous host run](group1-final-alley-host.log).
Four scan lines/group regress host throughput: the first s32 measurement
fell to 120.696 FPS, and packed/unfused measured 141.043 FPS. Grouping is
an explicit portable tuning parameter, not a universal default.

## What changed

The coalesced selector assigns 32 lanes to a pixel. Grouping independent
paths improves the tested C100's scheduling. The new packed volume assigns
two adjacent disparities to each 32-bit word, halving buffer storage and
the dominant volume traffic without requiring 16-bit Vulkan extensions.
The cost shader also shares the reference-image loads between each pair.

The valid 3×3 SAD bound is 9×255=2295. A normalized directional score is
at most 2295+288=2583; eight summed scores are at most 20664. Invalid
costs retain the original 1,000,000 semantics through a 65535 marker.
Invalid accumulated scores store bit 15 plus their exact excess over
`path_count × 1000000`. Thus even the invalid intermediate sums decode
exactly; there is no quantization. Padded halfwords at odd disparity counts
are excluded from reductions. Input grayscale range 0…255 is checked.

The last path can select directly from its register-held sums. This avoids
writing and rereading the last accumulated volume and removes two runtime
calls per stereo pair. Its shared-memory fallback remains available for
incompatible subgroup layouts. An extra diagnostic `path7_sum` export
checks the complete eight-direction sum outside timing.

Explicitly prefetching the next path pixel passed the oracle but measured
only 9.753 FPS on C100. That experiment was removed; see
[negative result](prefetch-board.log). No clock, voltage, firmware or
driver change was used for these gains.

## Correctness

[Agreement checks](packed-fused-agreement.log) compare every output pixel
in all 50 pairs with the original outputs. Packed and fused host/C100
outputs are **bit-identical**, including rejection masks. The original
ground-truth quality measurements therefore remain applicable.

The independent CPU oracle checks both full cost volumes, both complete
eight-direction sums and final disparity, plus unchanged caller inputs.
Fixtures include known shifts, unrelated random images, ambiguous constants
and maximum-SAD inputs. Both GPUs passed D=3,65,128; D=33 additionally
checks odd padding and partial workgroups. D=65/128 exercise two chunks,
including valid disparities on both sides of the chunk boundary.

Evidence: [C100 disparity-range checks](fused-range-board.log),
[D=3 host](fused-d3-host.log), [D=65 host](fused-d65-host.log),
[D=128 host](fused-d128-host.log),
[grouped/fallback fixture](fused-fixture-host.log),
[one-group host fixture](fused-group1-fixture-host.log).

## How far from the hardware limit?

NVIDIA specifies **25.6 GB/s** for the Nano's 64-bit LPDDR4 interface:
[official specifications](https://developer.nvidia.com/embedded/jetson-nano).
An estimated sustained 20 GB/s is a useful modeling point; it has not been
established as this Vulkan pipeline's measured DRAM throughput.

The portable resident-buffer XOR benchmark measured **11.371 GB/s** on
C100 for a 16 MiB input plus 16 MiB output, median 2.951 ms. That includes
output allocation and a completed runtime call, excludes upload/readback,
and verifies output and input contents. It measures this particular
buffer-transform implementation, **not a universal DRAM ceiling**. Smaller
4 MiB buffers measured 7.517 GB/s, showing dispatch/size effects.
See [benchmark documentation](../../../bench/vulkan/README.md),
[C100 result](stream-board-fixed.log), [host result](stream-host.log).

A simple cost-volume traffic model illustrates the remaining headroom:

- There are 43,520 pixels × 64 disparities = 2,785,280 volume elements.
- One s32 volume is B=11,141,120 bytes; a packed volume is V=B/2.
- Original: two cost writes + 46B for path passes + two selection reads
  total approximately **50B = 557,056,000 bytes/pair**.
- Packed/fused: two cost writes + 44V for paths (last pass reads costs
  and accumulated scores without writing another volume), approximately
  **46V = 256,245,760 bytes/pair**.

| Bandwidth assumption | Original volume-only model | Packed/fused volume-only model |
|---|---:|---:|
| 11.371 GB/s buffer-transform result | 20.4 FPS | 44.4 FPS |
| 20 GB/s assumed sustained bandwidth | 35.9 FPS | 78.0 FPS |
| 25.6 GB/s specified interface peak | 46.0 FPS | 99.9 FPS |

These are **optimistic layout-specific streaming models**, not predicted
full-pipeline performance or measured utilization. They omit image loads,
cache behavior, memory transaction efficiency, arithmetic, subgroup
reductions, sequential path dependencies, CPU/runtime dispatch and small
buffers. Cache reuse can reduce physical DRAM traffic. Conversely, access
patterns and scheduling can prevent sustained streaming rates.

At 20 GB/s the packed/fused volume traffic alone would take about 12.8 ms.
The actual frame takes about 98.7 ms. So there is substantial distance from
the streaming reference, but RAM bandwidth alone cannot identify the next
speedup. In particular, FP16 TFLOPS cannot predict this integer recurrence.

## Remaining work

The first fused C100 run averages 76.54 ms for aggregation plus selection,
14.97 ms for cost generation, 3.59 ms upload, 1.66 ms consistency and
1.77 ms readback. See [stage means](packed-fused-stage-means.txt).
In fused runs the phase named `select` includes the eighth directional
pass; combine it with `paths` for comparisons.

30 FPS at this resolution still needs approximately **3×** the present
throughput. The next useful measurements are per-direction completed GPU
times and generated shader instruction/register behavior. Likely work
areas are latency hiding in the sequential recurrence, subgroup/workgroup
occupancy, cost reuse, and submitting the graph through fewer synchronous
runtime calls. None of the models proves a particular final FPS.

## Reproduction and state

From `bench/`, after setting `GLSLANG`, `SPIRV_VAL` and the source compiler:

```sh
VIDEO_STORAGE=packed16 VIDEO_FUSE_SELECT=1 VIDEO_LINES_PER_GROUP=4 \
  STEREO_OUT=/tmp/video.bin mix run --no-compile stereo/video/export.exs
```

Use `VIDEO_LINES_PER_GROUP=1` on the tested host GPU. Then run the same
runner/data and C100 child harness described in
[the benchmark README](../../../bench/stereo/video/README.md).
The generic exporter/API has no C100 clock or device-specific behavior;
board controls remain in `c100_gpu`.

The unsigned stream test also exposed cold exported-model loading rejecting
the type atom in `:safe` term decoding. Loading `Nx.Type` before decoding
fixes the shared loader. A fresh-child regression and Vulkan tests passed
([9 passed](loader-tests-final.log)). Only the loader BEAM in the isolated
`/root/nxiree-stereo` runtime was updated; its previous BEAM was saved.
The first failure is retained in [stream-board.log](stream-board.log).

Python strict checks pass with the pinned Python 3.13 dependencies;
GLSL modules pass SPIR-V validation during export. Kernels own individual
GPU stages, the oracle remains independent, and board policy stays outside
the portable benchmark. The runner is in the 200–250 line warning band;
future additions should extract its oracle/resource checks before growing it.

[Final board state](optimized-final-board-state.log) records normal devfreq,
unchanged boot ID, zero GPU errors and bit-for-bit agreement for the fresh
final-source export. [Source/bundle hashes](source-and-bundle-sha256.txt)
identify the saved artifacts. Changes remain uncommitted.
