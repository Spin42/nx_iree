# Native VGA stereo: 31.3 FPS prepared pairs and paced 30 Hz replay

At **640×480 per eye / D64**, the retained changes reach **31.30 whole-loop
GPU FPS** on 200 prepared pairs, versus **23.90** before them: a **31.0%**
throughput improvement. A separate 600-pair network replay with two ordinary
Nx.Serving contexts and two background CPU workers reaches **30.004 wall
FPS**, checking every input and disparity hash. With otherwise idle CPUs,
**four contexts sustain 30.001 wall FPS over 1,200 pairs**; details below.

There is no pyramid, ROI, resolution reduction, disparity-range reduction
or subpixel precision reduction. There are no attached cameras. These runs
do not include camera capture, calibration, rectification or video decoding.
Historical driver errors remain; this is bounded diagnostic evidence.

## Retained portable changes

The first change reuses upstream libSGM's duplicated-column shared tile,
making vertical feature addresses invariant across rows instead of
recomputing wrapped indices. Source:
[upstream vertical tile](https://github.com/fixstars/libSGM/blob/e4c669bb9e039b60f04178b080ac7d6077a4d4bd/src/cost_aggregation.cu).

The larger gain comes from expressing candidate selection with standard
GLSL `mix`, avoiding structured per-pixel branches, followed by bounded
branch-free subpixel refinement. The original final native selector has
33 reconvergence setup sites. Positive refinement denominators are integer
cost differences exactly represented in f32, so clamping the speculative
denominator to at least one leaves valid results unchanged. The original
f32 refinement formula and neighbor-range masks remain.

The SGM recurrence, classic 48-bit Census, three paths, D64, P1=8/P2=64,
uniqueness=1.0526315789473684, projected consistency and median filtering
are unchanged. No device identifier or Maxwell-specific shader feature is
added. No extra exporter option is needed for these source improvements.

## Prepared-pair and CPU controls

| Run | Whole-loop FPS | Matcher CPU cores |
| --- | ---: | ---: |
| Original GPU, two contexts, two CPU workers, 200 pairs | 23.90 | see raw log |
| Duplicated-tile GPU, same conditions, 200 pairs | 25.08 | 0.33 |
| Plus conditional candidate selection, 200 pairs | 30.06 | see raw log |
| Plus bounded branch-free refinement, 200 pairs | **31.30** | **0.31** |
| CPU OpenCV 3WAY, four threads, two CPU workers, 50 pairs | 8.03 | 1.93 |
| CPU OpenCV 3WAY, four threads, no CPU workers, 50 pairs | 13.49 | 3.55 |

GPU is 3.90× the loaded CPU control's throughput. The algorithms use
different matching costs and parameters; this is not an identical-algorithm
speedup. CPU workers use existing OTP/OpenSSL SHA256, about two cores,
measured over the whole guard callback rather than the matcher window.

The final 200-pair GPU run lasts 6.391 seconds; p50/p95 processing latency is
61.72/65.36 ms with two in-flight contexts. Overlap throughput differs from
frame latency. Upload, matching, readback, synchronization, task scheduling
and thermal/memory guards are included. Disk I/O, input preparation and
three warmups/context are excluded. All 200 repeated outputs agree.

## Paced network replay and the idle-CPU gap

The host uses FFmpeg's standard rawvideo TCP sender at 30 stereo pairs/s:
1280×480 side-by-side gray, 640×480 per eye, 18.432 MB/s payload. The board
uses Evision for splitting/conversion, Nx.BinaryBackend inputs,
NxIREE.Serving adapters, and ordinary Nx.Serving processes under an OTP
Supervisor. Multiple contexts use standard Task.async_stream; no custom
native executor or queue is added. Every frame is checked against its
source SHA256 and previously validated GPU output. No frames are dropped.

| 600-pair replay | Wall FPS | Completion cadence | Processing p50 / p95 | CPU cores |
| --- | ---: | ---: | ---: | ---: |
| Two contexts, two CPU workers | **30.004** | 30.350 | 60.87 / 67.50 ms | 0.87 |
| Two contexts, otherwise idle CPU | 26.800 | 27.106 | 67.27 / 78.21 ms | 1.20 |
| Three contexts, otherwise idle CPU | 29.833 | 30.215 | 87.91 / 126.29 ms | 1.20 |

The longer 1,200-pair idle replay with three contexts reaches only
29.728 wall / 29.882 completion FPS, ending with 359 ms scheduled delay.
Four contexts reach **30.0008 wall / 30.1662 completion FPS** over 39.999 s,
using **1.19 CPU cores**. Processing p50/p95 is **124.21/134.48 ms**,
maximum 199.11 ms. All 1,200 frame hashes pass, with no dropped frames or
new GPU errors. Scheduled completion delay starts at 199.11 ms and ends at
-21.06 ms; its p95/max is 226.06/264.07 ms. Negative relative delay reflects
the sender's pacing/initial phase, not negative physical capture latency.

The same four-context configuration with two background CPU workers also
keeps pace: **29.9953 wall / 30.4132 completion FPS** over 600 pairs, using
0.91 matcher CPU cores. Processing p50/p95 is 120.94/152.17 ms, maximum
275.92 ms. All hashes pass and no new GPU errors occur. The few milliseconds
of startup/drain distinguish its wall rate from exactly 30; this is not a
claim that every frame meets a 33.3 ms latency deadline. The lower-latency
two-context configuration is only established with the loaded CPU control.

The two-context idle run fails pacing: scheduled completion delay grows
from 249 ms to 2.38 seconds. Its correctness success marker is not a
throughput pass. The idle prepared-pair profile shows 88.1% device-active
time and 219 ms of queue gaps, versus 93.7% and 104 ms with CPU workers.
Steady shader times are similar. Host submission gaps now matter; these
measurements do not establish a particular CPU frequency as their cause.

Three serving contexts hide most gaps in the short idle run, but the longer
follow-up falls slightly behind. **Four contexts are the measured idle-CPU
configuration**, trading roughly four-frame processing latency for sustained
throughput. No CPU performance governor or frequency floor is needed.
Read-only steady samples show the GPU at 921.6 MHz while CPU frequency
varies mostly around 408–612 MHz. Startup/drain remain in wall FPS; completion
cadence measures first-to-last completion. Processing latency starts when
the board receives a complete frame, excluding time already queued in TCP.
`timing.csv` records arrival lateness too: scheduled completion delay equals
processing latency plus arrival lateness. No synchronized host-capture
latency or camera deadline is claimed.

## Correctness and useful depth

All 50 downloaded maps from the baseline, duplicated-tile and final
200-pair runs are byte-identical to the previously validated C100 reference:
15,360,000 pixels per comparison. Each final run also checks every repeat.
The four independent CPU fixture cases pass for D64/classic three-path
fusion, D128/classic fusion, BLOCK4, BLOCK16, and D64/symmetric eight-path
scans: **20 final-configuration cases** on the host discrete Vulkan GPU.
These include arbitrary-width edge groups, reverse and diagonal paths.

On the original Sintel alley_1 source, resized directly from 1024×436 to
640×480, the common visible region x>=64 gives:

| Measure | GPU Census SGM | CPU OpenCV 3WAY |
| --- | ---: | ---: |
| Coverage, all visible ground truth | 97.56% | 96.52% |
| >3px error or invalid, all visible | 8.12% | 8.38% |
| Coverage, ground truth inside D64 range | 99.31% | 98.97% |
| Mean absolute error of accepted in-range pixels | 0.560 px | 0.478 px |
| >3px error or invalid, in-range pixels | 2.09% | 2.25% |

**7.62% of visible ground truth lies outside D64**; maximum disparity is
85.43 pixels. This does not cover arbitrary close objects or camera optics.
Resizing to 4:3 changes the source aspect ratio and slightly enlarges its
vertical dimension; this is a native pixel-workload test, not a claim about
particular camera geometry. Quality reuses existing NumPy/OpenCV helpers.
`fps: null` in GPU quality JSON means timing comes from the throughput log.
CPU JSON's `fps` is processing-only; use raw-log `wall_fps` for comparisons.

## GPU execution evidence

The baseline has 250 valid dispatches, no dropped/truncated records and
97.6% device-active time. Steady per-frame dispatch times are approximately
3.69 ms Census, 13.54 ms horizontal scans, 22.10 ms fused vertical/selection,
0.76 ms consistency and 0.96 ms median. The duplicated tile reduces vertical
time to ~20.03 ms. With all retained changes, horizontal is ~12.48 ms and
vertical/selection ~13.35 ms; the other stages remain nearly unchanged.
The final loaded and idle profiles each have 250 valid dispatches and no
dropped/truncated records. Early outliers occur; the idle profile's first
Census dispatch is 44 ms. Use steady and aggregate timings separately.

Mesa NAK compilation reports zero memory spills/SLM in all five shaders;
horizontal and fused vertical use 48 GPRs and a reported maximum 40 warps/SM.
Census uses 24 GPRs/64 warps. Both original and final captures confirm those
allocation figures. The driver exposes one compute-capable queue. Another
IREE queue is not evidence of a second compute engine. Timestamp calibration
frequency is not GPU clock.

A fresh final-shader compiler capture confirms the mechanism: relative to
the duplicated-tile control, the vertical selector falls from **731 to
522 native instructions**, and **34 to 5 reconvergence setups**. Horizontal
falls from 949 to 921 instructions and 27 to 23 setups. Both final path
shaders still use 48 GPRs, report 40 warps/SM maximum, and have zero memory
spills/SLM. The speedup therefore does not require a claimed occupancy gain.
See [compiler summary](compiler-summary.log) and
[final allocation statistics](shader-refine-stats.log).

## Rejected experiments

Short 50-pair C100 runs below use two contexts/two CPU workers unless stated.
They are diagnostics, not confidence intervals. No rejected option remains
in the exporter. All completed guards add zero new GPU errors.

| Experiment | FPS |
| --- | ---: |
| Original configuration | 23.37 |
| Separate vertical and selector | 21.69 |
| DP block 4 / 16, both directions | 17.59 / 20.94 |
| Path workgroups 64 / 32 | 21.85 / 20.05 |
| Precomputed cost, one / four costs per thread | 15.06 / 20.74 |
| Canonical min-before-normalization | 22.39 |
| Direct vertical feature reads | 21.58 |
| Split/repadded shared tile planes | 23.23 |
| Existing symmetric 31-bit Census | 23.95 |
| Uniqueness subgroup ballot | 21.54 |
| Sum horizontal byte volumes | 22.28 |
| Duplicated tile, 128 threads (retained) | 24.56 |
| Duplicated tile, vertical WG 256 / 512 | 24.71 / 22.49 |
| Separate horizontal direction exports | 24.70 |
| Relative neighbour shuffles | 24.51 |
| Upstream lazy in-place recurrence | 24.72 |
| Shared-cost subpixel lookup | 23.87 |
| glslang/SPIRV-Tools -Os | 24.63 |
| Direct fixed-bit Census placement | 22.68 |
| Integer disparity | 25.72 |
| Two independent vertical states | 22.76 |
| Balanced DP minimum tree | 24.01 |
| Horizontal loop unroll 2 | 23.61 |

Cost precomputation sped up scans but its extra stage cost ~33.8 ms.
Integer disparity worsened common/in-range MAE to 0.629 px and bad3-or-invalid
to 2.37%, versus 0.560 px/2.09% with unchanged subpixel precision. It was
reverted. The two-state experiment also lost to its one-state aligned
control (24.67 FPS). Shader changes above pass independent CPU oracles.

An isolated Mesa integer-register-bank heuristic variant passes C100
fixtures and process-map driver identity checks, but 24.39 FPS versus
24.08 FPS for its freshly rebuilt unmodified-source control is no useful
gain. Original build source is restored and rebuilt. No driver patch or
production library replacement is retained. Isolated driver binaries and
source snapshots remain only as local evidence.

Verbose NAK capture initially exceeded the guard timeout. The child stopped
and debug environment was restored. File capture then succeeded without
new GPU errors. The historical **23 nv50cal_space:-110 errors remain**;
this is not clean driver/boot stability acceptance. Continue stopping any
actual boot/load series at its first failure; never clear dmesg to pass.

## Reproduction and artifacts

Portable sources are in `bench/stereo/census/`; CPU control/evaluation in
`bench/stereo/opencv/`. Export with VIDEO_WIDTH=640, VIDEO_HEIGHT=480,
VIDEO_DISPARITIES=64, VIDEO_PATHS=3, VIDEO_P1=8, VIDEO_P2=64,
VIDEO_UNIQUENESS=1.0526315789473684, VIDEO_SELECT=projected,
VIDEO_SELECT_LANES=8, VIDEO_DP_BLOCK=8, VIDEO_PATH_SCHEDULE=tiled,
VIDEO_CENSUS=classic, VIDEO_EXECUTION=graph, VIDEO_PROJECTION=shared,
VIDEO_FUSION=vertical and VIDEO_MAPPABLE=true. IREE memoization stays off.

Isolated C100 runtime: `/root/nxiree/sgm-serving-runtime`.
Validated model: `/root/nxiree/sgm-vga30-tools/selectopsrefine.bin`.
Prepared inputs: `/root/nxiree/sgm-vga30-input`.
Network runner: `/root/nxiree/sgm-vga30-tools/network.exs`; expected manifest
`expected-network.txt` in the same directory. Set VIDEO_CONCURRENCY=4 for the
idle-CPU replay, VIDEO_FPS=30, VIDEO_FRAMES=600 (or 1200), and supply the
side-by-side rawgray clip from the host. Default concurrency stays one for
compatibility. Transport-only VIDEO_MATCH=false does not load GPU models.

Guard/controller code belongs in the dedicated `c100_gpu` library. Existing
historical errors require explicit diagnostic mode and timeout <=120 s.
The controller stops/unloads CPU helpers and restores environment in `after`.
Do not replace production libraries. CPU stays schedutil, cap 1.479 GHz;
GPU stays simple_ondemand, 76.8–921.6 MHz; EMC stays 1.6 GHz. No voltage,
U-Boot, firmware, clock-limit, flash or reboot change occurs. Boot ID remains
`6f672cb9-e70d-433c-8894-7231f93bf6b7`, validated slot B.

Raw evidence is alongside this report. Model/data/profile cache:
`~/.cache/stereo-bench/vga30-20261022`; network source:
`~/.cache/stereo-bench/network60-20261022/stereo640-60.gray`.
Reproduction controls are in that cache's `controls/` directory. Full scratch
and source backups are archived in `scratch-archive/`; they retain historical
paths, while `controls/stream-network.sh` uses its own directory for logs and
controller lookup. For example, from the sibling `nx_iree` project:

```sh
VIDEO_CONCURRENCY=4 VIDEO_FPS=30 VIDEO_TAG=next-vga30 \
NETWORK_BUNDLE=/root/nxiree/sgm-vga30-tools/selectopsrefine.bin \
VIDEO_EXPECTED=/root/nxiree/sgm-vga30-tools/expected-network.txt \
bash ~/.cache/stereo-bench/vga30-20261022/controls/stream-network.sh \
  next-vga30 600 0 \
  ~/.cache/stereo-bench/network60-20261022/stereo640-60.gray
```

Choose a new case name and VIDEO_TAG for each run; inspect the guard and
pacing results before starting another. The script does not flash or reboot.
The default two-context CPU-loaded test remains a lower-latency alternative,
not an idle-CPU guarantee. One-context matching and transport-only modes also
pass 50-frame regression replays after the concurrency changes.

Final cleanup removes owned board output maps and isolated driver variants,
retaining the model, tools, inputs and runtime for reproduction. Only the main
BEAM remains, CPU helper is unloaded, debug environment is restored and the
TCP listener is closed. [Final state](final-state.log) confirms the unchanged
driver/runtime hashes, boot ID, validated slot and governors, with 23 historical
GPU errors and zero new ones. Available memory is about 3.58 GB. Local
temporary directories and the debug journal are archived outside the source
tree. No commits or pushes were made. Earlier reports remain valid
for their smaller 320×136 workloads, not this native VGA measurement.
