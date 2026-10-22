# Fusion, mapped readback and CPU feeding

Follow-up: the [1,000-pair CPU-coexistence test](../stereo-concurrency-20261022/README.md)
uses normal governors and reaches 150–152 GPU FPS with two background CPU
workers, versus 51–52 for CPU 3WAY under the same work. These longer-run
conditions differ from the short measurements below.

The portable Vulkan prototype now has a conditional whole-loop throughput
lead over four-thread CPU OpenCV 3WAY at **320×136 / D64**. With matched,
temporary CPU-performance policy, GPU runs span **121–140 FPS**, versus
**84.72 / 89.02 FPS** for CPU on alley_1 / temple_2. This is about **1.36–1.65×**,
not a 2.9× gain, a hardware ceiling or NVIDIA-equivalent performance.
Normal CPU schedutil gives **100.65 / 103.97 GPU FPS**. Production acceptance
and standard Evision GPU SGBM integration remain open.

These are short, guarded 50-pair runs, two independent IREE contexts, three
warmups per context, normal GPU `simple_ondemand` at 76.8–921.6 MHz, and
unchanged EMC/firmware/voltage. Whole-loop time includes upload, matching,
readback, synchronization, task scheduling and thermal guards; input
preparation, disk I/O and warmups are excluded. CPU wall timing also includes
its whole sequence and guards, retaining outputs until after timing. GPU
stream throughput is not single-frame latency or camera-to-depth FPS.

## Measured comparisons

| Configuration | alley_1 FPS | temple_2 FPS |
| --- | ---: | ---: |
| CPU OpenCV 3WAY, four threads, CPU performance | 84.72 | 89.02 |
| Fused/vector GPU, staged output, CPU schedutil | 77.47–80.29 | 79.04 |
| Fused/vector GPU, staged output, CPU performance | 115.05–115.72 | 115.23–115.28 |
| Fused/vector GPU, mapped output, CPU schedutil | 100.65 | 103.97 |
| Fused/vector GPU, mapped output, CPU performance | 124.49–132.73 | 121.13–140.38 |

The paired repeat runs use the same corrected runtime for both allocation
policies: [alley baseline](baseline-repeat-alley-board.log),
[mapped alley](mappable-repeat-alley-board.log),
[temple baseline](baseline-repeat-temple-board.log),
[mapped temple](mappable-repeat-temple-board.log). The mapped policy gains
15.4% / 5.1% in those pairs. Other runs vary; these ranges are observations,
not confidence intervals. [First alley](mappable-performance-alley-board.log),
[first temple](mappable-performance-temple-board.log),
[normal alley](mappable-normal-alley-board.log),
[normal temple](mappable-normal-temple-board.log).

The CPU reference's processing-only FPS is 93.26 / 99.00; its whole-loop
84.72 / 89.02 figures above use the fairer boundary. Historical 100.78 / 93.71
figures used processing-only timing. [CPU alley](cpu-reference-alley-repeat-board.log),
[CPU temple](cpu-reference-temple-board.log). Set `OPENCV_MODES=3way` to select
this reference; the existing benchmark defaults still run all modes.
All 100 CPU reference outputs also equal the prior 3WAY outputs:
[CPU agreement](cpu-reference-agreement.log).

Linux process user+system CPU ticks (USER_HZ=100 on this target) give a coarse
capacity check. The paired mapped GPU runs use 23 ticks/376.714 ms and
24 ticks/412.779 ms, roughly **0.61 / 0.58 CPU cores** on average. CPU 3WAY
uses 203 ticks/590.155 ms and 197 ticks/561.644 ms, roughly **3.44 / 3.51 cores**.
These count the benchmark process's threads, not other services or IRQ time;
10-ms tick granularity limits precision. The BEAM heartbeat remains around
2–3 ms p95 during the GPU loop. This is scheduling evidence, not a real-time
guarantee.

A separate **single-context** check reaches **107.23 / 105.40 whole-loop FPS**
with p50 processing latency **7.881 / 7.850 ms**, still ahead of CPU whole-loop
throughput without overlapping frames. Two contexts improve stream throughput
but raise p50 latency to roughly 12–15 ms; choose according to latency needs.
Both checks use the same bounded CPU-performance policy and add zero GPU
errors. [One alley](one-alley-board.log), [one temple](one-temple-board.log).

## Retained changes

1. **Fuse vertical aggregation and projected selection.** The existing
   Apache-2.0 libSGM recurrence retains vertical-only state in registers,
   adds the two horizontal path costs for selection, and preserves the
   existing ties, uniqueness and subpixel math. `VIDEO_FUSION=vertical`
   removes one byte-cost volume write/read (5.57 MB/frame at this shape).
   Horizontal aggregation stays separate; the graph still has five dispatches.
   Classic feature and packed path accesses use explicit `uvec2` loads/stores,
   preserving the raw tensor byte layout. Fresh GM20B NAK output confirms
   64-bit global load operations without requiring Int64 arithmetic.
2. **Reuse IREE's mappable external-resource policy.**
   `external_resources_mappable: true` on either generic Vulkan exporter
   forwards existing `--iree-hal-external-resources-mappable=true`.
   `VIDEO_MAPPABLE=true` selects it for this benchmark. It affects external
   ABI buffers, leaves transient allocation policy alone, and still copies
   results into an Elixir binary. Default false preserves ordinary placement;
   requiring host-visible device-local memory can regress discrete GPUs.
3. **Fix mapped-source cache invalidation in IREE.** The first C100 mappable
   fixture failed its path oracle while RADV passed. IREE's direct mapped
   transfer read non-coherent memory without invalidation. The seven-line
   [0005 patch](../../../patches/iree/0005-invalidate-mapped-transfer-sources.patch)
   calls the existing invalidation API. Identical C100 fixtures fail with
   the old runtime and pass with the corrected one; the harness stopped at
   that first failure. [Hardware red](mappable-fixture-board.log),
   [green](mappable-fixture-fixed-board.log). Four native regression cases
   fail before the fix; all six pass afterwards, including error cleanup,
   coherent read and host upload. [Native red](transfer-native-red.log),
   [green](transfer-native-green.log).
4. **Reuse IREE profiling and direct mapped upload.**
   `NxIREE.Profiling.capture/3` exposes existing file profiling, with teardown
   in `after`. The generic [0004 patch](../../../patches/iree/0004-serialize-profile-metadata.patch)
   serializes metadata using IREE's existing sink lock; two concurrent
   contexts otherwise corrupted record headers. The native overlap proof
   fails before the fix and passes after it, including 100 repetitions:
   [red](native-profile-red.log), [green](native-profile-tests.log).
   Tensor upload also removes an intermediate native host allocation/copy
   and validates shape/type byte length: [old length acceptance](upload-length-red.log).

The actual two-context trace changes from **100 command buffers / 550
operations** to **50 / 500** for 50 frames, with all 250 dispatch events valid
and no dropped records. Each original graph has five dispatches and five
barriers, followed by a separate 174,080-byte readback-copy command buffer.
The mapped trace has no copy commands. [Original commands](final-baseline-command.jsonl),
[mapped commands](mappable-command.jsonl), [mapped summary](mappable-summary.jsonl).
These traces establish copy removal; FPS above comes from unprofiled runs.
The earlier four-barrier interpretation was incorrect.

The inspected upstream base is
`89ece87b89852100c0a4ac3d515e14842d090967` (`iree-3.13.0rc20261016`).
The existing policy is defined in
[ResolveTopologyQueries.cpp](https://github.com/iree-org/iree/blob/89ece87b89852100c0a4ac3d515e14842d090967/compiler/src/iree/compiler/Dialect/HAL/Transforms/ResolveTopologyQueries.cpp#L20).
No allocator or alternate executor was introduced.

## Correctness, rejected experiments and limits

All 100 downloaded fused predictions equal the preceding unfused graph
byte for byte; all 100 mapped predictions equal those fused predictions:
[fusion agreement](final-agreement.log), [mapped agreement](mappable-agreement.log).
Therefore the [previous quality measurements](../stereo-graph-20261022/README.md)
still apply: this is comparable test-scene quality, not exact OpenCV output
or a general accuracy guarantee. D64/three-path, D128/three-path and
D128/eight-path independent intermediate fixtures pass on RADV and C100,
including graph/separate agreement and caller-input preservation.
[D64](mappable-fixture-fixed-board.log), [D128 three](mappable-d128-three-board.log),
[D128 eight](mappable-d128-eight-board.log).
All 50 full-size RADV outputs are also byte-identical to its prior host
baseline. Cross-device subpixel rounding differs slightly; the maximum
absolute difference is below the existing 0.00001-pixel oracle tolerance,
so cross-device byte equality is not claimed. [Host agreement](host-agreement.log),
[host run](host-throughput.log).

The final core suite passes **813 cases** (744 doctests, 69 tests), with
shader regeneration enabled: [full suite](full-tests.log). The dedicated
C100 package reports four tests, zero failures and one hardware-only exclusion:
[C100 tests](c100-tests.log). Both native fixes have failing-before/passing-after
proofs above. All source patches pass isolated application checks.

Independent vertical disparity blocks 2/4/16 were slower than block 8 and
were removed. Block 4 improves occupancy but still loses throughput; register
count alone is not the bottleneck. A host-tuple/batched NIF argument path
also gave no repeatable gain and was removed. Its extra API/decoder is not
retained. Direct mapped upload remains on the ordinary tensor path.

`c100_gpu/scripts/cpu_performance_probe.exs` is an explicit bounded diagnostic,
not a boot default. It selects the existing CPU performance governor within
the unchanged 1479000-kHz maximum and restores the saved governor in `after`.
Successful runs and an intentional guard-timeout control verify restoration:
[timeout and restoration](cpu-restore-timeout-board.log). Normal-policy
staged runs toggle roughly 80→116→77 FPS when changing only CPU governor.
This identifies host feeding/power management as a large practical lever;
GPU arithmetic optimization alone does not remove that dependency.

All completed diagnostics add **zero** GPU errors, retaining 23 historical
`nv50cal_space: -110` messages. The earlier rapid native-submission failure
remains unresolved. Boot ID `6f672cb9-e70d-433c-8894-7231f93bf6b7`, validated
slot B and firmware UUID `1ffb5cb3-94bd-54a7-f1e2-ec9661ce6746` are unchanged.
No flash, reboot, default board native-library installation, voltage or GPU/EMC
policy change occurred. Final CPU governor is schedutil. This is diagnostic
performance evidence, not clean driver or sustained thermal acceptance.
[Final state](final-board-state.log), [owned temporary staging cleanup](board-cleanup.log).

## Reproduction

Use the compiler/shader-tool setup in the [benchmark README](../../../bench/stereo/census/README.md).
From `bench/`:

```sh
VIDEO_CENSUS=classic VIDEO_PATH_SCHEDULE=tiled VIDEO_PATHS=3 \
VIDEO_SELECT=projected VIDEO_SELECT_LANES=8 VIDEO_DP_BLOCK=8 \
VIDEO_PROJECTION=shared VIDEO_EXECUTION=graph VIDEO_FUSION=vertical \
VIDEO_MAPPABLE=true VIDEO_P1=8 VIDEO_P2=64 \
VIDEO_UNIQUENESS=1.0526315789473684 STEREO_OUT=/tmp/fused.bin \
  mix run --no-compile stereo/census/export.exs
```

Run `throughput.exs` with `VIDEO_CONCURRENCY=2`, `STEREO_BUNDLE`, `VIDEO_DATA`,
`VIDEO_OUTPUT` and `NXIREE_DEVICE_URI`. Optional `STEREO_PROFILE` captures
the loop separately. On C100 use the dedicated guarded child; the isolated
current runtime is `/root/nxiree/sgm-final-mappable-runtime`, the model is
`/root/nxiree/sgm-final-mappable.bin`, and final harness tools are in
`/root/nxiree/sgm-final-tools`. Existing default libraries remain untouched.

Models, corrected and pre-fix DSOs, raw profiles, prediction files and a
board reproduction controller are retained in
`~/.cache/stereo-bench/fused-20261022/`. Artifact hashes and final regression
logs accompany this report. The generic mappable flag is an opt-in for each
device; it is not C100 detection or a recommended universal Vulkan default.
