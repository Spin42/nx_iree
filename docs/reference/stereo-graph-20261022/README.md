# One IREE invocation per stereo frame

Follow-up: [OpenCV-style shared projection](../stereo-shared-20261022/README.md)
now reaches 61–69 FPS with byte-identical predictions. Paired comparisons
show 9–20% gains. Generic allocator-aware uploads reduce upload time, but
do not establish a consistent additional FPS gain. CPU remains faster.

Composing the unchanged three-direction kernels through the generic
`NxIREE.Vulkan.export_pipeline/4` improves C100 throughput from 36–39 FPS to
**48.93 / 49.38 FPS**, on all 50 alley_1 / temple_2 pairs respectively.
All 100 downloaded outputs are byte-identical to the preceding matcher:
[agreement](agreement.log). The composed projected selector reaches
**49.13 / 53.18 FPS**, with its previously measured quality confirmed on
downloaded board predictions. CPU OpenCV 3WAY still leads at 100.78 / 93.71
FPS. This does not complete the performance goal.

Normal devfreq, unchanged firmware/NIF/driver and the same 320×136/D64
inputs, penalties and three warmups apply. Timings include upload, complete
matching, readback and synchronization, excluding decode/resize/disk. Source
and bundle digests are in [sha256.txt](sha256.txt).

| Pipeline | alley_1 FPS | temple_2 FPS | alley bad1/rejected | temple bad1/rejected |
|---|---:|---:|---:|---:|
| Separate, libSGM rolling selection | 38.57 | 36.08 | 4.849% | 2.353% |
| Composed, same shaders | 48.93 | 49.38 | identical | identical |
| Composed, projected selection | 49.13 | 53.18 | 4.481% | 2.142% |
| CPU OpenCV 3WAY | 100.78 | 93.71 | 4.434% | 2.284% |

Projected accepted MAE is 0.541 / 0.245 px; bad3-or-rejected is 2.251% /
0.593%. It improves some metrics and worsens others versus CPU. These two
resized synthetic sequences do not establish general accuracy. Evidence:
[original-selector board run](board.log), [projected board run](projected-board.log),
[alley quality](quality-projected-alley.json), [temple quality](quality-projected-temple.json).

The API exports named immutable-buffer dataflow as ordinary IREE MLIR
dispatches. IREE retains responsibility for allocation, barriers and command
submission. There is no custom executor or target-only native code. Its
single-shader API is preserved. Independent dependent-transform tests verify
changed inputs, returned intermediates/aliases, caller preservation and
invalid dataflow on host Vulkan. The full NxIREE suite passes **809 checks**:
[suite](full-tests.log). Both stereo mappings pass four intermediate oracle
cases and exact graph-versus-separate checks on RADV and NVK:
[host](fixtures-host.log), [projected host](projected-fixtures-host.log), and
the first child in each board log.

Export from `bench/` with `VIDEO_EXECUTION=graph`, along with the preceding
classic/tiled/P1=8/P2=64/uniqueness=1.0526315789473684 flags. Keep
`VIDEO_SELECT=fused VIDEO_PATHS=3` for the byte-identical comparison, or
`VIDEO_SELECT=projected VIDEO_SELECT_LANES=8 VIDEO_PATHS=3` for projection.
Individual stages remain in the bundle for oracle checks; measured frames
invoke only `pipeline`. The generic API is documented in
[vulkan-kernels.md](../../vulkan-kernels.md).

## Diagnostic status and remaining bottlenecks

These are **diagnostic runs after an unresolved native submission timeout**,
not clean driver acceptance. The dedicated C100 harness explicitly snapshots
the 23 existing `nv50cal_space: -110` messages when
`C100_GPU_DIAGNOSTIC=1`, retains them in output, stops on every new error,
and reports `BOARD_DIAGNOSTIC_PASS` rather than `BOARD_STEREO_PASS`.
Each child reports `GPU_ERRORS=23 NEW_GPU_ERRORS=0`. Default acceptance still
refuses a boot containing errors. The [preceding investigation](../stereo-projection-20261022/README.md)
records the rejected rapid native benchmark and confirmed generic IREE
mapped-input flush fix. Do not describe this board as fully load-stable.

Read-only 10 ms frequency sampling during another projected run observes
active frequencies from 460.8 to 921.6 MHz, often 691.2 MHz:
[samples](frequency.csv), [run](frequency-board.log). Startup/idle samples
are included. This is evidence that the normal governor does not hold peak
frequency, not proof of a particular clock-related speedup. No frequency,
voltage or governor setting changed. Board remains on validated slot B,
boot `6f672cb9-e70d-433c-8894-7231f93bf6b7`, min/idle 76.8 MHz, cap 921.6 MHz.

The graph removes avoidable invocation boundaries but not the sequential
path recurrence or projected gather work. A row-shared scatter was drafted
but removed uncompiled when the user redirected the investigation toward
OpenCV/Evision reuse; no row-scatter implementation or speed claim is retained.
The new [Evision acceleration probe](../evision-acceleration-20261022/README.md)
establishes a workable standard keyword UMat call and the remaining SGBM gap.
