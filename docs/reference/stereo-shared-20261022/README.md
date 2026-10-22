# Shared projected consistency and allocator-aware uploads

The portable three-path GPU matcher now reaches **61–69 FPS** in bounded
C100 diagnostics, with outputs unchanged from the preceding projected
matcher. Paired comparisons show **9–20%** improvement from replacing the
projected-right gather with OpenCV's scatter method. CPU OpenCV 3WAY still
leads at 100.78/93.71 FPS. The GPU performance goal remains incomplete.

All runs use 320×136/D64, classic Census, three tiled paths, P1=8, P2=64,
uniqueness=1/0.95, eight selection lanes and one composed IREE invocation.
All 50 Sintel pairs per sequence are processed after three warmups. Timings
include upload, complete GPU matching, readback and completion; preparation,
decode and disk I/O are excluded. Normal devfreq remains enabled.

## Reused algorithm and measured comparison

`consistency_projected.comp` adapts the already-attributed OpenCV winner
scatter, preserving its packed cost and equal-cost preference for larger
disparity. One workgroup per row initializes a shared right map, scatters
accepted winners with `atomicMin`, synchronizes, then checks the existing
floor/ceil correspondence rule. Only shared storage is mutable. Width 320
needs 1,280 shared bytes. The exporter limits this mode to width ≤4096.
The preceding gather remains available; no new matcher or executor is added.

| Sequence | Gather, paired FPS | Shared, paired FPS | Gain | Shared initial FPS |
| --- | ---: | ---: | ---: | ---: |
| alley_1 | 55.73 | 60.85 | 9.2% | 68.65 |
| temple_2 | 54.44 | 65.46 | 20.2% | 60.83 |

Paired p50/p95 latency is 16.828/25.152 → 16.147/23.337 ms on alley and
17.630/25.538 → 15.118/19.017 ms on temple. These are individual full-sequence
comparisons, not confidence intervals. Devfreq and timing variability prevent
attributing the larger change from earlier 49–53 FPS entirely to scatter.
Evidence: [initial shared run](shared-board.log), [paired repeat](repeat-board.log).

All 100 downloaded shared outputs are byte-identical to the previous
projected GPU outputs. Thus the previously measured bad1-or-rejected
4.481%/2.142%, bad3-or-rejected 2.251%/0.593%, and accepted MAE 0.541/0.245 px
are unchanged. These two resized synthetic scenes do not establish general
accuracy. [Agreement](shared-agreement.txt),
[preceding quality measurements](../stereo-graph-20261022/README.md).

The earlier native compiler artifact reports no path spills; path spilling
was not the reason to make this change. A fresh separate-stage baseline
measures projected consistency at 6.50 ms including invocation overhead:
[baseline](baseline-board.log). The shared scatter removes up to 2D global
candidate probes per output; this is a work-count improvement, not a claim
of a 64× stage speedup or a GPU timestamp measurement.

## Generic IREE upload change

`IREETensor::upload` now reuses IREE's `iree_hal_buffer_view_generate_buffer`
instead of always using `allocate_buffer_copy`. IREE queries whether a
high-performance mapping is available, invokes a memcpy callback directly
into compatible device memory, and retains its staged fallback elsewhere.
No device-specific memory type is forced and no allocator is introduced.
The existing [IREE flush patch](../../../patches/iree/0003-flush-generated-buffer-mappings.patch)
is required for correctness on non-coherent mapped memory.

An isolated C100 runtime was selected through a separate runtime root; each
child verifies its actual loaded DSO path from `/proc/self/maps`. The normal
board NIF/runtime was not overwritten. Paired runs keep the shared matcher
and inputs fixed:

| Sequence | Original upload ms | Mapped upload ms | Original FPS | Mapped FPS |
| --- | ---: | ---: | ---: | ---: |
| alley_1 | 2.679 | 1.729 | 59.83 | 59.42 |
| temple_2 | 2.957 | 1.587 | 54.86 | 61.64 |

Upload time improves, but total FPS does not improve consistently. Do not
claim this change closes the performance gap or isolate GPU clocks as its
explanation. Graph/readback times also vary. [Paired runtime log](mapped-board.log),
[phase means](mapped-phase-means.txt). All 100 mapped-runtime outputs remain
byte-identical: [agreement](mapped-agreement.txt). Host RADV full-frame output
is also unchanged: [host agreement](mapped-host-agreement.txt).

## Correctness and runtime status

- D64/three-path and D128/eight-path independent intermediate fixtures pass
  on RADV and C100. They check Census, paths, winner selection, consistency,
  median output, graph/separate agreement and input preservation.
- Explicit consistency edge cases pass on both devices, including conflicts,
  missing matches, equal-cost ties and search/image boundaries.
- The retained gather branch compiles and passes the original host fixtures.
- Full NxIREE regression run: **808 passed, 1 skipped** because shader-tool
  paths were absent. That owner test separately passes with explicit tools:
  [full suite](full-tests.log), [shader regeneration](shader-tools-test.log).
  All shader exports also run SPIR-V validation.

Host evidence: [D64](fixture-host.log), [D128](d128-fixture-host.log),
[mapped D64](mapped-fixture-host.log), [edges D64](edges-host.log),
[edges D128](d128-edges-corrected-host.log), [gather](gather-fixture-host.log).
Target evidence: first children of the shared/repeat/mapped logs above and
[edge checks](edges-board.log). A host D128 edge invocation and a target D64
edge invocation initially lacked their `.npy` setup files and failed before
matching. The target harness stopped immediately; these are rejected setup
runs, not passing GPU tests. Fixtures were staged explicitly before reruns.
[Rejected target setup](edges-missing-fixture-rejected.log).

Every completed diagnostic reports **NEW_GPU_ERRORS=0** while retaining the
23 historical `nv50cal_space: -110` messages. The earlier native submission
failure remains unresolved; these results do not establish clean load
acceptance. Boot ID remains `6f672cb9-e70d-433c-8894-7231f93bf6b7`, validated
slot B, `simple_ondemand`, min 76.8 MHz and cap 921.6 MHz. No reboot, firmware,
driver, default target native-library or power-policy change occurred.
[Final board state](final-board-state.log).

## Reproduction and retained artifacts

Use the tool setup in the [benchmark README](../../../bench/stereo/census/README.md).
From `bench/`, export with:

```sh
VIDEO_CENSUS=classic VIDEO_PATH_SCHEDULE=tiled VIDEO_PATHS=3 \
VIDEO_SELECT=projected VIDEO_SELECT_LANES=8 VIDEO_PROJECTION=shared \
VIDEO_EXECUTION=graph VIDEO_P1=8 VIDEO_P2=64 \
VIDEO_UNIQUENESS=1.0526315789473684 STEREO_OUT=/tmp/shared.bin \
  mix run --no-compile stereo/census/export.exs
```

Run the existing benchmark and C100 live-error guard. Export `gather` for
the comparison. Independent fixtures and the projected oracle remain the
existing reusable scripts, with matching dimensions/path parameters.
Build the generic runtime normally with the existing patched IREE tree;
the upload change needs no extra configuration.

Validated data/bundles and original/mapped DSOs are retained under
`~/.cache/stereo-bench/shared-projection-20261022/` for follow-up, with source
and artifact hashes in [sha256.txt](sha256.txt). The isolated target runtime
is retained at `/root/nxiree/sgm-next-runtime`, plus `sgm-next-*` benchmark
bundles and result directories. They are experiment artifacts, not a firmware
installation. Temporary conductor/path-printing scripts are removed after QA.
