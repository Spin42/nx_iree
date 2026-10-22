# Projected stereo selection and native-runtime findings

Follow-up: the [composed IREE pipeline](../stereo-graph-20261022/README.md)
now reaches 49–53 FPS in bounded diagnostics, with no new GPU errors but the
earlier native-submission failure still unresolved. Statements below about
unimplemented graph export describe the earlier experiment.

The OpenCV-style projected-right-map experiment **does not improve C100
throughput**. Keep the preceding [28–29 FPS eight-path / 36–39 FPS three-path
candidate](../stereo-census-reuse-20261022/README.md) as the measured best.
Optimized CPU OpenCV 3WAY still leads at 94–101 FPS. The performance goal
remains open.

Two independent defects were exposed while measuring: a missing IREE CLI
input flush (fixed and reproduced), and a Nouveau channel-space timeout
during rapid native submissions (unresolved). No firmware, kernel, driver,
target NIF, voltage or clock policy was changed. No reboot occurred.

## Projected-right-map experiment

`bench/stereo/census/select_projected.comp` retains the libSGM left winner
and cost; `consistency_projected.comp` adapts OpenCV's projected pseudo-right
map, tie behavior and floor/ceil consistency rule. `filter.comp` provides
the subsequent median-only pass. Apache-2.0 and BSD-3-Clause source notices
and licenses are retained. The pinned OpenCV source is
[stereosgbm.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/stereo/src/stereosgbm.cpp).

The portable implementation gathers candidate projections. It does not
reproduce OpenCV's O(pixels) mutable scatter implementation: each projected
right lookup searches up to D left winners. That extra work and an added
dispatch are plausible costs; measured stage timings have not established
their individual GPU costs.

Same preparation and timing boundary as the preceding report: all 50 pairs
per scene, 320×136, D64, three warmups, transfers and synchronization
included, decode/resize/disk excluded. Classic Census, tiled paths, P1=8,
P2=64, uniqueness=1/0.95 remain fixed.

| Variant | Paths | alley_1 FPS | temple_2 FPS |
|---|---:|---:|---:|
| Initial projected selection, 32 lanes | 3 | 29.22 | 27.76 |
| Initial projected selection, 32 lanes | 8 | 20.61 | 21.33 |
| Projected selection, 8 lanes | 3 | 32.89 | not run |
| Packed loads + scalar consistency, 8 lanes | 3 | 30.26 | not run |

Initial runs: [board.log](board.log). Lane trials: [lanes-board.log](lanes-board.log).
Current packed/scalar source: [packed-board.log](packed-board.log), including
four intermediate oracle cases for each of 4/8/16 lanes on host and C100.
The initial mapping also passed D128/eight-path fixtures. The final packed
mapping was tested with D64/three paths; do not extend that claim to other
dimensions or path counts without rerunning them.

The initial three-path host outputs give common-region bad1-or-rejected
**4.481% / 2.142%**, bad3-or-rejected **2.251% / 0.593%**, accepted MAE
0.541 / 0.245 pixels. These are **host-output** quality measurements:
[alley](quality-three-alley-host.json), [temple](quality-three-temple-host.json).
Real-frame projected board outputs have not been downloaded and compared.
The throughput result already rejects this implementation as a speedup.

`projected_oracle.py` uses an independent CPU scatter and OpenCV median
filter to generate intermediate expectations. `check_projected.exs` tests
defined disagreements, missing correspondences, equal-cost ties, integer
correspondences and image/search boundaries. Its C100 result is
`PROJECTED_EDGES_PASS`: [edge QA](nif-edges-board.log).

Export with `VIDEO_SELECT=projected`, optionally `VIDEO_SELECT_LANES=4|8|16|32`.
After generating base fixtures, run `uv run
bench/stereo/census/projected_oracle.py FIXTURE_DIRECTORY` from the repo root.
Add `--disparities 128` for those fixtures. This rewrites only oracle files;
use a copied fixture directory to retain the preceding algorithm's oracles.

## Confirmed generic IREE fix

The same edge inputs passed NxIREE but failed native `iree-run-module`.
Literal all-invalid inputs also failed. Explicit matcher output invalidation
did not fix the failure; that speculative source change was restored.

IREE's `iree_hal_buffer_view_generate_buffer_in_situ` invokes the generator
on mapped device memory and unmaps it without flushing. On this non-coherent
device the CLI's generated inputs can remain stale. Four lines now flush
after a successful callback and propagate any flush failure through existing
cleanup. See [patch 0003](../../../patches/iree/0003-flush-generated-buffer-mappings.patch).
NxIREE already uploads through a flushing transfer API; this fix does not
explain or change the preceding stereo FPS.

Alternating original and flush-only native tools, using identical VMFB and
NumPy inputs: original fails 3/3; flush-only passes 3/3. Separate NumPy and
literal tests pass 3/3 each with the fix. Evidence:
[toggle](native-toggle-board.log), [six passing checks](native-flush-board.log),
[rejected invalidation experiment](native-invalidate-board.log).
All seven existing `BufferViewUtilTest` cases pass:
[suite](buffer-view-tests.log). The CPU suite does not reproduce non-coherent
hardware; the alternating C100 runs are the regression proof.

Only isolated CLI binaries were staged on the board. The application NIF
and Vulkan libraries were not replaced. Tool/model/source digests are in
[sha256.txt](sha256.txt).

The retained `consistency.vmfb`, `left.npy`, `winners.npy`, `expected.npy`
allow the same native regression to be rerun after resolving the later
runtime stress failure. With a correctly built native tool and matching
device libraries, from this evidence directory:

```sh
NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1 iree-run-module \
  --device=vulkan --module=consistency.vmfb --function=main \
  --input=@left.npy --input=@winners.npy --expected_output=@expected.npy
```

Literal variant: replace the two inputs with `--input=17x73xf32=-1` and
`--input=17x73xui32=4294967295`, and the expectation with
`--expected_output=17x73xf32=-1`. These commands ran through isolated native
CLIs before the later stress failure, not through a CPU matcher.

## Rejected native profiling run: channel-space timeout

To separate invocation overhead from device work, a temporary host export
added IREE's `--iree-hal-benchmark-dispatch-repeat-count=32`. The upstream
pass repeats each dispatch with intervening barriers. These synthetic zero
inputs are a profiling probe, never a frame-FPS measurement.

During the **N=1 median** invocation, the first 391-iteration repetition
reported 0.461 ms. Subsequent repetitions stalled and emitted 23 messages:

```text
[21016.628276] nouveau 57000000.gpu: iree-benchmark-[4904]: nv50cal_space: -110
```

The tool still exited zero. The previous log regex missed this signature,
so the run printed a false success marker and continued through N=32 before
the manual log inspection caught it. **Reject that success marker and the
profiling conclusion.** This run cannot establish how much a composed IREE
pipeline would improve throughput. The process had exited before an attempted
kill. GPU work was then stopped until adding a live guard and an explicit
diagnostic baseline; the follow-up above documents subsequent bounded runs.

Evidence: [full benchmark output](native-repeat-rejected.log),
[full dmesg](native-repeat-failure-dmesg.log). `nv50cal_space` is logged when
`nvif_chan_gpfifo_wait` returns an error; its implementation includes both
main-push-buffer and GPFIFO-space waits. Which wait failed, and why, remain
unconfirmed. Ring wrap/submission pressure, synchronization/lifetime issues
and device progress are hypotheses, not established root causes. The local
kernel source is `drivers/gpu/drm/nouveau/{nouveau_exec.c,nvif/chan.c, nvif/chan906f.c}`;
the [upstream channel-progress change](https://git.zx2c4.com/linux-rng/commit/drivers/gpu?id=862450a85b85a3e88e9065c9d2421f3ef17d054c)
explains the semaphore-based progress tracking used by this kernel.

The dedicated `c100_gpu/scripts/stereo_board_run.exs` now shares its parser
with `board_gpu_log.exs`, rejects a boot already containing GPU errors,
polls the kernel log every 250 ms while a child runs, and terminates the
child on the first detected error. The new captured-signature regression
fails before the parser change and passes after it: [red](guard-red.log),
[green](guard-green.log). Full C100 library suite: four tests, zero failures,
one hardware test excluded: [suite](guard-suite.log).
The actual board harness now rejects the contaminated log before starting
a GPU child: [manual QA](guard-rejection-board.log).

Warm-boot acceptance remains the earlier 30 distinct passing boots; this
new runtime stress failure is separate and blocks further load acceptance.
Board remains boot `6f672cb9-e70d-433c-8894-7231f93bf6b7`, validated slot B,
`simple_ondemand`, min/idle 76.8 MHz, cap 921.6 MHz. Next investigate a bounded
fixed-iteration native reproduction with the live guard; do not hide the
failure by reducing iterations and declaring success. Generic graph export
has not been implemented. Row-shared projected scatter may avoid the current
gather cost, but it also remains unimplemented and unmeasured.
