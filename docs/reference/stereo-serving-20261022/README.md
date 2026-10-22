# Exported GPU stereo through Nx.Serving

The existing GPU matcher now runs through standard `Nx.Serving` input and
output streams without a compiler on the C100. At **320×136 / D64**, one
loaded IREE context processes **1,000 scheduled pairs in about 10.001 s**
while two independent CPU workers run. This validates useful CPU offload
through an application scheduling API. It does not establish live-camera
operation, hard deadlines or long-run driver stability.

| Clip, two CPU workers | Whole-loop pairs/s | Matcher CPU cores | Arrival-to-readback p50 / p95 | Scheduled-arrival-to-readback p95 / max |
| --- | ---: | ---: | ---: | ---: |
| alley_1 | 99.990 | 0.638 | 10.413 / 21.116 ms | 39.859 / 51.326 ms |
| temple_2, clean repeat | 99.985 | 0.669 | 12.253 / 24.136 ms | 40.536 / 55.966 ms |

The target is 100 pairs/s, with an absolute 10-ms arrival schedule. Both
runs finish with about 11 ms scheduled latency, so they catch up after
temporary lateness. This is throughput with pipeline latency: only 3 / 5
of the 1,000 outputs meet a 10-ms completion deadline. Arrival lateness
p95 is 19.116 / 20.334 ms, maximum 31.059 / 35.281 ms. No frames are dropped.
Each run repeats its 50-pair clip twenty times; it is not 1,000 distinct
camera frames or a thermal-equilibrium test.

An initial 100-pair **idle-CPU** control only reaches **79.29 pairs/s**,
with arrival lateness growing to 249 ms. A 100-pair loaded control reaches
99.21 pairs/s and recovers its initial lateness. Therefore this does not
prove 100-Hz operation with arbitrary background activity. CPU power/scheduling
behavior remains an investigation target; these observations alone do not
identify its cause. Normal governors and frequency limits are unchanged.

The first long temple run overlaps SFTP activity during its startup/window.
Its 147.645-ms maximum scheduled latency is retained in
[the diagnostic log](board-loaded-temple-transfer.log), excluded from the
main table. The clean repeat runs with no concurrent file transfers.

## Existing libraries and the integration fix

`NxIREE.Serving.new/1` is a small wrapper around the existing
`Nx.Serving.Default` implementation. It uses public `Nx.LazyContainer`
materialization, uploads owned host inputs, borrows resident inputs, runs
the loaded export, and adds the serving axis to resident outputs. It does
not implement another queue, worker pool, compiler or GPU algorithm.

Each `Nx.Batch.stack/1` entry is one complete exported invocation. The
maximum batch size is one. Nx.Serving serializes stream submission until
the preceding invocation completes. There can be one prefetched entry;
applications must consume output promptly. Unlimited independent clients
or a slow output consumer are not covered by a global queue bound.

Previously, eager reshape/squeeze called the compiler. That fails on a
compiler-free target merely to add/remove a batch axis. The backend now
uses IREE's existing `iree_hal_buffer_view_create` and checked shape helpers.
Each view independently retains its buffer; source metadata is not mutated.
There is no GPU dispatch or tensor copy for these operations.

The owner regression fails before the fix with
`iree-compile: Unknown command line argument '--invalid-metadata-compile'`
and passes afterward: [red](reshape-red.log), [green](reshape-green.log).
Tests cover independent alias deallocation, native use of the new dimensions,
scalar squeeze, invalid/overflowing dimensions and released inputs.
Serving tests cover changing streamed inputs, compiler-disabled execution,
resident outputs and preservation of caller-owned inputs:
[before](serving-red.log), [after](serving-green.log).

The full host suite reports **817 passed (744 doctests, 73 tests), 1 skipped**:
[full suite](full-tests.log). Additional resident-view tests run on the
RX7800XT with actual Vulkan computation: [Vulkan views](vulkan-views-final.log).
The same paced runner processes 100 pairs on RADV at 100.105 pairs/s and
checks all 50 distinct maps against earlier host predictions:
[host replay](host-paced.log). These tests run without a shader change.

## Measurement boundary and correctness

Timing includes Nx.Serving scheduling, uploads, the complete GPU matcher,
output synchronization/readback, per-frame thermal/memory guards and repeat
comparison. Input loading/preparation, three inline warmups, file writes and
comparison with an external reference directory are excluded. One context
is used here; the preceding bulk benchmark uses two contexts and therefore
its 150–152 pairs/s is a different execution boundary.

Inputs are the existing rectified, grayscale prepared s32 Sintel pairs.
Capture, decoding, rectification and image conversion are not measured.
The matcher, its quality settings, compiler flags and model bytes remain
the preceding libSGM-derived three-path implementation. It is not identical
arithmetic to CPU OpenCV 3WAY. See the
[preceding CPU comparison](../stereo-concurrency-20261022/README.md).

All repeated copies agree within each run. The **200 downloaded C100 maps**
from the four retained configurations exactly match their earlier GPU
predictions: [agreement](board-agreement.log). The host runner independently
checks its 50 maps. Raw per-frame timings and predictions are cached under
`~/.cache/stereo-bench/serving-20261022/`, including `timing.csv` in each output
directory. [Artifact hashes](artifact-hashes.sha256),
[source/runtime hashes](source-hashes.sha256).

Background work is the existing dedicated C100 OTP/OpenSSL SHA256 probe,
with two workers in the main Nerves BEAM and matching in a child BEAM.
It uses about 1.96 CPU cores over the full guarded callback. That accounting
window includes startup and output handling, while matcher CPU usage above
covers the measured stream. They must not be summed as a precise shared-window
CPU measurement. This is controlled synthetic work, not a named application.

## Board state and reproduction

Every completed board run reports `NEW_GPU_ERRORS=0`. The 23 historical
`nv50cal_space: -110` messages remain, so clean driver acceptance is open.
Boot ID stays `6f672cb9-e70d-433c-8894-7231f93bf6b7`, validated slot B,
firmware UUID `1ffb5cb3-94bd-54a7-f1e2-ec9661ce6746`. No firmware upload,
reboot, voltage, default native-library, CPU/GPU governor or limit change
occurs. CPU stays schedutil, cap 1479000 kHz; GPU stays simple_ondemand,
76.8–921.6 MHz; EMC remains 1.6 GHz. [Final state](final-state.log).

The new runtime is isolated in `/root/nxiree/sgm-serving-runtime`. It copies
the preceding corrected runtime, replacing only the experiment's ARM NIF,
runtime DSO and updated BEAM modules. The unchanged model remains
`/root/nxiree/sgm-final-mappable.bin`. Runner:
`/root/nxiree/sgm-serving-tools/serving.exs`. Reproduction controller:
`~/.cache/stereo-bench/serving-20261022/board_control.exs`:

```sh
elixir board_control.exs 2 alley 20 100
elixir board_control.exs 2 temple 20 100
```

Arguments are CPU-worker count, clip, repetitions and scheduled pairs/s.
It restores environment and stops/unloads the temporary CPU helper in
`after`. The existing child guard stops immediately on new GPU errors,
temperature/memory guard failure, abnormal exit or timeout. A pacing miss
is reported in the timings; the correctness success marker is not a
deadline assertion. Temporary board outputs are removed after verification;
isolated runtime/tools remain for reproduction.

The generic sources are `lib/nx_iree/serving.ex` and
`bench/stereo/census/serving.exs`. C100 policy and guards remain in the
dedicated `c100_gpu` library. Live cameras are absent (`/dev/video*` is empty).
Remaining work includes idle/bursty CPU pacing, camera capture/synchronization,
longer stability testing and standard Evision GPU integration.
