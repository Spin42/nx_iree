# Stereo device timing, saturation and BEAM responsiveness

CPU parity is not the acceptance target. The GPU must provide a worthwhile
end-to-end throughput advantage at comparable quality while leaving CPU
capacity for Elixir. **The current implementation does not meet that bar.**
The findings here identify larger targets and reject several explanations;
they are not a claim that the GPU's useful capacity is limited to these FPS.

## Actual device execution

Existing IREE tools can timestamp opaque Vulkan shaders. The prior
`--dispatch_statistics` requests lightweight statistics, which Vulkan resolves
without dispatch timestamps. Explicit
`--device_profiling_mode=dispatch-events,device-queue-events,queue-events`
enables native timestamp queries around dispatch. No shader or executor was
written to obtain these measurements.

A single-shot capture is cold and unsuitable as a steady-state measurement.
A separately bounded **five-invocation** native capture contains 25 valid
dispatch events with calibrated device clocks and no dropped records.
Its first invocation takes 61.20 ms of device work; the following four take
7.260, 7.266, 7.266 and 7.233 ms. Their stage means are:

| Stage | Warm device ms | Share |
| --- | ---: | ---: |
| Census | 0.579 | 8.0% |
| Three directional paths | 4.121 | 56.8% |
| Left selection and subpixel refinement | 2.288 | 31.5% |
| Shared projected consistency | 0.123 | 1.7% |
| Median | 0.144 | 2.0% |

Paths and selection account for about **88%** of measured GPU execution.
This is the existing 320×136/D64 classic-Census, three-path, shared-projection
mapping. It is an instrumented native capture on four warmed invocations,
not a device limit, a 138-FPS end-to-end forecast or a confidence interval.
It excludes Elixir input transfer/readback. The profiler can perturb
scheduling, and devfreq remains enabled. Do not subtract these numbers from
unrelated frame timings and label the remainder CPU overhead.

Artifacts: [raw profile](pipeline-warm.ireeprof),
[dispatch rows](dispatch-warm.jsonl), [clock/record summary](summary-warm.jsonl),
[stage means](gpu-stage-means.txt), [native run](profile-warm-board.log).
A separate one-shot run also compares its result against the previously
validated board output: [run](profile-board.log), [profile](pipeline.ireeprof).

The inspected IREE base is `89ece87b89852100c0a4ac3d515e14842d090967`
(`iree-3.13.0rc20261016`) with this repository's patches. Relevant upstream
source is [lightweight statistics](https://github.com/iree-org/iree/blob/89ece87b89852100c0a4ac3d515e14842d090967/tools/iree-benchmark-module-main.cc#L470),
[Vulkan profiling selection](https://github.com/iree-org/iree/blob/89ece87b89852100c0a4ac3d515e14842d090967/runtime/src/iree/hal/drivers/vulkan/logical_device.c#L251),
and [timestamp insertion](https://github.com/iree-org/iree/blob/89ece87b89852100c0a4ac3d515e14842d090967/runtime/src/iree/hal/drivers/vulkan/command_buffer.c#L1836).

Reproduce with the existing graph VMFB and prepared s32 inputs:

```sh
iree-benchmark-module --device=vulkan --module=pipeline.vmfb --function=main \
  --input=136x320xi32=@frame_0001.left.bin \
  --input=136x320xi32=@frame_0001.right.bin \
  --benchmark_min_time=5x --benchmark_repetitions=1 \
  --device_profiling_mode=dispatch-events,device-queue-events,queue-events \
  --device_profiling_output=pipeline.ireeprof
iree-profile dispatch --format=jsonl --dispatch_events pipeline.ireeprof
```

Use the patched tool with the mapped-input flush fix, and the C100 live
guard when running on the board. Never use the earlier rejected rapid
submission benchmark as timing evidence.

## Does feeding more frames solve it?

The portable [throughput benchmark](../../../bench/stereo/census/throughput.exs)
uses existing IREE contexts and `Task.async_stream`, with one owner per
context. It processes all 50 pairs, includes upload, matching, synchronization,
readback, task scheduling and thermal guards in wall time, and writes outputs
after timing. Three warmups per context and input preparation are excluded.
It reports processing latency and whole-sequence throughput separately.

| Contexts | alley_1 wall FPS | temple_2 wall FPS | alley p50 latency ms | temple p50 latency ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 51.76 | 56.25 | 15.216 | 14.771 |
| 2 | 65.94 | 65.73 | 23.008 | 22.769 |
| 4 | 67.10 | 66.23 | 40.435 | 39.844 |

Four contexts add latency without a useful further throughput gain.
Task concurrency alone is not the required architectural improvement.
These wall figures include guards, unlike older summed per-frame timings;
do not call their difference a regression or speedup against that older
boundary. All 100 downloaded two-context outputs are unchanged:
[agreement](agreement.txt). [One/two runs](board.log), [four](four-board.log),
[host discrete GPU](host-two.log).

## Ordinary-scheduler transfer waits

Source inspection found `allocate_buffer` and `read_buffer` registered as
ordinary NIFs although their IREE transfer calls can wait for the device.
The generic fix registers these two calls as dirty I/O, as already done for
GPU invocation. This reuses OTP's scheduler support. [OTP guidance](https://www.erlang.org/doc/apps/erts/erl_nif.html#long-running-nifs)
requires long native operations to avoid monopolizing ordinary schedulers;
dirty I/O permits waiting and limited data movement.

An isolated NIF comparison keeps the shaders and mapped runtime unchanged.
A regular BEAM heartbeat requests a 1-ms receive timeout during the timed
window. Its intervals measure scheduling responsiveness, not hard real-time
guarantees or CPU utilization. Paired two-context results:

| Sequence | Original heartbeat p95 ms | Dirty-I/O p95 ms | Original FPS | Dirty-I/O FPS |
| --- | ---: | ---: | ---: | ---: |
| alley_1 | 12.497 | 2.354 | 69.29 | 64.55 |
| temple_2 | 12.397 | 2.418 | 66.55 | 64.23 |

The scheduler fix improves responsiveness, **not stereo throughput** in
these runs. Maximum heartbeat delays still reach 10–17 ms. Single-context
throughput also decreases in this comparison. Removing blocking transfers
from ordinary schedulers is necessary for usable Elixir integration; more
dirty-job boundaries are not a throughput solution. Future native pipeline
integration should reduce those boundaries while retaining scheduler safety.
[Paired runs](dirty-board.log), [all 100 unchanged outputs](dirty-agreement.txt).

The host NIF is rebuilt from the current source; the normal board NIF is
untouched. Its separate experimental NIF is in
`/root/nxiree/sgm-pipeline-dirty-runtime`, sharing the previously tested mapped
runtime. Full regression evidence is [here](dirty-full-tests.log).

## Explanations ruled out and next priorities

Retained GM20B compiler output already shows `readonly|reorderable` NIR
loads and `ld.global.a64.constant.b32` in the paths shader. The local Mesa
26.1.8 NAK implementation maps that to Maxwell's read-only cache mode.
Adding `restrict` cannot enable a cache mode already in use. The prior
compiler evidence also reports no path spills. Neither is a supported
explanation for the current large performance shortfall.
[Captured compiler excerpts](compiler-cache-evidence.txt) retain the NIR
access flags, constant-load instructions and GPR/local-storage summary.

The high-value work is now aggregation/selection execution, avoiding
unnecessary materialization between them, and native pipeline boundaries.
Use licensed upstream libSGM/OpenCV methods and existing runtime facilities.
Do not optimize the 0.123-ms consistency pass as if it still dominated,
promise a multiplier from theoretical FLOPS, or accept CPU parity as success.

## Guard and board state

The dedicated C100 guard can now directly own a native executable from a
trusted `{absolute_executable, argv}` external-term file selected through
`C100_GPU_COMMAND_FILE`. `C100_GPU_TIMEOUT_MS` bounds execution to at most
120 seconds. Native PID ownership avoids an unmonitored grandchild when
killing a wrapper. The unset option retains the existing Elixir path.

Actual QA: native capture succeeds, the original Elixir fixture path succeeds,
and a direct `/bin/sleep 2` is killed/rejected at the configured 500-ms
deadline. Its PID is absent afterward. The C100 suite passes four tests
with one hardware test excluded. [Timeout](native-timeout-board.log),
[Elixir path](guard-elixir-board.log), [suite](c100-suite.log).
The initial timeout-wrapper probe failed because that executable was absent;
it never dispatched GPU work and is rejected setup evidence:
[log](profile-missing-timeout-rejected.log).

All completed GPU diagnostics retain the 23 historical errors and report
NEW_GPU_ERRORS=0. The earlier native stress failure remains unresolved.
Boot, validated slot B and normal governor/limits are unchanged; no reboot,
firmware, driver or default target NIF was replaced. Temporary environment
overrides were restored: [final state](final-board-state.log).

Source and artifact identities are in [sha256.txt](sha256.txt). Reusable
experimental binaries and downloaded predictions are cached under
`~/.cache/stereo-bench/saturation-20261022/`. Native capture commands and
temporary board conductors are removed after recording the evidence.
