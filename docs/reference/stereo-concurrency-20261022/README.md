# Stereo while other Elixir CPU work runs

The subsequent [Nx.Serving replay](../stereo-serving-20261022/README.md)
uses one context and scheduled frame arrivals: with two CPU workers it
processes 1,000 pairs in about 10.001 s, using 0.64–0.67 matcher CPU cores.
Its latency/pacing limits differ from the bulk throughput measurements below.

GPU offload now demonstrates its useful system-level advantage: at
**320×136 / D64**, GPU matching sustains **150–152 stereo pairs/s** while
two independent CPU workers run, versus **51–52 pairs/s** for four-thread
CPU OpenCV 3WAY with the same background work. That is about **2.9×** the
CPU matcher in this coexistence test. All runs use normal CPU/GPU governors;
no performance governor or fixed-clock diagnostic is selected.

| Clip | GPU, no background work | CPU, no background work | GPU, two CPU workers | CPU, two CPU workers |
| --- | ---: | ---: | ---: | ---: |
| alley_1 | 147.12 FPS | 85.11 FPS | 150.62 FPS | 51.92 FPS |
| temple_2 | 148.44 FPS | 83.95 FPS | 150.49–151.68 FPS | 51.42 FPS |

Each measurement processes its complete 50-pair clip **20 times in one
timing window**, 1,000 pairs total. There are four CPU runs and five GPU
runs, including a final loaded temple GPU repeat after the CPU runs. The
GPU measurement windows last about 6.6–6.8 seconds; loaded CPU windows
last about 19 seconds. These are repeated clips, not 1,000 distinct camera
frames or a thermal-equilibrium soak test.

Raw measurements: [GPU alley idle](gpu-normal-0-alley.log),
[GPU alley loaded](gpu-normal-2-alley.log),
[CPU alley idle](cpu-normal-0-alley.log), [CPU alley loaded](cpu-normal-2-alley.log),
[GPU temple idle](gpu-normal-0-temple.log), [GPU temple loaded](gpu-normal-2-temple.log),
[GPU temple loaded repeat](gpu-normal-2-temple-repeat.log),
[CPU temple idle](cpu-normal-0-temple.log), [CPU temple loaded](cpu-normal-2-temple.log).
[Extracted metrics](summary.csv).

## What runs concurrently

The dedicated `c100_gpu/scripts/cpu_load_probe.exs` uses existing OTP
`Task.Supervisor` and `:crypto.hash(:sha256, payload)` over a 64-KiB block.
The installed crypto library is OpenSSL 3.6.4. Zero or two supervised workers
run continuously in the main Nerves BEAM while the existing guard launches
the matcher in its isolated child BEAM. Their native worker pools are
independent. This is a controlled CPU workload, not a simulation of a named
application, camera processing or a RAM-bandwidth benchmark.

Both loaded GPU runs use about **0.71 CPU cores** in the benchmark child
during matching (0.75 in the final repeat). Loaded CPU matching uses about
**1.95 CPU cores**, with the other work competing for the four available
cores. Without background work, CPU matching uses 3.37–3.43 cores.

The main BEAM with two background workers averages **1.94–1.97 CPU cores**
over the full guarded callback. That accounting includes startup, matcher
execution, output validation/writing, monitoring and existing application
activity. It is a different window from the child's timed loop; do not
sum these figures into a precise same-window utilization claim. CPU ticks
are process user+system time, USER_HZ=100 on this target. The main BEAM's
zero-worker control averages about 0.04–0.05 cores over its callback.

With background work, GPU processing p50 is **11.4–11.6 ms**, p95 **14.6–15.0 ms**.
CPU processing p50 is **17.2–17.3 ms**, p95 **20.9–21.3 ms**. The GPU child’s
1-ms heartbeat has p95 intervals about **2.2 ms**. Processing latency excludes
the per-frame guards; whole-loop FPS includes them. These are scheduling
observations, not real-time deadline guarantees.

The longer windows differ from the previous short 50-pair benchmark. GPU
code, model, runtime and hardware limits are unchanged; do not label the
different FPS as a new shader optimization or attribute it to one frequency
change without a clock trace. The relevant comparison is the fresh CPU/GPU
pairing above under identical background-worker counts.

## Correctness and safety

Both portable benchmarks accept `VIDEO_REPEATS=1..20`, default 1. They retain
all results until after timing, require every repeated copy of an input to
have identical output, and write only one output per distinct input. All
1,000-copy checks pass in every matching run. All **400 downloaded maps**
from the eight CPU/GPU, idle/loaded, clip configurations equal their prior
validated outputs byte for byte: [agreement](agreement.log). For the repeated
loaded temple GPU configuration, the downloaded files are from the final run.
Earlier measured quality therefore remains applicable; this is not exact
OpenCV arithmetic or a broad accuracy benchmark.

GPU timing includes uploads, full matching/filtering, synchronization,
readback, pool scheduling and guards. CPU timing includes Mat creation,
native SGBM, f32 conversion, binary copying and guards. Input preparation,
three warmups per context/matcher, disk I/O and repeat-agreement checks are
excluded from both. C100 guards enforce temperature below 70°C and at least
2,000,000 KiB available memory; retained result binaries total about 174 MB
per 1,000-pair run. GPU concurrency remains two existing IREE contexts.

The CPU-load supervisor stops on callback success and failure. A live
negative control intentionally times out `/bin/sleep` through the same guard;
it stops the child and then all CPU workers:
[target timeout cleanup](load-timeout-board.log). The timeout is an injected
control, not a failed matching run. Host success/exception cleanup also passes:
[host load QA](load-host-qa.log). A host repeat-two run checks 100 results and
matches all 50 prior RADV outputs: [run](host-repeat.log),
[agreement](host-agreement.log). `VIDEO_REPEATS=0` is rejected before device
enumeration: [validation](invalid-repeat.log). The C100 package suite reports
four tests, zero failures, one hardware exclusion: [tests](c100-tests.log).
Native source/runtime/shaders are unchanged; the preceding 813-case core
regression result is linked in the [fusion report](../stereo-fused-20261022/README.md).

Every completed diagnostic reports **NEW_GPU_ERRORS=0**, retaining the 23
historical `nv50cal_space: -110` errors. That earlier native-submission issue
remains unresolved. Boot ID is `6f672cb9-e70d-433c-8894-7231f93bf6b7`,
validated slot B. CPU stays `schedutil`, max 1479000 kHz; GPU stays
`simple_ondemand`, min 76.8 MHz, cap 921.6 MHz; EMC remains 1.6 GHz. No
firmware, voltage, default library, clock-limit or boot-policy change occurs.
[Final state and runtime/model hashes](final-state.log) also confirm only the
original production BEAM remains. The temporary load module is unloaded and
the production process consumes zero CPU ticks in a final 500-ms idle check.
Temporary board output directories are removed after caching all predictions;
the reproduction tools remain: [cleanup](cleanup-board.log).

## Reproduction and remaining acceptance

The portable runners are `bench/stereo/census/throughput.exs` and
`bench/stereo/opencv/run.exs`. Set `VIDEO_REPEATS=20`; GPU also uses
`VIDEO_CONCURRENCY=2`, CPU `OPENCV_MODES=3way OPENCV_THREADS=4`.
Reuse the corrected model/runtime and input clips from the
[preceding staging instructions](../stereo-fused-20261022/README.md).

Board scripts are retained in `/root/nxiree/sgm-concurrency-tools`.
Load the CPU helper, then wrap the existing guard:

```elixir
Code.require_file("/root/nxiree/sgm-concurrency-tools/cpu_load_probe.exs")
C100GPU.CpuLoadProbe.run(2, fn ->
  Code.eval_file("/root/nxiree/sgm-final-tools/stereo_board_run.exs")
end)
```

The host controller and downloaded outputs are retained under
`~/.cache/stereo-bench/concurrency-20261022/`. The controller takes
`gpu|cpu normal 0|2 alley|temple`, sets/restores child environment and
propagates guard failure. It stops a test series at its first actual failure.

This verifies useful CPU offload and a larger throughput lead under contention.
It does **not** establish a 100-Hz camera pipeline: capture/synchronization,
rectification, conversion/resizing, paced input, long-run thermal/driver
stability and standard Evision GPU SGBM integration remain to be validated.
The board currently exposes no `/dev/video*` devices.
