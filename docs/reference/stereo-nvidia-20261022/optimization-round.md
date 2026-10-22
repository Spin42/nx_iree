# First Nerves optimization round

The retained generic change makes right-feature loads and horizontal window
updates branchless, with clamped addresses and the original zero padding.
It preserves the classic 48-bit Census, D64, three paths, penalties,
uniqueness, f32 subpixel refinement, projected consistency and median filter.
No driver, firmware, governor or runtime default is changed.

## Measured improvement

Resident-input graph measurements use the comparison's coherent IREE 3.11
runtime, caching allocator, unchanged inputs and three benchmark repetitions.
These exclude upload, readback and Elixir. The frozen original comparison
bundles remain unchanged.

| Measurement | Control FPS | Branchless loads FPS |
|---|---:|---:|
| VGA, first run | 30.7587 | 31.7256 |
| VGA, reverse-order repeat | 30.6333 | 31.6247 |
| 720p | 10.6313 | 10.9931 |
| VGA, complete Elixir loop | 29.8512 | 31.0400 |

The Elixir comparison uses IREE 3.13, four independent contexts and 200
prepared pairs (50 distinct pairs repeated four times), without background
CPU load. It includes upload, graph execution, readback, synchronization,
scheduling and resource guards. Input preparation, disk I/O and three
warmups per context are excluded. Processing latency p50/p95 changes from
120.875/139.044 ms to 117.606/131.104 ms. This is throughput with overlapping
frames, not a 33 ms per-frame latency or camera acceptance result.

Every repeated output agrees. All 50 distinct output maps are byte-identical
to the saved reference, covering 15,360,000 pixels per comparison. Twenty
independent CPU-oracle cases also pass on the host discrete Vulkan GPU:
D64 and D128, BLOCK4/8/16, and symmetric eight-direction scans, including
partial workgroups and borders. All board probes report zero GPU errors on
boot `a1415b28-a332-4bfe-9652-7069b9fc2819`.

The remaining target is NVIDIA's saved 45.796 FPS VGA / 15.5942 FPS 720p.
Relative to the repeated candidate measurements, another approximately
44.8% / 41.9% throughput improvement is needed. This round does not establish
NVIDIA parity or a hardware ceiling.

The [second round](#second-round-instruction-and-layout-restructuring) below
exceeds that target: 49.59 FPS VGA / 17.27 FPS 720p for the same graph, with
byte-identical output.

## What the diagnostics establish

The branchless shader's final NAK horizontal code decreases from 921 to 769
static instructions and from 23 to four reconvergence setups. Fused vertical
code decreases from 522 to 515 instructions and from five to four setups.
Both remain at 48 GPRs; this is not an occupancy improvement. Static totals
include cold code and must not be interpreted as executed instruction counts.

The current installed driver reproduces the expected hardware controls:
222.47 GFLOPS FP32 scalar, 76.07 GIOPS integer scalar, and 10.19 GB/s
device-copy payload throughput using the existing vkpeak binary. The copy
test counts the copied payload once, although it reads and writes it. This
is not a matched full-speed NVIDIA bandwidth comparison: the saved NVIDIA
copy result was measured with EMC at only 204 MHz. The current Nerves EMC
remains at 1600 MHz, and GPU execution reaches 921.6 MHz.

A temporary IREE 3.11 allocation trace records only four native allocations
over the complete timed benchmark: three of 1,228,800 bytes and one of
46,694,400 bytes. All use memory type 0, properties `0xb` (device-local,
host-visible, host-cached and noncoherent); all frees occur after the timed
repetitions. Repeated native allocations are therefore not the observed
throughput limiter. Forcing supported memory type 1, properties `0x7`
(device-local, host-visible, coherent), preserves correctness but gives no
gain: 30.7726 versus 30.8502 FPS for the traced control. No allocation-policy
change is retained, and all host-cache maintenance remains intact.

Captured feature and path loads retain `MemOrder::Constant`, which selects
Maxwell's read-only `.ci` caching in the inspected NAK source. The simple
claim that immutable SSBO annotations were lost is not supported.

## Rejected interventions

These are short, correctness-checked diagnostics, not confidence intervals.
The earlier report contains additional rejected strategies.

| Intervention | FPS | Relevant control FPS |
|---|---:|---:|
| Balanced register mux for subpixel neighbours | 29.7437 | 30.7587 |
| Subgroup-private vertical tile | 27.7929 | 30.6120 |
| Horizontal-only BLOCK4 | 27.7098 | 30.8217 |
| Horizontal-only BLOCK16 | 30.3940 | 30.8217 |
| SM53 global-load latency estimate 128 | 30.7220 | 30.7336 |
| SM53 global-load latency estimate 256 | 30.7256 | 30.7336 |
| Coherent Vulkan memory type 1 | 30.7726 | 30.8502 |
| Horizontal packed DP with current NAK | 30.9660 | 31.6969 |

The subgroup tile adds approximately 29.49 MB of right-feature loads per VGA
frame. The two scheduling experiments load their intended private libraries
with disk shader caching disabled. Final NAK captures prove that instruction
schedules change in both dominant stages, despite the flat throughput. This
rejects those particular scheduling estimates, not compiler optimization
generally. Both temporary Mesa and IREE source edits are restored byte-for-byte
and rebuilt; the installed board libraries are unchanged.

## Open measurements and research

The IREE 3.11 Tracy stage capture is now available; see the matched-runtime
profiling section below. A measured NVK timestamp-period mismatch must be
accounted for when interpreting its GPU durations. NVIDIA stage timings,
register statistics and full-speed compute/copy controls are now available
below. Native NVIDIA instruction listings and matched shader memory-service
probes remain missing.

Three research agents investigated native hot-loop lowering and register
pressure, memory/synchronization boundaries, and reusable upstream SGM
strategies. Their main findings are:

1. **Packed DP arithmetic is a distinct representation change.**
   [Hernandez-Juarez's GPU SGM paper](https://ddd.uab.cat/pub/artpub/2016/288061/iccs-2016.pdf)
   and its [upstream implementation](https://github.com/dhernandez0/sgm/blob/228f4353b20226c85897521601a436fff2d68059/cost_aggregation.h#L159-L164)
   pack four exact costs per integer. Our BLOCK8 mapping could retain eight
   disparities per lane in two words. For the current parameters, path
   costs are at most 112 and their minimum at most 48; all recurrence
   candidates stay below 128. This permits exact seven-bit byte minima,
   normalization without inter-byte borrow, and matching-cost addition
   without carry. A private independently implemented horizontal prototype
   passes the existing 20 host cases and lowers native register use from
   48 to 40, with static instruction count rising from 769 to 809. The board
   output check also passes, but throughput drops to 30.9660 versus 31.6969
   FPS: this standalone packed version is rejected and not retained. Efficient
   ternary-logic lowering is a distinct prerequisite to investigate before
   revisiting it. The GPL upstream implementation
   is research evidence, not copied into the Apache shader source.
2. **Current NAK does not expose all useful Maxwell integer instructions.**
   [MaxAs's MIT-licensed instruction definitions](https://github.com/NervanaSystems/maxas/blob/54eda7af086a46c9dae1688b691968235d560164/lib/MaxAs/MaxAsGrammar.pm#L313-L332)
   encode native IADD3 and LOP3. NVIDIA's
   [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3)
   independently documents SM50 LOP3 support. The inspected NAK SM50 backend
   lacks these encoders while enabling related lowering only at SM70.
   Narrow GPR support needs native encoding and behavioral tests; enabling
   SM70 carry/predicate semantics wholesale would be incorrect. This is a
   possible compiler improvement, not an implemented fix or measured gain.
3. **Bandwidth alone is an incomplete explanation.** The path volumes require
   approximately 256 logical bytes/pixel for writes plus subsequent reads,
   excluding Census and postprocessing. Their 78.64 MB VGA traffic would
   occupy about 3.86 ms at the copy test's 20.38 GB/s combined traffic rate.
   This calculation does not equate copy-engine and shader memory service.
   It motivates matched shader read/gather/latency tests, followed by live
   MC arbitration comparisons only if a service difference is measured.
4. **No verified faster drop-in Vulkan SGM replacement was found.**
   [An OpenCL SGM port](https://github.com/siposcsaba89/stereo-sgm-opencl/blob/8cf8b251dbe2d4ebdc37faa20e7457205d03af63/src/ocl/path_aggregation_common.cl#L37-L40)
   uses shared memory and repeated barriers where our port already uses
   subgroup communication. [clspv](https://github.com/google/clspv) and
   [clvk](https://github.com/kpet/clvk) provide a legitimate Vulkan reuse
   route but no demonstrated SGM speed advantage. The actual Vulkan shaders
   in [PS5CameraDriver](https://github.com/KROU4/PS5CameraDriver/blob/33a7f2fd4dda177a95c049c9238575511851dc9a/src/core/shaders/aggregate.hlsl)
   use different matching/penalty rules and shared/atomic reductions, so they
   are not an equivalent faster replacement on present evidence.

Full-period register-window rotation is another smaller lead: the native
horizontal loop contains about 19 register moves per pixel. Unlike the
rejected unroll-by-two experiment, a full eight-step role rotation would
need to prove those moves disappear without increasing register pressure.
None of these research estimates establishes NVIDIA parity.

## Maxwell LOP3 follow-up

An isolated compiler experiment now implements SM50 general-register LOP3
encoding and enables the existing ternary-logic fusion pass for that register
class. Predicate lowering is unchanged. NVIDIA's SM53 disassembler recognizes
the emitted instructions. A separate Vulkan probe checks all 256 Boolean
truth tables plus constant-mask expressions over 16,640 output words against
a CPU oracle; it passes on both the host GPU and the board. This is targeted
validation, not a complete compiler conformance suite.

The initial version introduces extra constant loads because LOP3 lacks the
full-width immediate supported by LOP32I. A second version preserves the
existing two-input AND/OR/XOR encoding after ternary fusion. For packed
horizontal aggregation, static instructions decrease from 809 to 781, with
40 GPRs unchanged. However, neither version beats the installed-driver,
unpacked branchless control in the guarded whole-graph benchmark:

| VGA resident graph | First LOP3 build (R13) | Preserve two-input forms (R14) |
|---|---:|---:|
| Installed driver, branchless shader | 31.8859 FPS | 31.7220 FPS |
| Private driver, branchless shader | 31.5669 FPS | 31.5668 FPS |
| Installed driver, packed horizontal shader | 31.0809 FPS | 31.0417 FPS |
| Private driver, packed horizontal shader | 31.6373 FPS | 31.4187 FPS |

All eight SGM output checks pass; all probes finish with zero GPU errors.
Runtime process maps identify the selected driver, shader disk caching is
disabled, and the frozen IREE 3.11 comparison model/runtime is reused.
These short measurements reject LOP3 alone as the missing parity fix; they
do not establish statistical differences between nearby rates. No compiler
or packed-shader change is retained in production. IADD3 remains untested.

The native packed minimum still uses masking, shifting and subtraction to
expand per-byte comparison bits. Efficient byte-permute lowering is a
separate unresolved opportunity, not a gain established by this experiment.
The matched-runtime stage profile below now identifies where further
compiler changes must save time to close the NVIDIA gap.

Sources, generators, the private compiler patch, truth-table probe, native
disassembly and full results are archived in
`~/.cache/stereo-bench/nvidia-parity-20261022/lop3/`. Original host and cross
compiler sources and built libraries are restored byte-for-byte. The final
board state retains the same validated boot, driver hash and normal clock
policies, with no benchmark children or scoped environment overrides left.

## Matched-runtime profiling

The private IREE 3.11 tracing build now captures the unchanged original VGA
graph and the retained branchless VGA/720p graphs. Tracing required a guard
against absent external-shader debug metadata: AddressSanitizer reproduced
a 40-byte heap overwrite before the guard and passed the same numeric
workload after it. The host Tracy capture tool also required the upstream
32-to-64-byte buffer fix in `ProcessZoneValue`. Neither diagnostic fix was
installed as a default runtime; original source files were restored exactly.

NVK reports `timestampPeriod=1`, but an independent Vulkan query probe on
this boot measures approximately **1.6276 ns per tick**. In one 500 ms sample,
307,531,808 GPU ticks correspond to a bracketed CPU interval of
500,255,347–500,785,669 ns. Two complete probe runs agree. Tracy already applies
the reported period, so this is not merely an exporter-unit issue. Raw trace
durations understate elapsed time. The table uses the measured-compatible
estimate `1e9 / 614400000`; these are approximate calibrated durations, not
a driver fix or an FPS improvement. Do not apply this correction to NVIDIA
without measuring its timer separately.

| Warm GPU stage, mean milliseconds | Original VGA | Branchless VGA | Branchless 720p |
|---|---:|---:|---:|
| Census | 3.685 | 3.682 | 10.200 |
| Horizontal aggregation | 12.412 | 11.508 | 34.484 |
| Fused vertical aggregation and selection | 13.345 | 13.364 | 40.330 |
| Projected consistency | 0.770 | 0.769 | 2.278 |
| Median | 0.958 | 0.959 | 2.877 |
| Whole GPU command buffer | 31.174 | 30.287 | 90.172 |
| Interval between graph starts | 32.169 | 31.143 | 91.314 |

Each graph contains five chronological dispatch zones. The analysis verifies
five dispatches inside each enclosing command-buffer zone and excludes the
first five graphs: 68, 71 and 24 warmed graphs respectively. Internal gaps
total approximately 0.004 ms per graph. The remaining roughly 0.9–1.1 ms
between graphs cannot explain the parity gap. Aggregation and selection
consume about 82% of GPU time; the retained branchless change saves almost
all of its time in the horizontal stage. Stage durations include memory
stalls and do not by themselves distinguish ALU from memory service limits.

Wall-clock throughput remains the acceptance metric. The captured branchless
runs give 32.1272 FPS VGA and 10.9532 FPS720p. A separate VGA timing control
gives 31.5882 FPS for the normal runtime and 32.0963 FPS with tracing compiled
but GPU tracing disabled. These short runs show a small build/instrumentation
effect; the 32.1 result is not a retained production gain. All profiling output
checks pass, and every successful board diagnostic adds zero GPU errors.

The same per-stage trace under L4T is now complete, with original and
branchless models, runtime overhead controls and independent timestamp
calibration. The user performed both OS switches. No firmware, default
driver/runtime, clock-policy change, commit or push was performed.

Raw traces, timer repro sources, logs, CSV analysis, build settings and the
L4T kit are preserved with checksums at
`~/.local/share/c100-gpu/experiments/stereo-tracing-20261022/`.
Build trees remain at
`~/.cache/stereo-bench/nvidia-parity-20261022/tracy/`.

### NVIDIA stage comparison

L4T boot `1c6ac9d3-0b5c-41b8-b240-68a4a1511e55` passes twelve numeric checks,
twelve timing controls, three Tracy captures, three vkpeak controls and six
standalone shader-statistics probes. All report zero GPU errors. Every
frequency sample records EMC1600MHz; timing probes reach GPU921.6MHz under
the normal governors. NVIDIA reports `timestampPeriod=1.62760413`, agreeing
with the independent timer probe. Its exported Tracy durations are already
nanoseconds and receive no additional scaling.

| Warm branchless VGA stage, median ms | NVK | NVIDIA |
|---|---:|---:|
| Census | 3.6824 | 2.2493 |
| Horizontal aggregation | 11.4984 | 7.3255 |
| Vertical aggregation and selection | 13.3594 | 10.5120 |
| Consistency | 0.7692 | 0.3656 |
| Median | 0.9600 | 0.4975 |

The largest absolute gaps are horizontal aggregation (~4.17ms), vertical
aggregation/selection (~2.85ms) and Census (~1.43ms). NVIDIA's branchless720p
stage medians are6.2438/21.8286/30.5951/1.1747/1.4862ms in the same order.
Its trace has occasional inter-dispatch pauses, so medians are preferable
for the shader comparison; wall-clock throughput remains authoritative.

The uninstrumented original NVIDIA graph repeats at44.7121FPS VGA and
15.4526FPS720p, compared with the frozen45.796/15.5942 baseline. The coherent
compatibility runtime gives44.6421/15.4180: its cache-maintenance fixes do
not explain the NVK deficit. NVIDIA's branchless graph gives44.6473/15.4773,
essentially unchanged from its original shader. Instrumentation lowers VGA
throughput to42.7996FPS with capture and43.3696 with GPU tracing disabled;
do not compare these instrumented FPS directly against the frozen baseline.

Full-speed vkpeak reports NVIDIA227.75GFLOPS,76.75GIOPS and9.96GB/s copy
payload throughput, versus NVK222.47/76.07/10.19. Both tools measure host
elapsed time, so the timestamp-period bug does not rescale these results.
These are separate builds of the existing vkpeak implementation, not a
binary-identical microbenchmark. They weaken a gross compute or DRAM-capacity
explanation; they do not establish equivalent shader cache/latency behavior.

NVIDIA's advertised `VK_KHR_pipeline_executable_properties` exposes register
statistics. A private helper reuses the Apache-2.0
[NVIDIA nvvk pipeline utilities](https://github.com/nvpro-samples/nvpro_core/blob/bc19d6ac3ef62938d0ea0e099735878457ce1b6e/nvvk/pipeline_vk.cpp)
to compile the same SPIR-V independently and query its statistics. This is
standalone compilation, not instrumentation of the timed IREE pipeline.

| Shader | NVIDIA GPRs | Captured NVK GPRs |
|---|---:|---:|
| Census | 32 | 24 |
| Horizontal | 40 | 48 |
| Vertical/selection | 40 | 48 |
| Consistency | 11 | 24 |
| Median | 24 | 24 |

NVIDIA reports zero local memory for all five stages. The first internal
representation request produces no files, so no native NVIDIA instruction
listing is claimed. Different register counts are a lead, not a diagnosis.

### Register-pressure follow-up: no retained gain

After returning to clean Nerves boot
`2e0cb4b9-6f3d-452b-b2d9-be6ee28dcd6d`, two private variants test that lead:

* Explicit returns after directional switch cases remove two workgroup/path
  values from the horizontal loop's live set. Compilation drops48→40GPRs,
  with771 versus769 instructions and no spills. Twenty host oracle cases
  and board output checks pass. R26 gives31.7195FPS versus31.8595 control.
* Masking invalid aggregate costs once and using a relative winner shortens
  vertical selector lifetimes. Compilation drops48→40GPRs and515→500
  instructions, removing predicate-register spills. Twenty host oracle
  cases and board output checks pass. R27 gives30.9011FPS versus31.5590 control.

Neither change is retained. Reducing the static occupancy restriction alone
does not produce a measured speedup in these tests. All probes report zero
GPU errors; original shader sources and default runtime/driver remain intact.
R28 captures from the board confirm that the actual IREE pipelines use the
reduced 40-register allocation; this is not a stale host compiler result.
Private sources, native captures and logs are in `early-return/` and
`masked-select/` under the optimization cache. Shader memory service and
instruction/dependency scheduling remain open explanations.

### Read-cache policy follow-up: existing policy helps

R29 changes only the SM53 encoder's constant global load policy from `.ci`
to `.cg` in a private driver. Native comparison of all five shaders confirms
identical instruction words except the two cache-policy bits on those loads;
register allocation and scheduling are unchanged. Both output checks pass,
but throughput falls from31.769 to25.6152FPS. The installed driver is unchanged.

R31 profiles use the independently measured timestamp correction above:

| Stage | Existing `.ci`, ms | Private `.cg`, ms |
|---|---:|---:|
| Census | 3.6826 | 3.7697 |
| Horizontal | 11.4405 | 11.7587 |
| Vertical/selection | 13.3684 | 13.4984 |
| Consistency | 0.7694 | 0.7697 |
| Median | 0.9555 | 7.7433 |

These are warmed means from73 control and58 candidate graphs. The current
read-only cache policy strongly benefits median filtering. Changing it does
little for the dominant aggregation stages and is rejected. This does not
rule out other memory latency or scheduling differences between drivers.
R30 initially captured the correctness process instead of the timing process;
the stage analyzer rejected it, and no timing conclusion uses that capture.
All board probes report zero GPU errors. Host/cross source files and build
libraries were restored after extracting the private drivers.

Sources, encoded-word verification, traces, CSVs, process maps, frequencies
and logs are in `cache-policy/` under the optimization cache and the durable
`~/.local/share/c100-gpu/experiments/stereo-tracing-20261022/` archive.

### Final instruction-ordering follow-up: bypass is slower

R32 privately skips only `opt_instr_sched_postpass` for SM53. The prepass,
register allocation and final hardware dependency calculation remain enabled.
All five shaders preserve their non-NOP instruction/operand multisets, GPRs
and spill counts, while their ordering changes. The scheduler-derived static
cycle estimate is zero in this diagnostic build and is not used as evidence.

Correctness passes, but candidate throughput is28.0281FPS versus31.5139FPS
control. R33 corrected warmed stage means confirm both major stages lose:

| Stage | Existing scheduler, ms | Postpass bypass, ms |
|---|---:|---:|
| Census | 3.6826 | 3.7382 |
| Horizontal | 11.4364 | 12.9960 |
| Vertical/selection | 13.3882 | 15.9499 |
| Consistency | 0.7694 | 0.7975 |
| Median | 0.9582 | 1.0587 |

All probes report zero GPU errors. The bypass is rejected; it demonstrates
that the current postpass improves these shaders, without proving its
scheduling is optimal. Source files and build libraries are restored;
the installed driver and runtime are unchanged. Full evidence is retained
in `postpass/` in the same cache and durable archive. NVIDIA parity remains
unachieved; further work needs a more specific dependency/throughput model
or a new algorithmic improvement that preserves output quality.

## Evidence and reproduction

### Packed matching-cost reuse: small experimental gain

An independent packed byte cost volume preserves the existing Census values,
zero-feature border padding, all disparities entering the recurrence, DP,
selection and postprocessing. It avoids computing each matching cost in all
three directions. The existing packing and vertical shared-tile layout are
reused; no 8-bit storage extension is required.

The first variant produces eight costs per thread with branchless feature
loads. R35 passes output checks but gives27.8407FPS versus31.5432 control.
R36 locates the problem: cost production10.1758ms, horizontal8.5906ms,
vertical/selection10.5580ms. Saving aggregation work does not cover the extra
producer. Earlier archived one/four-cost-per-thread prototypes also lost;
this is a revised producer, not a previously unexplored strategy.

The second variant reuses the existing shared right-feature tile in the
producer. R38 reduces production to4.9851ms; horizontal8.6019ms and
vertical/selection10.5567ms are unchanged. Total GPU command-buffer time is
29.5546ms versus30.1451ms for R34's original graph. Paired uninstrumented runs:

| Run order | Control FPS | Tiled cost FPS |
|---|---:|---:|
| R37: control, candidate | 31.5897 | 32.1331 |
| R39: candidate, control | 31.6837 | 32.3015 |

Both variants pass20 host CPU-oracle cases (D64/D128, BLOCK4/8/16,
eight directions, partial workgroups and borders), including every intermediate
path and final map, then pass board output checks. Missing intermediate cost
oracles are derived from the existing CPU Census feature oracles; that helper
matches the previously saved D64 cost oracle exactly. The first host test
caught a missing GLSL `out` qualifier, which was fixed before board testing.

The repeatable gain is about1.7–2.0%, with an extra19.66MB cost buffer at VGA.
The candidate remains private pending a larger improvement; the public shader
and exporter are unchanged. A possible follow-up is producing costs inside
the first horizontal scan, avoiding the separate producer, but that changes
dispatch structure and has not been implemented or measured. NVIDIA parity
remains unachieved. R35/R37/R39 all report zero new kernel messages; GPU
profiles use `TRACY_NO_SYS_TRACE=1`. Artifacts are in `cost-volume/` and
`cost-tiled/` under the optimization cache and durable archive.

### Fusing the cost producer into a path: R40–R44

The follow-up proposed above is now implemented privately as `cost-fused`.
The forward horizontal scan writes its existing path and packed raw matching
costs. A separate reverse scan consumes those costs; vertical aggregation and
selection consume costs plus the two horizontal paths. This removes the
separate tiled cost producer while preserving recurrence, invalid-disparity
padding, selection, consistency and median. The exporter explicitly guards
the three-path vertical-fusion case. It adds a 19.66 MB cost buffer at VGA.

| Run | Resolution | Paired control FPS | Candidate FPS |
| --- | --- | ---: | ---: |
| R40 | 640×480 | 31.6037 (retained loads) | 34.2377 |
| R42 | 640×480 | 31.6214 (retained loads) | 34.2054 |
| R43 | 1280×720 | 11.0302 (retained loads) | 11.3299 |

The repeated VGA gain is about8.2–8.3%; 720p gains only2.7% in this paired
sample. Neither achieves NVIDIA's frozen45.796/15.5942 FPS. These are
resident-input graph measurements, not a camera/network/Elixir throughput
claim. The full200-pair Elixir comparison has not yet been run for this
candidate, and no production shader/exporter change has been made.

R41 quiet GPU profiling (76 warmed graphs, corrected NVK timestamp units)
gives means3.6827 ms Census,7.5997 forward+cost,4.3087 reverse,
10.6208 vertical+selection,0.7692 consistency and0.9611 median;
command-buffer27.9470 ms and internal gaps0.0048 ms. Profile frame interval
29.7820 ms includes profiling/monitoring overhead and is not used for the FPS
claim. `TRACY_NO_SYS_TRACE=1` disables CPU sampling.

A distinct `cost-vertical` candidate then produced costs and the vertical path
from the existing shared Census tile, ran both horizontal paths on the cost
volume, and used the modern branchless selector as a pixel-parallel dispatch.
It adds a vertical volume with19.66 MB capacity and39.32 MB write/read traffic
at VGA. R44 measured **31.8376 FPS versus34.1242 for cost-fused**: reject it.
No separate stage profile or720p measurement was run for this slower variant.

Both private variants passed16 independent host-oracle cases acrossD64/D128
and BLOCK4/8/16, four frames each, including intermediate paths/costs and final
maps. All paired board output checks passed. R40/R42/R43/R44 each report
zero new kernel messages and zero GPU errors. The user then requested stopping
and writing the handover; R44 had finished, and no further experiment started.

Sources, reproducible scripts, models, traces and results are in cache
`cost-fused/` and `cost-vertical/`, copied to the durable experiment archive
`~/.local/share/c100-gpu/experiments/stereo-tracing-20261022/` with checksums.
The board remains on Nerves boot`2e0cb4b9-6f3d-452b-b2d9-be6ee28dcd6d`,
validated slot B, EMC1600MHz, normal governors and unchanged installed NVK.
Final-state check confirms restored scoped environment, one BEAM and no GPU
errors. This is a performance checkpoint, not a new boot-stability claim.
See `HANDOVER-board-testing.md` in the base-system worktree for resume steps.

### CPU profiling interrupts, identified after user report

The user noticed three `perf: interrupt took too long` messages that the
GPU-error guard did not flag. They first appear between R30's before/after
kernel snapshots at583.341,583.548 and583.678seconds after boot. Reported
average CPU sampling-handler durations are2516,3147 and3975nanoseconds.
Linux lowers `kernel.perf_event_max_sample_rate` to79250,63500 then50250;
this is a profiling-rate limit, not a CPU/GPU clock limit. The compiled
Tracy client enables perf-based system sampling by default, consistent with
these messages appearing during the first capture. No perf-event handles
remain open after the process exits.

R34 repeats the baseline GPU trace with the upstream-supported process-only
`TRACY_NO_SYS_TRACE=1`. All12 monitored samples show zero perf-event handles,
the before/after kernel logs are identical, and the five GPU stages remain
available. Corrected means are3.6816/11.3755/13.3577/0.7691/0.9571ms,
consistent with earlier stage measurements. Future traces use this setting;
the kernel sysctl is left untouched. Ordinary FPS comparisons already use
the uninstrumented runtime. Earlier traces include system-profiling overhead
and must not be presented as production throughput. The private experiment
runner now prints every new kernel message as well as GPU-error counts.
Evidence is in `irq-review/` under the optimization cache and durable archive.

Raw commands, sources, models, frequency samples and outputs are under
`~/.cache/stereo-bench/nvidia-parity-20261022/`. Relevant runs are `r1` through
`r12`; `r9-loads.log` contains the repeated resident measurements and
`r10-elixir.log` contains the complete Elixir comparison. `compiler-summary.log`
records native-code hashes, instruction counts and GPR counts. The `driver/`
subdirectory retains full NIR/NAK captures; `allocator/` retains the exact
temporary tracing/type-selection source and build logs.

Copies of the retained candidate's timing logs, host-oracle logs, compiler
summary and final board state are in [optimization-evidence/](optimization-evidence/).
The final state verifies the unchanged installed NVK hash, restored scoped
environment, one main BEAM, validated slot B and zero kernel GPU errors.
Research source clones are archived under the cache's `upstream/` directory.

`check-loads.sh` reproduces the existing host oracle matrix using the private
tested shader snapshot. `build-variants.sh` / `build-720.sh` reproduce the
legacy comparison models. Board-only controls remain in the dedicated
evidence directory and are not added to the generic Vulkan library API.

## Second round: instruction and layout restructuring

This round reaches NVIDIA parity. It starts from the private `cost-fused`
graph (34.24 / 11.33 FPS) and changes only shader source, the exporter's
buffer layout and the intermediate-oracle comparison. Output stays
byte-identical: every variant passed the 16 host-oracle cases (D64/D128,
BLOCK4/8/16, four frames each, including intermediate volumes), the board's
frozen-reference numeric check at both resolutions, and the complete 200-pair
Elixir comparison against the saved 50 reference maps. No driver, firmware,
runtime, governor, clock or voltage changed; every board probe reports zero
GPU errors and zero new kernel messages on boot
`2e0cb4b9-6f3d-452b-b2d9-be6ee28dcd6d`.

### Result

Resident-input graph throughput, coherent IREE 3.11 runtime, caching
allocator, three repetitions, paired in one guarded run per row:

| Run | 640x480 FPS | 1280x720 FPS |
|---|---:|---:|
| Frozen NVIDIA L4T target | 45.796 | 15.594 |
| `cost-fused` control (R45/R46, R63/R64) | 34.02, 34.19 | 11.39, 11.39 |
| `lean`: restructured shaders, row-major volumes (R45, R47, R46) | 41.83, 41.78 | 13.56 |
| `lean3`: + row-tiled byte volumes (R57, R58) | 47.28 | 16.93 |
| `lean4`: + interleaved tiled features (R59, R63, R60, R64) | 48.80, 48.29 | 16.83, 16.87 |
| `lean7`: + four rows per Census thread (R68, R69) | **49.59** | **17.27** |

The complete Elixir loop (IREE 3.13, four contexts, 200 prepared pairs, idle
CPU) gives 47.11 FPS for `lean7` versus 30.95 FPS for the retained branchless
control in the same session (R70), with processing latency p50/p95 of
73.8/102.9 ms versus 117.8/130.4 ms. All 50 distinct maps of both variants
match the saved reference byte-for-byte.

### What changed

Native code was inspected with the existing host NAK shim before every board
run. The old hot loops were issue-bound on instruction count: 748 instructions
per Census pixel, about 189 per horizontal pixel step and 379 per vertical
step. The restructured shaders compile to 272 per Census pixel, 148 per forward
step (NAK unrolls the loop twice: 295 per two steps), 97 per reverse step and
241 per vertical step, at 24/48/24/48 GPRs with no spills.

1. **Census** (`census_pair.comp`): each shared word packs the left and right
   pixel in 16-bit lanes offset by 0x100. One subtraction per neighbour gives
   both comparison bits without inter-lane borrows; a mask, shift and OR place
   them for both images at once, and each 16-neighbour chunk is inverted once.
   Every thread produces one pixel per tile row for four tiles, with an odd
   shared-row stride so the sixteen rows of a warp hit distinct banks.
2. **Recurrence** (`cost_dp.glsl`): the candidates are formed once as
   `min(min(a[k], P2), min(a[k-1], a[k+1]) + P1) + cost`; a missing boundary
   neighbour contributes `P2 - P1`. Shuffle lane indices, row offsets and the
   loop condition are loop-invariant and strength-reduced, which NAK does not do
   by itself.
3. **Selection** (`cost_select.glsl`): the forward and reverse path words are
   added in packed form (bytes stay below 128), candidates beyond the image
   carry a 0x8000 padding term instead of per-candidate predicates, the
   second-best exclusion is one unsigned range test per candidate, and the
   subpixel neighbours come from a static mux tree instead of eight selects.
4. **Layout**: ablations (R53/R54) showed the forward scan was bound by its
   memory transactions, not by instructions or load latency: removing its two
   stores saved 3.35 ms and its feature loads 2.65 ms of 7.57 ms, while one-step
   and two-step register prefetch gave nothing at 48 GPRs and lost a wave at
   56. Each warp scans four rows 40 KB apart, so every access touched four
   64-byte half-lines. Storing the cost and path volumes in tiles of sixteen
   consecutive rows per column makes each warp access 256 contiguous bytes:
   forward 7.57 → 5.02 ms, reverse 4.33 → 2.67 ms. Interleaving both images'
   features in the same tiled layout then cuts the forward scan to 3.98 ms. The
   vertical stage pays 1.1 ms for reading four half-lines per warp; pairing its
   rows costs eight registers and a wave, and a one-row prefetch lost 3%
   (R62), so the layout is kept as is. Padding the row pitch by two pixels
   instead of tiling was slower (R51), so cache-set aliasing is not the
   mechanism.

Corrected GPU stage means (ms, NVK timestamp units calibrated as before):

| Stage | `cost-fused` (R41) | `lean` (R50) | `lean3` (R56) | `lean4` (R61) | NVIDIA (branchless) |
|---|---:|---:|---:|---:|---:|
| Census | 3.68 | 1.52 | 1.52 | 2.22 | 2.25 |
| Forward horizontal + cost | 7.60 | 7.57 | 5.02 | 3.98 | 7.33 (both directions) |
| Reverse horizontal | 4.31 | 4.33 | 2.67 | 2.66 | — |
| Vertical + selection | 10.62 | 7.42 | 8.49 | 8.51 | 10.51 |
| Consistency | 0.77 | 0.77 | 0.77 | 0.77 | 0.37 |
| Median | 0.96 | 0.96 | 0.96 | 0.96 | 0.50 |
| Command buffer | 27.95 | 22.56 | 19.43 | 19.11 | — |

At 720p `lean4` spends 7.02 / 12.35 / 7.93 / 25.74 / 2.28 / 2.86 ms in the same
order (58.18 ms per command buffer, R65). Consistency and median were not
changed; they remain the only stages slower than NVIDIA's.

### Rejected in this round

| Variant | Result |
|---|---|
| Row pitch padded by two pixels (R51) | 40.85 vs 41.91 FPS |
| Forward features prefetched one/two steps (56 GPRs, R53) | 42.20 / 41.18 vs 42.00 |
| Reverse costs prefetched one/two steps (R54) | 40.86 / 40.88 vs 41.76 |
| Vertical words prefetched one row, row-major (R54) | 40.20 vs 41.76 |
| Vertical words prefetched one row, tiled (40 GPRs, R62) | 46.91 vs 48.47 |
| Two vertical rows per iteration | 56 GPRs, not timed |
| Selector padding recomputed per step instead of held (R68/R69) | 48.26 vs 49.59 VGA, 17.34 vs 17.27 720p |

### Reproduction and evidence

Private sources, scripts, NAK listings, ablation shaders, board logs, Tracy
captures and per-variant snapshots are in
`~/.cache/stereo-bench/nvidia-parity-20261022/lean/`, with runs R45–R72 under
`/root/nxiree/nvidia-parity-20261022/` on the board, and a checksummed copy in
`~/.local/share/c100-gpu/experiments/stereo-tracing-20261022/lean/`. The
retained shaders are `bench/stereo/census/{census_pair.comp,cost_first.comp,
cost_scan.comp,cost_dp.glsl,cost_select.glsl}`, selected by
`VIDEO_FUSION=cost`; `check-prod.sh` reproduces the host oracle matrix through
the production exporter and `run-elixir.sh` the board comparison.

### 960x540 follow-up

The user then asked whether 960x540 reaches 30 FPS. With the retained `lean7`
graph and a central 960x540 crop of the frozen 720p input (host RADV output as
the reference, matched within 3.9e-6), the resident-input graph measures
29.16 / 29.70 FPS (R73, R75); the pad-free selector variant gives 29.29 FPS in
the same run (R75). Corrected stage means (R74): Census 3.01, forward 7.21,
reverse 5.00, vertical + selection 14.27, consistency 1.29, median 1.61 ms,
32.39 ms per command buffer and 34.21 ms per frame. Reaching 30 FPS on the
resident graph needs about 3%; the untouched consistency and median stages
(2.9 ms) are the obvious candidates.

The complete Elixir loop is much slower at this size: 9.85 FPS with one
context, 21.18 FPS with two and 14.57 FPS with four (R76, R77; nine distinct
crops, twenty repeats, the centre crop checked against the host reference).
The cause is per-frame transient allocation, not the shaders: the same resident
benchmark without IREE's caching allocator drops from 29.3 to 13.9 FPS
(72.0 ms per frame, 34.5 ms of CPU, R78); at VGA the same change costs
49.0 → 24.5 FPS (R80), which four contexts can still overlap. The IREE 3.13
Elixir runtime
allocates natively because its caching wrapper rejects mapped allocations.
Pooling or caching the transient volumes in that runtime is the next step for
the loop; the shader work above is complete.

