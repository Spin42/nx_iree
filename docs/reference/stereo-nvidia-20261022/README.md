# Same Vulkan graph on Nerves/NVK and NVIDIA L4T

This comparison investigates the remaining efficiency gap. It does not assume
that the current implementation establishes a hardware ceiling. No CUDA is
used. The exported IREE VMFB contains the same Census SGM shader graph;
NVIDIA's Vulkan driver compiles its SPIR-V on L4T.

The [first optimization round](optimization-round.md) retains generic
branchless feature loads: 31.62 FPS VGA / 10.99 FPS 720p resident throughput,
and 31.04 FPS for a separate complete Elixir VGA loop. It records the
correctness checks, rejected experiments and remaining measurement gaps.

The [second round](optimization-round.md#second-round-instruction-and-layout-restructuring)
restructures the shaders and the volume layout and exceeds the NVIDIA target:
49.59 FPS VGA / 17.27 FPS 720p resident throughput and 47.11 FPS for the
complete four-context Elixir loop, byte-identical to the reference. Select it
with `VIDEO_FUSION=cost`.

## Matched Nerves comparison after returning from L4T

The user returned to Nerves from SD. Boot
`a1415b28-a332-4bfe-9652-7069b9fc2819` runs kernel6.18.57, validated slotB,
Mesa26.1.8/NVK GM20B, device VulkanAPI1.3.354. This verifies the SD boot
path with the installed eMMC boot configuration. The fresh kernel log has
zero GPU errors before and after the comparison.

Resident-input complete graph throughput, three repetitions, unchanged
Census/three-path/D64 graph and the same bounded IREE caching allocator:

| Case | Nerves/NVK FPS | L4T/NVIDIA FPS | NVIDIA / NVK |
|---|---:|---:|---:|
| 640x480, serial | **30.686** | 45.796 | 1.492x |
| 640x480, async4 | 30.746 | 45.931 | 1.494x |
| 1280x720, serial | **10.689** | 15.594 | 1.459x |
| 1280x720, async4 | 10.627 | 15.573 | 1.465x |

All12 staged binary/model/input checksum checks pass. All four numeric
output checks pass before any timing. Every sampled EMC rate is1600MHz;
GPU reaches921.6MHz during execution, returning to76.8MHz at idle. Both
systems retain their normal governors and the same CPU/GPU maximum limits.
CPU actual rates are not fixed or matched: the Nerves samples vary from
204 to1224MHz across these cases. Camera/network input, upload, output
readback and Elixir scheduling are excluded. VGA30 graph throughput is not
an end-to-end camera acceptance result. Nor does this short clean run resolve
the historical intermittent Nouveau load failure.

### Required runtime compatibility fixes

The frozen, unpatched IREE3.11 binary initially fails the Nerves VGA numeric
check at pixel516 (5.62308 versus5.60938). The captured full map has142,629
pixels beyond0.0001 tolerance and13,671 validity differences. The guard
stops immediately, before timing; the kernel reports zero GPU errors.

An isolated IREE3.11 build backports only the two generic host-cache fixes
already used by the production3.13 runtime: flush generated mapped input
buffers and invalidate noncoherent mapped sources before host transfer
reads. The identical four model/input cases then pass. The entire source
copy differs in only these two files. Models, shader binaries, allocator,
compiler flags and clock limits are unchanged; the frozen L4T artifacts
remain untouched. This is a required **runtime compatibility delta**, so
the two OS results must not be described as using identical runtime binaries.
The patches address initialization/readback outside the timed resident graph.

The durable Nerves bundle is
`~/.local/share/c100-gpu/baselines/nerves-sgm-20261022/`, with its manifest,
patched binaries, exact models/fixtures, two backports, source/build settings,
guard/controllers, raw results, frequency samples and profiles. Working
evidence is in `~/.cache/stereo-bench/nerves-compare-20261022/`.

### Where the time goes

A separate standard IREE3.13 dispatch-event capture, using the native
allocator and the modern envelope of the same shaders, passes both output
checks. At VGA it records355 valid dispatches/71 graph invocations, no
dropped/truncated records, and91.0% device-active time. Average GPU time per
graph is32.01ms:

| Stage | Average GPU ms |
|---|---:|
| Census | 4.25 |
| Horizontal aggregation | 12.71 |
| Fused vertical aggregation and selection | 13.33 |
| Projected consistency | 0.77 |
| Median filter | 0.96 |

Aggregation/selection account for **81.3% of active GPU time**. Average queue
gaps add3.17ms per graph; averages include cold outliers. Async4 also fails
to improve the matched3.11 throughput. These observations make aggregation
shader execution the main VGA target, while transfer/submission overhead
still matters for the complete Elixir pipeline. They do not by themselves
identify a particular NVK compiler optimization or hardware utilization limit.

The3.13 720p profile has only16 graphs,115.23ms average active GPU time,
48.48ms queue gaps and large outliers. It is an instrumented native-allocator
diagnostic, **not** the93.55ms cached3.11 control. An attempted3.13 caching
wrapper rejects `DEVICE_LOCAL|HOST_VISIBLE` / `MAPPING` allocation with both
the old bounded configuration and plain all-heap `caching`; both stop before
timing with zero kernel GPU errors. Removing the wrapper passes correctness.
Keep this allocator compatibility issue separate from shader execution;
do not enable that wrapper in the normal3.13 runtime on this evidence.

The measured driver/stack gap is about1.5x. Closing it alone would yield
about45.8FPS VGA or15.6FPS720p for this graph. **720p30 still needs about
1.92x beyond the measured NVIDIA result**, or2.81x over current NVK graph
throughput, plus a budget for input/output and camera processing. This is
not proof that720p30 is a hardware impossibility; it shows that batching
or host-gap removal alone cannot deliver it with this implementation.

## Earlier direct 720p Nerves result (different metric)

The unchanged classic48-bit Census/three-path/D64/f32-subpixel matcher at
1280x720 per eye processes50 unique Sintel pairs at **7.9937 whole-loop FPS**
with two contexts and two ordinary CPU workers. Warmup/input preparation/disk
I/O are excluded; upload, graph, readback, scheduling and guards are included.
Processing p50/p95 is220.43/335.77ms. No new GPU errors occur; the historical23
Nouveau errors remain. This is bounded diagnostic evidence, not stability
acceptance or live-camera throughput.

The GPU trace contains250valid dispatches with no dropped/truncated records.
Active dispatch time totals5.027seconds: **100.54ms per pair**. Visible device
span is6.124seconds,82.1%active, with1.097seconds of queue gaps. Horizontal
aggregation and vertical/selection each average roughly40ms per frame;
Census~14ms, consistency~2.25ms and median~2.87ms. Removing host gaps alone
cannot reduce100ms of GPU work to the33.3ms budget. Reaching30FPS requires
about3x faster GPU execution and3.75x current whole-loop throughput.

All50 downloaded maps agree numerically with the host discrete Vulkan GPU:
46,080,000pixels, maximum absolute difference0.000003815px, zero validity
mask differences (absolute tolerance0.0001, relative tolerancezero).

These are original1024x436 Sintel images resized directly to1280x720. They
exercise that pixel workload, but are upscaled/aspect-changed test data,
not native720p camera optics. D64 at1280width has half the disparity range
relative to image width as D64 at640width. D128 would be needed to preserve
that ratio and would add further matching work. No pyramid/ROI is used.

## Conditions needed for the NVIDIA comparison

The user manually booted L4T32.7.6, kernel4.9.337-tegra, glibc2.27, bootID
07e489bd-f9df-4447-b01c-dfff39c178c5. CPU governor is schedutil, cap1479MHz;
GPU nvhost_podgov,76.8–921.6MHz; nvpmodel reports MAXN. The NVIDIA device
reports Vulkan1.2.131, subgroup shuffle/timeline semaphores and mapped
external memory support.

**EMC remains at204MHz**, with zero recorded transitions to1600MHz. Nerves
ran its reference tests with EMC1600MHz. This known L4T configuration problem
confounds any direct driver-efficiency ratio until memory clocks are matched.
A slower L4T result must not be presented as the NVIDIA GPU's performance
ceiling. No clock, voltage, power-mode, firmware or boot-policy changes were
made during the initial204MHz measurements; the later boot fix is below.

The Nerves runtime binary requires glibc2.43 and cannot run unchanged on this
L4T userspace. Current IREE3.13 also requires Vulkan1.3; L4T's Vulkan1.2
loader rejects it before GPU dispatch. Standard IREE3.11 run/benchmark tools
are therefore cross-built against an isolated Ubuntu18.04 ARM64 sysroot.
The source pin is e4a3b0405d7d23554da26403658d0e8c3c5ecf25. The compiler
envelope uses the older executable-object attribute syntax; all five shader
SPIR-V blobs per resolution are byte-identical inside modern and legacy
VMFBs. No shader or runtime source was changed for the legacy build.
These tools measure GPU graph execution independently of Elixir/network
overhead. The later CLI control above includes two required host-cache
backports; the prior Elixir whole-loop FPS is a separate metric.

Device-specific load monitoring lives in dedicated c100_gpu's
scripts/l4t_vulkan_probe.sh. It rejects existing GPU errors, stops on the
first new kernel GPU error, enforces70C/1.5GB headroom and a maximum90second
child timeout, and records actualCPU/GPU/EMC frequencies. NonGPU success and
injected1second-timeout scenarios pass before running GPU workloads.

Model VMFB SHA256:
- VGA/D64:8af923b895ac80cdf807321e9276db36ce45f256be0d1560588cba9ae143d71d
- 720p/D64:7ed624a4cac9120b8fccb520bdf0c47b0ebffb979ff992bb4225f873c9bbae20

Legacy VMFB SHA256:
- VGA/D64:7ffc030a2de5e9a0012b9cc4667909f35a00cb3759552d0080b9e533eb485e24
- 720p/D64:175f3861fa9b0e2293954e0e03b880bfa4ff53228bb01d1ffda22ef1e38c9108

## NVIDIA measurements

Resident-input graph execution, three benchmark repetitions, D64:

| Allocation/submission | 640x480 FPS | 1280x720 FPS |
|---|---:|---:|
| Default allocator, synchronous | 9.020 | 3.047 |
| IREE caching allocator, synchronous | 17.022 | 5.677 |
| IREE caching allocator, four asynchronous invocations | 17.101 | 5.332 |

The cache is the existing IREE implementation, selected with
`--device_allocator=caching:device_local=256mib;512mib;64`.
Increasing the asynchronous 720p cache capacity to1GiB gives5.489FPS,
without a material improvement. No camera, network, Elixir scheduling or
output readback is included in these timings. CPU reaches1479MHz and GPU
921.6MHz during720p; EMC stays204MHz. Every guarded GPU run adds zero kernel
GPU errors. Default, cached and asynchronous correctness checks pass against
the first real Sintel reference frame at absolute tolerance0.0001.

Synchronous exports must use benchmark `--batch_size=1`: the generic legacy
benchmark invokes the function once per batch and assumes the batch count
is compiled into the function. Larger values would falsely inflate FPS.
The asynchronous controls instead use separately compiled coarse-fence ABI
exports and the standard async benchmark's actual four-invocation batches.

Raw JSON, frequency samples, binaries, input/output maps, captured MLIR and
SPIR-V are retained in
`~/.cache/stereo-bench/nvidia-compare-20261022/`.

## L4T EMC diagnosis, 2026-10-22

Read-only inspection on the same L4T boot confirms that MAXN already
requests1600MHz: `jetson_clocks --show` reports EMC min/max1600000000,
actual204000000. The bandwidth manager also requests1600000000 and records
zero frequency transitions. This is not an idle-frequency observation.

The active DRAM strap is0, manufacturer MR5=255 (Micron). Its selected
`emc-table@0/emc-table@1600000` has version
`13_1600000_12_V9.8.7_V1.6`, `nvidia,needs-training=0x2f0`, and
**`nvidia,trained=0`**. NVIDIA's local R32.7.6 source,
`drivers/memory/tegra/tegra210-emc.c:1580`, explicitly returns `-EINVAL`
for rates above204MHz when the selected table is untrained. The published
[driver source](https://github.com/OE4T/linux-tegra-4.9/blob/oe4t-patches-l4t-r32.7.6/drivers/memory/tegra/tegra210-emc.c)
contains the same condition. This identifies the immediate blocker.

The leading origin is C100's board-specific bootloader handling, not a RAM
speed limitation. [OKdo patch0005](https://github.com/LetsOKdo/c100-bootupd/blob/main/patches/0005-Patch-up-broken-emc-node-since-PCN211181.patch)
deletes the existing EMC node and injects a static PCN211181 overlay. Its
replacement high-speed table is untrained. Our earlier pre-U-Boot CBoot
capture in the system repository's `docs/l4t-reference/cboot/cboot-kernel.dts`
has the matching Micron table **trained=1** with nonzero trained values.
The current-boot live DT and driver gate confirm the blocker; a same-boot
before/after U-Boot capture would strengthen attribution of the data loss.

[NVIDIA describes the same204MHz symptom after replacing a bootloader-fixed
DT](https://forums.developer.nvidia.com/t/emc-clock-stuck-at-204-mhz/173460).
[OKdo documents why C100 needs a distinct bootloader and new-memory
overlay](https://github.com/LetsOKdo/c100-bootupd/blob/main/README.md).
Our extlinux entry already has no FDT line, so simply removing an FDT line
is not a fix here: OKdo's injected overlay still needs correction.

The remediation is to retain CBoot's correctly selected, trained EMC node
through the C100 boot path while keeping its necessary SDMMC3 support and
phandles intact. Preserve all training values; do not merely set trained=1
on static tables or bypass the driver's check. The read-only diagnosis was
followed by the boot-script fix and measurements below. The204MHz results
cannot establish NVIDIA's hardware ceiling.

## Installed C100 boot fix and full-speed L4T repeat

Dedicated C100 sources and rollback instructions are in sibling
`c100_gpu/docs/l4t-emc.md`. The file-only eMMC preboot hook retains SD-first
boot targets, stock SD boot command and the Nerves chain-loader. It bypasses
the destructive sysboot overlay handling only for the eMMC L4T entry, using
CBoot's freshly trained DT plus the necessary SDMMC3 overlay. No bootloader
binary, saved environment, SD file, CPU/GPU governor or voltage is changed.
The SD card was absent during the L4T tests. The subsequent user-initiated
Nerves SD boot and validated slotB are now verified above.

A one-shot test and unattended L4T boot reach **1600MHz actual EMC** with
training flag1. Unattended boot ID548701d7-ed83-4ae0-bee2-850587c8751b.
The same IREE3.11 binaries, models, inputs and allocator now measure:

| Resident-input graph | EMC204MHz | EMC1600MHz |
|---|---:|---:|
| VGA/D64 cached synchronous | 17.022 FPS | **45.796 FPS** |
| 720p/D64 cached synchronous | 5.677 FPS | **15.594 FPS** |
| VGA/D64 asynchronous4 | 17.101 FPS | 45.931 FPS |
| 720p/D64 asynchronous4 | 5.332 FPS | 15.573 FPS |

This is **2.69x/2.75x** the previous cached serial VGA/720p rates. Four
reference-output checks and four guarded timing probes pass with zero GPU
errors. Each timing case has three repetitions; all sampled EMC rates are
1600MHz and GPU reaches921.6MHz. Default EMC scaling works; no forced clock
override is applied. CPU/GPU governors and caps remain unchanged.

Raw evidence is retained under `~/.cache/stereo-bench/l4t-emc-20261022/`.
The results still exclude camera/network input, Elixir scheduling and output
readback. 720p30 is not reached by this unchanged graph. The subsequent
Nerves control and its runtime compatibility qualification are above.

The exact stack and a frozen comparison bundle are now saved in sibling
`c100_gpu/docs/l4t-sgm-baseline.md`, including native binaries, model/input
hashes, source snapshots and per-case argv. The durable bundle is
`/var/home/marc/.local/share/c100-gpu/baselines/l4t-sgm-20261022/`.
