# Why f32 matmul is slow on the Jetson Nano GPU

Measured on the board (GM20B, one SMM, 921.6 MHz pinned, EMC 1600 MHz), Nx →
nx_iree → IREE 3.13 SPIR-V → Mesa NVK with the firmware's NAK (Mesa patches
0001-0012), inputs on the device:

| IREE preset | 1024 | 2048 GFLOPS | share of vkpeak FFMA (222.7) |
| --- | --- | --- | --- |
| `rdna3` (64x64x16, wg 16x8, pipeline depth 2) | 50.4 | 50.7 | 23% |
| `adreno` (no shared memory, 32x128, 16x4 per thread) | 46.0 | 44.6 | 20% |
| `pascal` (32x128x32, wg 32x8, depth 1; IREE's NVIDIA default) | 38.8 | 39.8 | 18% |
| `valhall` (no shared memory, 8x32, 16-thread workgroups) | 25.1 | 24.5 | 11% |

The kernels the board ran were reproduced exactly by compiling the same
modules with the same NAK on the host against the nouveau drm-shim
(`NOUVEAU_CHIPSET=12b`), then disassembled and modelled. A cycle model of
one SMM driven by NAK's own scheduling annotations reproduces every measured
number within about 10% (pascal 40-44, rdna3 50-54, adreno 45, valhall 22-30)
and the measured 768 → 921.6 MHz scaling (model 1.12-1.18, measured 1.165).

## Ruled out

- **Per-call overhead**: 0.25-0.6 ms per dispatch against 340-700 ms per
  2048 matmul.
- **DRAM bandwidth for the shared-memory kernels**: tile traffic
  4·N³·(1/TM + 1/TN) is 1.1-1.3 GB per 2048 matmul, ~55-70 ms at 20 GB/s,
  against 340-430 ms measured. It contributes at most ~15% (pascal).
- **Spills**: none in any kernel.
- **The instruction mix alone**: FFMA is 73-78% of issue slots in the
  shared-memory kernels.

## Causes, by kernel

**rdna3 (best, 51 GFLOPS).** IREE emits the global→shared copy as
load/store pairs; NAK keeps that order and reuses one register quartet, so
the four DRAM loads of every k-tile are serialized: four full latencies per
iteration with only 8 warps (two workgroups at 240 GPRs) to cover them. The
schedulers idle ~42% of the time on these loads. Without them the same code
would reach ~94 GFLOPS.

**pascal (IREE's NVIDIA default, 40 GFLOPS).** 255 GPRs allow a single
workgroup (8 warps, 2 per scheduler). The next tile's loads are issued at
the end of the loop and consumed right after the barrier at the top, so the
whole SMM waits for DRAM each iteration. 55% of FFMAs depend on the FFMA just
before them (accumulator chains in source order), which with 2 warps per
scheduler caps FFMA throughput at ~42% of peak on its own. Its 4x4 thread tile
also bounds it by shared-memory bandwidth at 50% of peak (32 words/clk feed
128 FFMA/clk only when each thread holds 8x8).

**adreno (45 GFLOPS).** No shared memory and no software pipelining: 20
global loads at the top of each iteration, used ~70 instructions later, with
12 warps resident; schedulers idle ~69% waiting on memory.

**valhall (25 GFLOPS).** Workgroups of 16 threads fill half a warp, and B is
re-read from DRAM in 128-byte rows at an 8 KB stride (~4.3 GB per matmul at
an effective ~6 GB/s). Memory-throughput bound; nothing short of a different
kernel structure helps.

## Causes common to all kernels, ranked by cost

1. **Global-memory latency is not hidden.** Software pipelining is absent or
   ineffective (rdna3: serialized copy; pascal: loads consumed right after
   the barrier; adreno/valhall: none), and NAK does not hoist loads past
   other memory operations (it orders every memory instruction after the
   previous one).
2. **Too few warps.** 160-255 GPRs leave 8-12 of 64 warps resident, so no
   other workgroup runs while one waits. Bringing rdna3 or pascal to ≤128
   GPRs roughly doubles their throughput in the model (rdna3 ~100, pascal
   ~78).
3. **Thread tiles too small for Maxwell's shared memory.** IREE's NVIDIA
   default is 4x4 per thread (≤50% of peak by shared-memory bandwidth); rdna3
   is 8x4 (≤67%). Hand-tuned Maxwell SGEMM (maxas, cuBLAS, CLBlast's Maxwell
   tunings) uses 8x8 per thread at ≤128 GPRs.
4. **NAK keeps source order for FMAs.** Serial accumulator chains are not
   interleaved (pascal: +30% if fixed). NAK also has no dual issue and uses
   flat placeholder latencies on SM50.
5. **Smaller losses.** Address arithmetic recomputed each iteration and phi
   moves take 22-43% of non-FFMA slots; 26-31% of FFMAs keep a register-bank
   conflict (~3-4%); IREE's 4-float shared-row padding leaves some LDS bank
   conflicts.

## What the fix work in this repository addresses

`patches/mesa/0013` (order only conflicting memory operations) together with
IREE pipeline depth 1 through `bench/tuning/matmul_spec.py` targets causes 1
and 2: offline, a 64x64x16 tile drops from 240 to 128 GPRs and issues its
loads ahead of the FMAs. The model predicts ~107-118 GFLOPS for the best
configurations (64x64x8 and 64x64x16, depth 1). These are predictions,
not measurements; they still carry cause 4 and a 4x8 thread tile (cause 3).

Realistic ceiling for a portable kernel on this stack: CLBlast-class portable
kernels reach 65-75% of peak on Maxwell with NVIDIA's compiler; without dual
issue and real latency tables in NAK, 50-65% (118-155 GFLOPS) is the
plausible range. No published cuBLAS SGEMM figure for the Nano was found.

## Evidence

Analysis scripts and notes (cycle model `smsim.py`,
loop/bank-conflict analysis `loopstats.py`, NAK dependency analysis
`nakdeps.py`, prior-art notes with sources) are in `../dev/analysis/`;
reproducing the board kernels offline: `../dev/mesa-host-build-orig.sh` and `../dev/measured-sass.sh`.
