# Maxwell SGEMM research notes (2026-10-21)

Raw sources downloaded into ./dl/ (IREE KernelConfig.cpp/.h, NVIDIAConfig.cpp,
GPUReduceBankConflicts.cpp, MaterializeUserConfigs.cpp, MaterializeTuningSpecs.cpp,
KnownTargets.cpp, Passes.td; CLBlast xgemm_32.hpp; NAK sm50.rs, calc_instr_deps.rs,
opt_instr_sched_*.rs, assign_regs.rs, ir.rs; nvk_physical_device.c, nvk_private.h;
maxas SGEMM wiki markdown; Mesa MR JSON; CLBlast GTX750Ti sgemm plot PNG).

## maxas (Scott Gray) key numbers
- within 2% of theoretical; 98% on GM107, 96% on GM204; bound 512 FFMA / 520 issue slots.
- 64-thread: 64x64 tile; 256-thread: 128x128 tile; 8x8 per thread; C in R0-63,
  A frags R64-71/R80-87, B frags R72-79/R88-95; stay < 128 regs (4 warps/scheduler).
- shared: 8 k-lines x width x 4 B x 2 buffers x 2 matrices = 8 KB (64 thr) / 16 KB (256 thr).
- STS.128 stores (bank conflicts accepted), LDS.128 reads, zigzag mapping to avoid read conflicts.
- LDS/STS/TLD/BAR/BRA dual-issued with FFMA ("free"); 16 unavoidable bank conflicts hidden by .reuse.
- "70% of theoretical -> 98%" is attributed to bank choice + reuse ordering + exact mem-op placement (not possible with ptxas).

## Maxwell measured numbers
- cuBLAS 6.5 on GTX 750 Ti @1320 MHz: 1600 GFLOPS at N=4096 = 95% of 1670 peak (Gray, NVIDIA forum 34141).
- CLBlast chart GTX750Ti (CUDA 8.0): CLBlast ~950-1000 GFLOPS (sizes >= 1280, mult. of 128),
  ~820-840 near 2048; cuBLAS ~1340-1420; clBLAS ~580-780. Nominal peak 1306 (base clock).
- CLBlast DB (xgemm_32.hpp) params order: GEMMK,KREG,KWG,KWI,MDIMA,MDIMC,MWG,NDIMB,NDIMC,NWG,SA,SB,STRM,STRN,VWM,VWN
  - GTX 750 Ti: 0,1,16,2,16,16,128,32,8,64,1,1,0,1,8,2  -> 128x64 tile, K16, 16x8 thr, 8x8/thread
  - GTX 980:    0,1,16,2,16,16,64,16,8,128,1,1,1,0,4,8  -> 64x128, K16, 16x8 thr, 4x16/thread
  - GTX 970:    0,1,32,2,16,16,64,8,8,64,1,1,0,0,4,2    -> 64x64, K32, 16x8 thr, 4x8/thread
  - TITAN X:    0,1,16,2,8,16,128,8,8,128,1,1,1,1,4,8   -> 128x128, K16, 16x8 thr, 8x16/thread
- TX1: Shield TV matrixMulCUBLAS sample 153 GFLOPS (unknown clocks); clpeak X1 @921 MHz 220.8 GFLOPS;
  "NVIDIA Jetson Platform Characterization": TX1 GPU microbench 465 GFLOP/s (of 512).
- No measured cuBLAS SGEMM number for Jetson Nano found.

## IREE (main, Oct 2026)
- NVIDIAConfig.cpp: coop-matrix first; else workgroupXY={subgroupSize,8}, threadMNK f32={4,4,32}, promotion on.
- KernelConfig.cpp: K tile = bit_floor(residual) ... -> 32 for f32; pad = 128 bits; depth default 1/store 1;
  multibuffer bytes = single*(depth or depth+1); maxBytes = wgp maxWorkgroupMemoryBytes.
- Default result f32: tile [32,128,32], wg [32,8,1], 4x4/thread (test config_nvidia_matmul.mlir).
- Padding pass GPUReduceBankConflicts: innermost dim + 128 bits on every shared alloc.
- Pipelines: MatmulPromoteVectorize (multibuffer if depth>1 or store_stage==0; GPUPipelining epilogue peeling).
- User override: compilation_info attr; --iree-codegen-tuning-spec-path; --iree-codegen-transform-dialect-library=file@__kernel_config.
- KnownTargets: NVIDIA aliases pascal/sm_60-62 (48 KB smem), turing/sm_75 (64 KB), volta, ampere, ada, sm_120/121; no maxwell/sm_5x.
- NVK: maxComputeSharedMemorySize = min(48 KB, hw); subgroup 32.

## NAK (main, Oct 2026)
- sm50.rs instr_latency: 6/13 cycles for everything ("This is BS and we know it").
- ir.rs MIN_INSTR_DELAY = 1 -> no dual issue; add_reuse never called -> no .reuse flags; assign_regs no bank logic.
- IMad/IMul on sm<70 estimated 86 cycles (legacy IMUL, no XMAD lowering).
- MRs: !32311 greedy sched (25.1), !35821 Kepler latencies (25.2), !37108 cross-block (25.3), !33306 prepass (26.0);
  !42977 / !43389 (XMAD) / !43390 (reuse, dual-issue, bank spreading) closed unmerged (fork postadelmaga/mesa).
