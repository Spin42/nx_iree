# Jetson matmul gap — main-thread findings (draft)

Measured on board (unpatched NAK, IREE presets, 921.6 MHz pinned), GFLOPS at 2048:
pascal 39.8, adreno 44.6, rdna3 50.7, valhall 24.5; vkpeak FFMA peak 222.7.

Kernels reproduced offline: tmp/measured/<t>.{bin,nak.log,sass} (unpatched NAK on GM20B drm-shim).

## Ruled out
- Per-call overhead: tiny dispatch 0.25-0.6 ms vs 340-700 ms per 2048 matmul.
- DRAM bandwidth (shared-memory configs): traffic 4N^3(1/TM+1/TN): pascal 1.3 GB (~67 ms at 20 GB/s) vs 432 ms;
  rdna3 1.1 GB (~54 ms) vs 339 ms. valhall 5.4 GB (~270 ms) vs 700 ms — significant share only for valhall.
- Register spills: none in any kernel.
- Instruction mix alone: FFMA is 77-83% of loop instructions in pascal/rdna3 (=> ~180-195 GFLOPS if latency hidden).

## Occupancy (SM53: 64K regs, 64 warps, 32 CTAs per SM)
| kernel | GPRs | warps/WG | WGs/SM | resident warps |
| pascal | 255 | 8 | 1 | 8 |
| rdna3 | 240 | 4 | 2 | 8 |
| adreno | 160 | 2 | 6 | 12 |
| valhall | 64 | 0.5 (16 threads) | 32 | 32 half-empty |

## Global-load exposure (instructions between LDG and first use)
- rdna3 main loop: 1-32 instrs, each LDG feeds an STS immediately, all four reuse R48 => serialized.
- pascal: next tile's LDGs issued at loop end; consumed by STS ~40-49 instrs after loop top, behind a BAR; 2 BAR/iter; 1 WG/SM => SM idles.
- adreno: 69-182 instrs (partial hiding, no shared memory, no barriers).
- valhall: 15-59 instrs.

## pascal per-iteration budget (k-tile 32, WG 32x128)
- measured: 262144 flop/iter at 43.2 flop/clk => ~6070 clk
- issue: 8 warps x ~720 instr / 4 schedulers ~ 1440 clk
- shared memory: thread tile 4x4 => 512 FFMA vs 128 LDS.128 per warp; A loads broadcast (~1 clk), B loads 4 clk => ~2560 clk/iter > issue => smem-bound at <=40% of peak even with perfect latency hiding
- remainder ~3000 clk: exposed LDG latency + barriers with a single resident WG
