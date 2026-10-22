# Jetson Nano results

**Current baseline (2026-10-22):** normal GPU devfreq has been restored and
validated across 30 warm reboots. GPU initialization traps are failures,
not acceptable warnings. See [current Vulkan/stereo work](stereo-path.md).
The performance measurements below used the earlier pinned-clock setup.
Later checks in `../c100_gpu/docs/normal-dvfs.md` measured continuous 1024²
matmul at 29.7–35.9 ms with the extracted tuning, including readback, under
the normal governor. These end-to-end times differ from the kernel metric
below; 100 ms pauses also exposed idle-to-load ramp-up costs.

OKdo C100 (Jetson Nano module, GM20B Maxwell GPU), Nerves with Mesa NVK and
the system's NAK patches 0001-0012, GPU pinned at 921.6 MHz, EMC at
1600 MHz, IREE 3.13. Modules are compiled on the host with `NxIREE.export/3`
and run on the board with only Nx and nx_iree (scripts now in
`c100_gpu/bench/jetson/`). For reference, vkpeak measures 222 GFLOPS
fp32 on this GPU under NVK.

## Matmul, f32 (GFLOPS)

| Vulkan target | 256 | 512 | 1024 |
| --- | --- | --- | --- |
| generic | 29.8 | 33.8 | 36.5 |
| nvk capability list | 28.7 | 31.4 | 34.0 |
| pascal | 28.9 | 38.1 | 42.2 |
| nvk + Maxwell matmul tuning | 47.8 | 83.3 | 98.8 |
| nvk + 64x64x16 tiles, no multi-buffering | 27.9 | 73.5 | 83.7 |
| nvk + 128x64x8 tiles | 21.8 | 30.5 | 33.7 |

Every 256 result matches the CPU within 8e-5. The tuning spec gives the same
speed with the stock driver and with NAK patch 0013 (98.5 vs 98.8 GFLOPS),
so the gain is the tiling, not the scheduler change.

## DistilBERT SST-2, batch of 4, sequence 32

| Module | Labels vs EXLA | Max logit difference | Per batch |
| --- | --- | --- | --- |
| generic | identical | 3.0e-6 | 494 ms |
| nvk | identical | 3.0e-6 | 345 ms |
| nvk + Maxwell matmul tuning | identical | 3.0e-6 | 145 ms |

Copying the 268 MB of parameters to the GPU takes 0.56 s, once.

## Targets

`--iree-vulkan-target=pascal` cannot compile BERT- or GPT-style models in
IREE 3.13, so models use `C100GPU.Target.compiler_flags/0` from the dedicated
`c100_gpu` library. It compiles every model in `bench/real_models.exs` and, with the
tuning spec, matches or beats pascal's matmul speed.

## Historical board notes (superseded by the current baseline)

  * The GPU clock must not change under load: `c100-limits` (system commit
    08b906c) pins it while the GPU is idle. Dynamic devfreq wedged the GPU at
    boot.
  * With the clock pinned at 921.6 MHz and the EMC at 1600 MHz the board
    draws about 1 A at 5 V while idle, about twice L4T's idle.
  * A `gr: TRAP ... INJECTED_BUNDLE_ERROR` from the boot-time `vulkaninfo`
    appeared on one of seven boots; the channel and the boot completed.
