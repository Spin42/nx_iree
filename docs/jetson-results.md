# Jetson Nano results

OKdo C100 (Jetson Nano module, GM20B Maxwell GPU), Nerves with Mesa NVK and
the system's NAK patches 0001-0012, GPU pinned at 921.6 MHz, EMC at
1600 MHz, IREE 3.13. Modules are compiled on the host with `NxIREE.export/3`
and run on the board with only Nx and nx_iree (`bench/jetson/run.exs`,
`bench/jetson/run_model.exs`). For reference, vkpeak measures 222 GFLOPS
fp32 on this GPU under NVK.

## Matmul, f32 (GFLOPS)

| Vulkan target | 256 | 512 | 1024 |
| --- | --- | --- | --- |
| generic | 29.8 | 33.8 | 36.5 |
| nvk capability list | 28.7 | 31.4 | 34.0 |
| pascal | 28.9 | 38.1 | 42.2 |
| nvk + `iree_tuning: :maxwell_matmul` | 47.8 | 83.3 | 98.8 |
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
| nvk + `iree_tuning: :maxwell_matmul` | identical | 3.0e-6 | 145 ms |

Copying the 268 MB of parameters to the GPU takes 0.56 s, once.

## Targets

`--iree-vulkan-target=pascal` cannot compile BERT- or GPT-style models in
IREE 3.13, so models use the nvk capability list from the README's Vulkan
notes. It compiles every model in `bench/real_models.exs` and, with the
tuning spec, matches or beats pascal's matmul speed.

## Board notes

  * The GPU clock must not change under load: `c100-limits` (system commit
    08b906c) pins it while the GPU is idle. Dynamic devfreq wedged the GPU at
    boot.
  * With the clock pinned at 921.6 MHz and the EMC at 1600 MHz the board
    draws about 1 A at 5 V while idle, about twice L4T's idle.
  * A `gr: TRAP ... INJECTED_BUNDLE_ERROR` from the boot-time `vulkaninfo`
    appeared on one of seven boots; the channel and the boot completed.
