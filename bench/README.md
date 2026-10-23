# Benchmarks

Host benchmarks run with `mix run <script>` from this directory. Vulkan
devices are picked by URI substring with `NXIREE_DEVICE_MATCH`, and extra
compiler flags (for example `--iree-vulkan-target=rdna3`) come from
`NXIREE_FLAGS`.

| Script | Measures |
| --- | --- |
| `bench.exs` | matmul throughput against EXLA on the CPU, plus an elementwise chain |
| `chain.exs` | the chain split into transfers, elementwise part and reduction |
| `latency.exs` | per-call latency of a tiny function (`OP=sum` for a reduction) |
| `sweep.exs` | reduction time against size, host and device-resident inputs |
| `pjrt.exs` | EXLA through IREE's Vulkan PJRT plugin (`../patches/iree-pjrt-exla.patch`) |
| `aot.exs` | compiles test modules ahead of time for an explicit Vulkan target |
| `real_models.exs` | pretrained BERT, DistilBERT, GPT-2, ResNet-50 and ViT servings against EXLA (`docs/real-models.md`) |
| `offline_cache.exs` | a serving run from the module cache in a VM without EXLA |

Vendor-neutral Vulkan workloads with their own READMEs: `yolo/` (YOLOv5
through IREE, with preprocessing, compile/run scripts and a per-family
tuning spec generator), `stereo/` (Census SGBM graphs and GLSL shaders) and
`vulkan/` (memory streaming). Device-specific tuning and board runners live
in the device's library, such as `../c100_gpu`.

## Vulkan i8 masks

IREE 3.13's SPIR-V backend misreads an i8 tensor that a batch matmul
dispatch consumes from memory when the Vulkan target has no 8-bit storage
(the default `vp_android_baseline_2022` profile), which makes BERT's masked
attention weights uniform. `vulkan_i8_mask_repro.mlir` is the standalone
StableHLO reproducer for `iree-compile`/`iree-run-module`. NxIREE.Compiler
avoids it by declaring
`--iree-vulkan-target-features=spirv:v1.3,cap:Shader,cap:StorageBuffer8BitAccess,cap:Int8,ext:SPV_KHR_8bit_storage`
unless the flags choose a `--iree-vulkan-target`; device targets such as
`pascal` and `rdna3` include 8-bit storage.

| Script | Checks |
| --- | --- |
| `bert_attn_print.exs` | BERT-tiny's first-layer attention weights on Vulkan against EXLA |
| `attn_bisect.exs` | the pieces of that graph one by one, ending with the i8 mask + batch matmul |
| `bert_trace.exs` | BERT-tiny with `--iree-flow-trace-dispatch-tensors` (`NXIREE_BACKEND=cpu` or `vulkan`) |
| `bert_dump.exs` | writes BERT-tiny's StableHLO and Vulkan bytecode to `OUT` for standalone runs |
| `sort_inplace_check.exs` | that sort, argsort and a fused sort leave their device input untouched |

## Device-specific benchmarks

`aot.exs` writes `vulkan-modules.bin` for `NXIREE_VULKAN_TARGET` (default
`generic`). The explicit `nvk` target obtains C100 capabilities from the
separate [`c100_gpu`](../../c100_gpu) library. Set `AOT_OUT` to choose the
output file.

The C100/Jetson runners and DistilBERT exporter now live in
`c100_gpu/bench/jetson/`; the current board harness lives in
`c100_gpu/scripts/stereo_board_run.exs`. The old clock-pinning harness is
retained there only as historical material.
