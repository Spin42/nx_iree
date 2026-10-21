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
| `aot.exs` | compiles the Jetson test modules ahead of time |

## Jetson

`aot.exs` writes `jetson-modules.bin` for `NXIREE_VULKAN_TARGET` (default
`pascal`). `jetson/run.exs` loads those modules on the board with a
cross-built `libnx_iree.so`, and `jetson/board_run.exs` runs it in a separate
BEAM from the board's main node with the GPU clock pinned
(`NXIREE_PIN_HZ`, default 921.6 MHz).
