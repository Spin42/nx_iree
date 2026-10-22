# Vulkan recurrence and allocator evidence

These logs cover the portable SGM directional recurrence, shared IREE
allocator repair, and C100-specific library extraction. They are not a
complete stereo matcher or a comparison with NVIDIA SGBM.

IREE source and original compiler: `89ece87b89852100c0a4ac3d515e14842d090967`.
Host device: AMD Navi32, RADV, 16 GiB VRAM, PCI 03:00.0.
C100: NVK GM20B, Mesa 26.1.8, accepted firmware
`1ffb5cb3-94bd-54a7-f1e2-ec9661ce6746`, validated slot B.
No firmware upload or reboot was performed for these Vulkan tests.

| Evidence | Files |
| --- | --- |
| Independent scalar reference vs Nx evaluator and IREE CPU | `c100-stereo-host-check.log` |
| Original child-process OOM and post-failure board state | `c100-stereo-board.log`, `c100-stereo-failure-state.log` |
| Bounded small-shape board memory probe | `c100-stereo-memory.log` |
| Host reproduction despite GC and explicit release | `stereo-host-memory.log` |
| Stable live reservations but growing backing slabs | `stereo-host-memory-stats.log` |
| Same workload with TLSF fallback search | `stereo-host-memory-fixed.log` |
| Allocator regression before/after patch | `iree-pool-red.log`, `iree-pool-green.log` |
| Fresh and repeated source-patch application | `stereo-patch-apply.log` |
| Clean runtimes, seven fixtures and ten repeats per size | `stereo-host-clean.log`, `stereo-board-fixed.log` |
| Experimental topology annotation, exact results and unchanged inputs | `stereo-topology-host.log`, `stereo-topology-board.log` |
| Generic tuning provider tests | `nxiree-generic-tuning-test.log` |
| Extracted C100 tuning on host discrete GPU | `c100-gpu-discrete-test.log` |
| Dedicated C100 harness on real board | `c100-gpu-harness.log` |
| Full portable Vulkan suite | `stereo-nx-iree-vulkan-tests.log` |
| Full CPU suite after extraction, 798 passing cases | `stereo-nx-iree-cpu-tests.log` |
| Compiler directory selection unit and real Mix checks | `stereo-compiler-selection-test.log`, `stereo-compiler-dir-validation.log` |
| Standalone benchmark after library extraction | `stereo-bench-after-extraction.log` |
| C100 Buildroot package install and unchanged board script hash | `c100-gpu-package-build.log`, `c100-gpu-package-board-check.log` |
| Live NVK target capability confirmation | `c100-gpu-features.log` |
| Pad/slice fusion experiment, no established speed gain | `stereo-pad-fusion-host.log`, `stereo-pad-fusion-board.log` |
| Continuous baseline with the original compiler | `stereo-continuous-baseline-board.log` |
| Continuous host baseline after the compiler build finished | `stereo-continuous-baseline-host.log` |
| Original missed-elision regression and unsafe intermediate cases | `stereo-affinity-red.log`, `stereo-affinity-ownership-red.log`, `stereo-affinity-unknown-producer-red.log`, `stereo-affinity-zero-trip-red.log` |
| All four compiler pass suites, thirteen new cases | `stereo-affinity-green.log` |
| Normal exports with default and forced VMVX constant evaluation | `stereo-affinity-normal-export-smoke.log` |
| Reproducible compiler build recipe, no additional rebuild needed | `stereo-affinity-build-recipe.log` |
| Final patch application, fresh and repeated | `stereo-patches-final-apply.log` |
| Normal AOT export and both GPUs, empty write sets and unchanged inputs | `stereo-affinity-export.log`, `stereo-affinity-host.log`, `stereo-affinity-board.log` |
| Full CPU suite with the patched compiler | `stereo-affinity-cpu-tests.log` |
| Full Vulkan suite with the patched compiler | `stereo-affinity-vulkan-tests.log` |
| Dedicated C100 library with patched compiler on host GPU | `c100-gpu-source-compiler-test.log` |
| Compiler version and final patch digests | `compiler-version.txt`, `patch-sha256.txt` |
| Restored normal board power state and cleared temporary environment | `stereo-final-board-state.log` |

The topology annotation is an experiment, not a shipped workaround. It
demonstrates that unnecessary transfers explain most allocation traffic.
Allocator statistics came from temporary instrumentation, removed before
the clean runtime tests. Timing numbers include readback and use ordinary
power management; they are not frame rates.

Device-specific frequency probes and the explicit 921.6 MHz comparison
live in `~/Projects/c100_gpu/docs/reference/normal-dvfs-20261022/` with
their scripts. That controlled comparison restored the normal minimum
and governor afterward. Normal-DVFS board latency remains variable.

See [the investigation](../../stereo-path.md) and
[benchmark reproduction](../../../bench/stereo/README.md).
