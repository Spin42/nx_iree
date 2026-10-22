# Whole-line Vulkan recurrence evidence, 2026-10-22

The final source is in `bench/stereo/{path_shared,path_subgroup}.comp`,
`export_vulkan.exs`, `path.exs`, `reference.exs` and `run.exs`.
`lib/nx_iree/vulkan.ex` provides the workload-independent import API.
`source-sha256.txt` records the tested source files.

## Correctness and speed

- `stereo-opt-final-*-export.log`: each shader variant compiled through the
  new API; every generated SPIR-V object passed `spirv-val` before import.
- `stereo-opt-final-{shared,subgroup,subgroup_shared}-host.log`: exact results
  on AMD Navi32/RADV, URI `vulkan://GPU-00000000030000000000000000000000`.
- `stereo-opt-final-controlled-board.log`: same-clock C100/NVK comparison,
  previous Nx export followed by all three final kernel variants. Each
  variant checks sixteen fixtures, two benchmark shapes and ten repeats
  per benchmark, including unchanged resident inputs. Every child exited 0;
  every GPU error count is 0. The clock minimum is restored in `after`.
- `stereo-opt-final-subgroup-board.log` and
  `stereo-opt-final-shared-variants-board.log`: preceding normal-DVFS checks.
  The final controlled run also includes the subsequently strengthened
  minima-at-chunk-boundaries fixtures. No firmware/runtime changes occurred.
- `stereo-opt-normalized-nx-check.log`: sixteen fixtures, each checked on
  the Nx evaluator and IREE CPU (32 exact comparisons).

`PATH_BENCH` records total wall time including output readback.
`PATH_PHASES` separates completed invocation and readback. IREE's public
function contains a final `hal.fence.await`, so the former includes GPU
completion plus runtime overhead, rather than only submission latency.

The 160×120×64 controlled medians are 109.4065 ms (previous Nx),
13.1475 ms (shared), 11.138 ms (subgroup), and 12.561 ms (forced fallback).
Subgroup invocation median is 3.402 ms and readback median is 7.753 ms.
These are directional cost-volume measurements, not full stereo frame rates.

## Arithmetic regression

`stereo-opt-arithmetic-red-host.log` captures a real GPU failure of the
initial shader on the large-penalty fixture. Final logs show the same case
passing, plus the finite-boundary-sentinel regression.

`stereo-opt-arithmetic-red.js` and `.json` preserve a supplementary CPU
wrapping-arithmetic reproduction and 2,000 deterministic comparisons of
the normalized/clamped formula against the scalar reference. These are not
substitutes for the real GPU checks above.

## API and environment

`nxiree-vulkan-api-test.log` is the isolated host-only API test, including
GLSL fixture regeneration and SPIR-V validation. Full regression-suite logs
`stereo-opt-full-cpu-tests.log` and `stereo-opt-full-vulkan-tests.log` show
805 CPU checks passing (one GPU-specific test excluded) and 741 Vulkan
checks passing. API tests use independent signed
integer and float/unsigned kernels; they do not mirror the SGM shader.

`stereo-opt-optimized-controlled-board.log` records an additional final
comparison using `glslang -Os`. It reduced shader size but not invocation
time (about 3.36 ms without it versus 3.43 ms with it). The option is not
enabled in the repository exporter. All exact checks still passed and the
normal clock minimum was restored afterward.

The host compiler is IREE v3.13.0 source revision
`89ece87b89852100c0a4ac3d515e14842d090967`, with the existing allocator and
affinity patches. No additional IREE compiler/runtime patch is needed for
external dispatch. Reproduction commands are in `bench/stereo/README.md`
and `docs/vulkan-kernels.md`. The shader compiler/validator used here came
from the system's existing Buildroot host tools (glslang 16.1.0,
SPIRV-Tools 1.4.350.0).

`stereo-opt-final-board-state.log` confirms normal `simple_ondemand`, minimum
and idle 76.8 MHz, maximum 921.6 MHz, no GPU errors, no temporary benchmark
environment overrides, and unchanged boot ID
`6f672cb9-e70d-433c-8894-7231f93bf6b7`.

Final bundles remain staged under `/root/nxiree/stereo-opt-final-*.bin` and
use the already-tested runtime at `/root/nxiree-stereo`. Generated host
bundles are `/tmp/stereo-opt-final-*.bin`. Temporary experiments are inactive;
the repository source and these logs are the durable reproduction record.
