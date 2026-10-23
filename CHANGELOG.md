# Changelog

## Unreleased

### Breaking

  * Requires Nx and EXLA 1.0 and IREE 3.13.
  * EXLA is optional; it is only needed to compile functions.

### Added

  * `NxIREE.export/3`, `NxIREE.load_exported/2` and `NxIREE.run/2` compile
    functions on the host and run them where neither EXLA nor the IREE
    compiler is installed.
  * Cross-compilation of the runtime when `CROSSCOMPILE` is set (Nerves).
  * `:drivers` and `:default_device` configuration; devices are created on
    first use.
  * `NxIREE.CompileError` with the compiler's diagnostics.
  * Compiled defn functions are cached and their modules are loaded once per
    device.
  * Functions jitted for the CPU use the host CPU's features.
  * `:iree_module_cache`: modules compiled on a host are found by devices
    running the same defn code without EXLA or the IREE compiler.
  * `NxIREE.Vulkan.Kernel` and `NxIREE.Vulkan.Pipeline`: precompiled SPIR-V
    compute shaders as `Nx.block/4`s, dispatched when compiling for Vulkan
    and falling back to the block's default elsewhere, with specialization
    constants.
  * Module-cache keys contain the contents of transform-dialect tuning specs
    passed with `--iree-codegen-transform-dialect-library`, not their paths.
  * Named io_calls without a hook are removed before lowering, so
    Bumblebee text generation compiles; host calls that would run Elixir
    code raise a clear error.

### Fixed

  * Inputs are no longer modified by modules that sort, scatter or update
    slices in place.
  * `u8`/`s8` tensors read by vectorized Vulkan dispatches, which broke
    masked attention in BERT-like models.
  * Calls wait for device work to finish before returning results.
  * Tensors from another device are copied instead of passed as foreign
    buffers.
  * `Nx.Defn.to_backend/1` returns `NxIREE.Backend` on the configured device,
    and backend operations run on their inputs' device.
  * f64 stays f64 on the CPU and raises a clear error on Vulkan.
  * Reading or passing a deallocated tensor raises instead of crashing.

## v0.0.1 (2024-10-04)

First release, including an Nx.Defn compiler and companion Nx.Backend.
