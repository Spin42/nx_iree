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
