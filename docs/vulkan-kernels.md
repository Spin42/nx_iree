# Custom Vulkan compute kernels

`NxIREE.Vulkan.export/4` packages a precompiled SPIR-V compute shader into the
same format used by `NxIREE.export/3`. Load it once with
`NxIREE.load_exported/2`, then call it with `NxIREE.run/2`. Exporting requires
the host IREE compiler; running the exported binary needs neither that
compiler nor EXLA.

This is a small interface for static dispatches and pipelines. It contains
no device-specific tuning. The shader's memory accesses, synchronization and
work distribution remain the shader author's responsibility.

## Example

Save this independent integer transform as `transform.comp`:

```glsl
#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Input { int input_values[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Output { int output_values[]; };
void main() {
  uint i = gl_GlobalInvocationID.x;
  output_values[i] = input_values[i] * 3 + 7;
}
```

Compile and validate it on the host:

```sh
glslangValidator -V --target-env vulkan1.1 -S comp transform.comp -o transform.spv
spirv-val --target-env vulkan1.1 transform.spv
```

Export it for four workgroups, each containing the shader's one invocation:

```elixir
binary = NxIREE.Vulkan.export(
  File.read!("transform.spv"),
  [Nx.template({4}, :s32)],
  Nx.template({4}, :s32),
  workgroups: {4, 1, 1}
)

# This part can run on a different machine with the IREE Vulkan runtime.
{:ok, [gpu | _]} = NxIREE.list_devices("vulkan")
model = NxIREE.load_exported(binary, device: gpu)
input = Nx.tensor([1, 2, 3, 4], type: :s32, backend: Nx.BinaryBackend)
result = NxIREE.run(model, [input])
Nx.backend_transfer(result, Nx.BinaryBackend) |> Nx.to_flat_list()
# => [10, 13, 16, 19]
```

Use a URI returned by `NxIREE.list_devices("vulkan")` to select a particular
GPU. Outputs remain device-resident until transferred or deallocated through
the normal Nx backend API.

## Shader and tensor contract

`NxIREE.Vulkan.export_pipeline/4` composes several shaders into one runtime
invocation. Inputs and stage outputs have unique atom names; each stage
reads initial inputs or earlier outputs. IREE manages intermediate storage
and synchronization. Names are host export metadata, not runtime arguments.

For example, applying the transform shader twice:

```elixir
tensor = Nx.template({4}, :s32)
shader = File.read!("transform.spv")
binary = NxIREE.Vulkan.export_pipeline(
  [input: tensor],
  [
    [spirv: shader, inputs: [:input], outputs: [first: tensor], workgroups: {4, 1, 1}],
    [spirv: shader, inputs: [:first], outputs: [second: tensor], workgroups: {4, 1, 1}]
  ],
  :second
)
```

Load and run normally, with positional inputs in the initial keyword order.
Return a tuple of names to expose several outputs or intermediates. Every
shader follows the contract below. Pipeline options accept `target_features`
and `external_resources_mappable`;
workgroups belong to each stage. Shapes are fixed at export time. This
interface does not enable command-buffer memoization or support mutable
external buffers. It introduces no device-specific policy.

- The compute entry point is `main`. Descriptor set 0 contains contiguous
  storage-buffer bindings: all inputs in list order, then all outputs in
  tuple order. Every declared binding must match the corresponding tensor.
- Inputs are read-only. Qualify their GLSL storage buffers with `readonly`.
  Outputs are new, uninitialized buffers; the shader must write every output
  element before it is returned and must not read an output before writing it.
- Tensors are dense row-major `s32`, `u32` or `f32` arrays with static positive
  dimensions. Scalars use shape `{}` and occupy one element in a buffer.
  Vectorized axes, maps and nested containers are unsupported by this API.
- Input templates form a flat list. The output template is one tensor or a
  nonempty flat tuple of tensors. For example,
  `{Nx.template({4}, :f32), Nx.template({4}, :u32)}` declares two output
  buffers and `run/2` returns that tuple.
- There are no push constants, specialization arguments, sampled images or
  externally supplied output buffers in this initial interface. Pass runtime
  scalar parameters as additional scalar tensor inputs.
- `workgroups: {x, y, z}` is required; each dispatch count is in `1..65535`.
  These are workgroup counts, not local sizes or element counts. The shader's
  compiled local size and bounds checks determine the covered elements.

The input and output buffers must match these declarations. SPIR-V reflection
does not infer or verify the shader's full binding layout or access behavior.
The wrapper checks the binary header, instruction lengths and compute entry
point; use `spirv-val` for complete SPIR-V validation.

The export keeps the compiler's normal input-write analysis. It does not
replace that analysis with a hardcoded claim that inputs are immutable.
Analysis cannot prove that an opaque shader obeys its declared read-only
bindings; that remains part of the shader contract.

## Portable target and capabilities

The default target features are `spirv:v1.3,cap:Shader`, matching the generic
IREE custom-dispatch target. Its conservative resource description uses
128 invocations per workgroup, local dimensions up to `{128, 128, 64}` and
16 KiB of workgroup memory. Keep the shader within those limits and the
destination device's limits. The wrapper does not infer resource use from
SPIR-V.

If the shader requires additional capabilities or extensions, declare them
explicitly, for example:

```elixir
target_features: "spirv:v1.3,cap:Shader,cap:GroupNonUniform,cap:GroupNonUniformArithmetic"
```

The option accepts a version followed by `cap:Name` and `ext:SPV_Name`
entries. Syntax is checked before generating MLIR; this is not a check that
the driver supports those features or that a capability name is meaningful.
The SPIR-V binary's version must not exceed the declared version. No GPU
model, subgroup width or device-specific capability list is selected for you.

The export embeds the shader into the VM bytecode. Temporary shader paths
use an OS process identifier plus cryptographic randomness and are removed
after compilation, including exception paths. Paths are escaped as MLIR
string literals, and arbitrary compiler flags or MLIR attributes are not
accepted by this API.

Host tensor uploads reuse IREE's allocator-aware buffer generator. On
compatible shared-memory devices it initializes mapped device memory
directly; otherwise IREE retains its staged-transfer fallback. The bundled
IREE mapped-generator flush patch keeps this correct on non-coherent
devices. This changes neither input ownership nor result placement, and
does not by itself guarantee an end-to-end performance improvement.

Both export functions accept `external_resources_mappable: true`, forwarding
IREE's existing `--iree-hal-external-resources-mappable` option. This requests
host-visible, device-local, mappable external buffers, while leaving transient
intermediates under normal IREE allocation policy. On shared-memory devices
it can remove a readback staging allocation and submission. It still copies
the result into an Elixir binary. The default is false: this placement can
regress discrete-GPU performance and is not automatic device detection.
The bundled mapped-source invalidation patch is required for correct
readback on non-coherent memory. Uploads also avoid an intermediate native
host allocation/copy and validate the binary length against its shape/type.

## Device profiling

`NxIREE.Profiling.capture(device, path, fn -> work end)` wraps IREE's existing
file profiler and returns the function's result. Inspect the `.ireeprof`
file with `iree-profile summary`, `dispatch`, `command` and `queue`. It captures
real dispatch/queue events, including work from independent contexts on that
device; it adds no alternate executor. Use a separate unprofiled run for FPS.

The device must be idle during setup/teardown. The function must await all
tasks and device work it starts. Do not nest captures or change profiling
concurrently. The session ends in `after`, including an exception path;
unsupported device capabilities raise. Captures overwrite their destination.
The bundled profile-metadata serialization patch protects concurrent-context
file records. The [stereo follow-up](reference/stereo-fused-20261022/README.md)
contains a real two-context capture and readback-copy comparison.

## Verification

`test/nx_iree/vulkan_test.exs` includes independent integer and float/unsigned
shader fixtures, their GLSL source and validated SPIR-V. Host-only tests run
with:

```sh
mix test test/nx_iree/vulkan_test.exs --exclude vulkan
```

Set `GLSLANG` and `SPIRV_VAL` to their executable paths to regenerate and
validate the fixtures during that run. The normal export tests use embedded
binaries and require no shader toolchain.

The `vulkan` test exercises normal load/run, changed inputs, tuple outputs
and preservation of device-resident input data. Run it on a selected GPU:

```sh
NX_IREE_DEFAULT_DRIVER=vulkan NX_IREE_DEFAULT_DEVICE='<URI substring>' \
  mix test test/nx_iree/vulkan_test.exs --only vulkan
```
