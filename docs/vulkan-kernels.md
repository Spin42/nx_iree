# Custom Vulkan compute kernels

A precompiled SPIR-V compute shader is an `Nx.block/4` in any Nx function:
`NxIREE.Vulkan.Kernel` for one dispatch, `NxIREE.Vulkan.Pipeline` for
several dispatches that pass intermediate tensors to each other. When
`NxIREE.Compiler` compiles the function for the `vulkan-spirv` backend, the
block runs its shaders; everywhere else (other IREE backends, EXLA,
`Nx.BinaryBackend`) it runs its default function. The same code therefore
works on every Nx backend, and the default is the reference implementation
the shader must match. This mirrors how EXLA lowers blocks to custom calls
(`EXLA.CustomCall`), which is also the mechanism used here.

## Example

Save this transform as `transform.comp`, then compile and validate it:

```glsl
#version 450
layout(local_size_x = 1) in;
layout(constant_id = 0) const int SCALE = 3;
layout(set = 0, binding = 0, std430) readonly buffer Input { int input_values[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Output { int output_values[]; };
void main() {
  uint i = gl_GlobalInvocationID.x;
  output_values[i] = input_values[i] * SCALE + 7;
}
```

```sh
glslangValidator -V --target-env vulkan1.1 -S comp transform.comp -o transform.spv
spirv-val --target-env vulkan1.1 transform.spv
```

```elixir
defmodule Transform do
  import Nx.Defn

  @spirv File.read!("transform.spv")

  defn apply(x, opts \\ []) do
    scale = opts[:scale]
    kernel = %NxIREE.Vulkan.Kernel{spirv: @spirv, workgroups: {4, 1, 1}, specialization: %{0 => scale}}
    Nx.block(kernel, [x], Nx.template({4}, :s32), fn _kernel, x -> x * scale + 7 end)
  end
end

{:ok, [gpu | _]} = NxIREE.list_devices("vulkan")
f = Nx.Defn.jit(&Transform.apply(&1, scale: 3), compiler: NxIREE.Compiler, iree_runtime_options: [device: gpu])
f.(Nx.tensor([1, 2, 3, 4], type: :s32)) |> Nx.to_flat_list()
# => [10, 13, 16, 19]
```

`:specialization` sets specialization constants by `constant_id`, replacing
their defaults in the binary as `VkSpecializationInfo` would, so one shader
serves several sizes or parameters and drivers still compile them as
constants. Module caches and ahead-of-time exports work as for any other
function.

## Pipelines

```elixir
pipeline = %NxIREE.Vulkan.Pipeline{
  inputs: [:x],
  stages: [
    {kernel, [:x], [first: Nx.template({4}, :s32)]},
    {kernel, [:first], [second: Nx.template({4}, :s32)]}
  ],
  outputs: [:second]
}

Nx.block(pipeline, [x], Nx.template({4}, :s32), fn _pipeline, x -> x |> step() |> step() end)
```

Stages read the block's inputs and earlier stages' outputs by name, in
execution order. Intermediate tensors stay on the device; IREE allocates
them and orders the dispatches. [`nx_stereo`](../../nx_stereo) builds a
six-stage stereo matcher this way.

## Shader contract

- The entry point is `main`, in descriptor set 0.
- Storage-buffer bindings are the stage's inputs in order, then its outputs
  in order. Inputs are read-only.
- Every element of a block output must be written. Intermediate tensors may
  be larger than the region a later stage reads.
- Shapes and workgroup counts are static; scalars are one-element buffers.
- The SPIR-V is linked as is. The capabilities and extensions it declares
  must be available on the device; the rest of the function is compiled for
  the target the `:iree_compiler_flags` select.
- Unsigned tensors are bitcast to signless ones at the block's boundary;
  the shader sees the same bits.
