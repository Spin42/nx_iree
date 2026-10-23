defmodule NxIREE.Vulkan.Kernel do
  @moduledoc """
  A precompiled SPIR-V compute shader used as an `Nx.block/4`.

      kernel = %NxIREE.Vulkan.Kernel{spirv: File.read!("add.spv"), workgroups: {4, 1, 1}}

      defn add(a, b) do
        Nx.block(kernel, [a, b], Nx.template({256}, :f32), fn _kernel, a, b -> a + b end)
      end

  When `NxIREE.Compiler` compiles for the `vulkan-spirv` backend, the block
  runs as one dispatch of the shader. Everywhere else (other IREE backends,
  EXLA, `Nx.BinaryBackend`) the block runs its default function, so the
  same code works on every Nx backend and the default doubles as the
  reference implementation. Several dispatches that pass intermediate
  tensors to each other form an `NxIREE.Vulkan.Pipeline`.

  ## Shader contract

  The shader's entry point is `main` in descriptor set 0. Its storage-buffer
  bindings are the block's inputs in order, followed by its outputs in order.
  Inputs are read-only and every output element must be written. Shapes and
  workgroup counts are static; scalars are one-element buffers.

  The shader is linked as is: the capabilities and extensions it declares
  must be available on the device, and the rest of the function is compiled
  for the target given by the `:iree_compiler_flags`.

  ## Fields

    * `:spirv` - the SPIR-V binary.
    * `:workgroups` - the `{x, y, z}` dispatch counts, each in 1..65535.
    * `:specialization` - values for the shader's specialization constants,
      as a map from `constant_id` to a 32-bit integer, float or boolean.
      They replace the constants' default values in the binary, as
      `VkSpecializationInfo` would. Defaults to `%{}`.
  """

  @enforce_keys [:spirv, :workgroups]
  defstruct [:spirv, :workgroups, specialization: %{}]

  @type t :: %__MODULE__{
          spirv: binary(),
          workgroups: {pos_integer(), pos_integer(), pos_integer()},
          specialization: %{non_neg_integer() => integer() | float() | boolean()}
        }

  @call_target "nx_iree.vulkan.kernel"

  @doc false
  def call_target, do: @call_target

  @doc false
  # Collects the blocks lowered while `fun` builds the module for `backend`.
  def lowering(backend, fun) do
    previous = Process.put(__MODULE__, %{backend: backend, kernels: []})

    try do
      result = fun.()
      {result, Process.get(__MODULE__).kernels}
    after
      if previous, do: Process.put(__MODULE__, previous), else: Process.delete(__MODULE__)
    end
  end

  @doc false
  # Registers a block met during lowering and returns its index, or :skip
  # when the module being built is not a Vulkan module.
  def register(block, out, args) do
    case Process.get(__MODULE__) do
      %{backend: "vulkan-spirv", kernels: kernels} = state ->
        entry = normalize(block, out, args)
        Process.put(__MODULE__, %{state | kernels: kernels ++ [entry]})
        {:ok, length(kernels)}

      _ ->
        :skip
    end
  end

  # Every block lowers as a pipeline of stages over named tensors
  defp normalize(%__MODULE__{} = kernel, out, args) do
    inputs = args |> Enum.with_index() |> Enum.map(fn {t, i} -> {:"input#{i}", template(t)} end)

    outputs =
      [out]
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.with_index()
      |> Enum.map(fn {t, i} -> {:"output#{i}", template(t)} end)

    %{
      inputs: inputs,
      stages: [stage(kernel, Keyword.keys(inputs), outputs)],
      outputs: Keyword.keys(outputs)
    }
  end

  defp normalize(%NxIREE.Vulkan.Pipeline{} = pipeline, out, args) do
    NxIREE.Vulkan.Pipeline.validate!(pipeline, out, args)

    %{
      inputs: Enum.zip(pipeline.inputs, Enum.map(args, &template/1)),
      stages:
        Enum.map(pipeline.stages, fn {kernel, reads, writes} -> stage(kernel, reads, writes) end),
      outputs: pipeline.outputs
    }
  end

  defp stage(%__MODULE__{} = kernel, inputs, outputs) do
    validate!(kernel)

    %{
      spirv: NxIREE.Vulkan.SPIRV.specialize(kernel.spirv, kernel.specialization),
      workgroups: kernel.workgroups,
      inputs: inputs,
      outputs: Enum.map(outputs, fn {name, t} -> {name, template(t)} end)
    }
  end

  defp template(tensor), do: Nx.template(tensor.shape, tensor.type)

  defp validate!(%__MODULE__{
         spirv: spirv,
         workgroups: workgroups,
         specialization: specialization
       }) do
    unless NxIREE.Vulkan.SPIRV.valid?(spirv) do
      raise ArgumentError, "NxIREE.Vulkan.Kernel :spirv must be a little-endian SPIR-V binary"
    end

    case workgroups do
      {x, y, z} when x in 1..65_535 and y in 1..65_535 and z in 1..65_535 ->
        :ok

      other ->
        raise ArgumentError,
              "NxIREE.Vulkan.Kernel :workgroups must be {x, y, z} in 1..65535, got: #{inspect(other)}"
    end

    unless is_map(specialization) do
      raise ArgumentError,
            "NxIREE.Vulkan.Kernel :specialization must be a map, got: #{inspect(specialization)}"
    end
  end
end

if Code.ensure_loaded?(EXLA.CustomCall) do
  for block <- [NxIREE.Vulkan.Kernel, NxIREE.Vulkan.Pipeline] do
    defimpl EXLA.CustomCall, for: block do
      def call(block, out, args, _client) do
        case NxIREE.Vulkan.Kernel.register(block, out, args) do
          {:ok, index} ->
            {:ok,
             %EXLA.CustomCall.Spec{
               call_target_name: NxIREE.Vulkan.Kernel.call_target(),
               backend_config: [{"kernel", Integer.to_string(index)}]
             }}

          :skip ->
            :skip
        end
      end
    end
  end
end
