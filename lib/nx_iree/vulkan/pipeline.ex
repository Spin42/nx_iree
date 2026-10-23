defmodule NxIREE.Vulkan.Pipeline do
  @moduledoc """
  Several `NxIREE.Vulkan.Kernel` dispatches used as one `Nx.block/4`.

  The stages read and write named tensors: the block's inputs, and the
  outputs of earlier stages. The block returns the named tensors listed in
  `:outputs`.

      pipeline = %NxIREE.Vulkan.Pipeline{
        inputs: [:image],
        stages: [
          {blur, [:image], [blurred: Nx.template({480, 640}, :f32)]},
          {edges, [:blurred], [edges: Nx.template({480, 640}, :u32)]}
        ],
        outputs: [:edges]
      }

      Nx.block(pipeline, [image], Nx.template({480, 640}, :u32), &MyFilters.edges/2)

  As with a single kernel, the block runs its default function wherever it
  is not compiled for Vulkan by `NxIREE.Compiler`. Intermediate tensors live
  on the device between the dispatches; dispatches run in order.

  ## Fields

    * `:inputs` - names for the block's inputs, in order.
    * `:stages` - `{kernel, input_names, outputs}` tuples, in execution
      order. `outputs` is a keyword list of names and tensor templates; the
      shader's bindings are the inputs, then the outputs, in order.
    * `:outputs` - the names of the tensors the block returns, in the order
      of its output template.
  """

  @enforce_keys [:inputs, :stages, :outputs]
  defstruct [:inputs, :stages, :outputs]

  @type t :: %__MODULE__{
          inputs: [atom()],
          stages: [{NxIREE.Vulkan.Kernel.t(), [atom()], keyword(Nx.Tensor.t())}],
          outputs: [atom()]
        }

  @doc false
  def validate!(%__MODULE__{inputs: inputs, stages: stages, outputs: outputs}, out, args) do
    unless is_list(inputs) and Enum.all?(inputs, &is_atom/1) and length(inputs) == length(args) do
      raise ArgumentError,
            "NxIREE.Vulkan.Pipeline :inputs must name the block's #{length(args)} inputs, got: #{inspect(inputs)}"
    end

    Enum.reduce(stages, MapSet.new(inputs), fn
      {%NxIREE.Vulkan.Kernel{}, reads, writes}, defined when is_list(reads) and is_list(writes) ->
        for name <- reads, not MapSet.member?(defined, name) do
          raise ArgumentError,
                "NxIREE.Vulkan.Pipeline stage reads #{inspect(name)} before it is defined"
        end

        for {name, template} <- writes do
          unless is_atom(name) and is_struct(template, Nx.Tensor) do
            raise ArgumentError,
                  "NxIREE.Vulkan.Pipeline stage outputs must be name: template pairs"
          end

          if MapSet.member?(defined, name) do
            raise ArgumentError, "NxIREE.Vulkan.Pipeline defines #{inspect(name)} twice"
          end
        end

        MapSet.union(defined, MapSet.new(Keyword.keys(writes)))

      stage, _ ->
        raise ArgumentError,
              "NxIREE.Vulkan.Pipeline stages are {kernel, input_names, outputs}, got: #{inspect(stage)}"
    end)

    templates = Map.new(Enum.flat_map(stages, fn {_, _, writes} -> writes end))

    for name <- outputs, not Map.has_key?(templates, name) do
      raise ArgumentError,
            "NxIREE.Vulkan.Pipeline output #{inspect(name)} is not written by any stage"
    end

    expected = [out] |> Nx.Defn.Composite.flatten_list() |> Enum.map(&{&1.shape, &1.type})
    actual = Enum.map(outputs, fn name -> {templates[name].shape, templates[name].type} end)

    unless expected == actual do
      raise ArgumentError,
            "NxIREE.Vulkan.Pipeline outputs #{inspect(actual)} do not match the block's output #{inspect(expected)}"
    end
  end
end
