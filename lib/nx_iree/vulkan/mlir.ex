defmodule NxIREE.Vulkan.MLIR do
  @moduledoc false

  def render(spec, directory) do
    executables =
      spec.stages
      |> Enum.with_index()
      |> Enum.map_join("\n", fn {stage, index} ->
        path = Path.join(directory, "kernel#{index}.spv")
        File.write!(path, stage.spirv)
        executable(stage, {index, path})
      end)

    dispatches =
      spec.stages
      |> Enum.with_index()
      |> Enum.map_join("\n", fn {stage, index} ->
        "#{names(stage.outputs)} = flow.dispatch @kernel#{index}::@main[](#{names(stage.inputs)})" <>
          " : (#{types(stage.inputs)}) -> (#{types(stage.outputs)})"
      end)

    parameters = Enum.map_join(spec.inputs, ", ", fn {id, t} -> "%v#{id}: #{tensor_type(t)}" end)

    """
    #spirv_target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      iree_codegen.target_info = #iree_gpu.target<
        arch = "", features = "#{spec.features}", wgp = <
          compute = fp32|int32, storage = b32, subgroup = none,
          subgroup_size_choices = [64, 64],
          max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
          max_workgroup_memory_bytes = 16384,
          max_workgroup_counts = [65535, 65535, 65535]>
      >
    }>
    #vulkan_target = #hal.device.target<"vulkan", [#spirv_target]> : !hal.device
    module @module attributes {hal.device.targets = [#vulkan_target]} {
      #{executables}
      func.func @main(#{parameters}) -> (#{types(spec.outputs)}) {
        #{dispatches}
        return #{names(spec.outputs)} : #{types(spec.outputs)}
      }
    }
    """
  end

  defp executable(stage, {index, path}) do
    {x, y, z} = stage.workgroups

    bindings =
      (Enum.map(stage.inputs, fn _ -> "#hal.pipeline.binding<storage_buffer, ReadOnly>" end) ++
         Enum.map(stage.outputs, fn _ -> "#hal.pipeline.binding<storage_buffer>" end))
      |> Enum.join(",\n")

    """
    hal.executable.source private @kernel#{index} <objects = #hal.executable.objects<{
      #spirv_target = [#hal.executable.object<{path = "#{mlir_string(path)}"}>]
    }>> {
      hal.executable.export public @main ordinal(0)
        layout(#hal.pipeline.layout<bindings = [#{bindings}]>)
        count(%device: !hal.device) -> (index, index, index) {
          %x = arith.constant #{x} : index
          %y = arith.constant #{y} : index
          %z = arith.constant #{z} : index
          hal.return %x, %y, %z : index, index, index
        }
    }
    """
  end

  defp names(values), do: Enum.map_join(values, ", ", fn {id, _} -> "%v#{id}" end)
  defp types(values), do: Enum.map_join(values, ", ", fn {_, tensor} -> tensor_type(tensor) end)

  defp tensor_type(%Nx.Tensor{shape: shape, type: type}) do
    dimensions = shape |> Tuple.to_list() |> Enum.map(&Integer.to_string/1)

    element =
      case type do
        {:s, 32} -> "i32"
        {:u, 32} -> "ui32"
        {:f, 32} -> "f32"
      end

    "tensor<" <> Enum.join(dimensions ++ [element], "x") <> ">"
  end

  defp mlir_string(binary) do
    for <<byte <- binary>>, into: "" do
      if byte in 32..126 and byte not in [?", ?\\],
        do: <<byte>>,
        else: "\\" <> (byte |> Integer.to_string(16) |> String.pad_leading(2, "0"))
    end
  end
end
