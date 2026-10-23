defmodule NxIREE.Vulkan.MLIR do
  @moduledoc false

  @kernel_call ~r/^(\s*)(%[\w#]+(?::\d+)?) = stablehlo\.custom_call @nx_iree\.vulkan\.kernel\(([^)]*)\) \{[^\n]*backend_config = \{kernel = (\d+) : i64\}[^\n]*\} : \(([^)]*)\) -> (.+)$/m

  # Replaces the custom calls EXLA emitted for NxIREE.Vulkan blocks in
  # `module` with dispatches of their shaders, written to `directory`. Each
  # block is a pipeline of stages over named tensors (see
  # NxIREE.Vulkan.Kernel.register/3). IREE links an executable object into
  # every variant of a matching target, and the bare vulkan-spirv target
  # matches all of them. Dispatches take signless tensors, as IREE's input
  # conversion makes every other operation signless, so unsigned block
  # inputs and outputs are bitcast at the block's boundary.
  def inline_kernels(module, [], _directory), do: module

  def inline_kernels(module, blocks, directory) do
    executables =
      for {block, n} <- Enum.with_index(blocks), {stage, j} <- Enum.with_index(block.stages) do
        path = Path.join(directory, "kernel#{n}_#{j}.spv")
        File.write!(path, stage.spirv)

        summary = %{
          workgroups: stage.workgroups,
          inputs: length(stage.inputs),
          outputs: length(stage.outputs)
        }

        executable(summary, {n, path}, "nx_iree_kernel#{n}_#{j}", "#nx_iree_kernel_target")
      end

    {body, renames} =
      Regex.scan(@kernel_call, module)
      |> Enum.reduce({module, []}, fn [line | captures], {body, renames} ->
        {text, new_renames} = dispatches(captures, blocks)
        {String.replace(body, line, text, global: false), new_renames ++ renames}
      end)

    body =
      Enum.reduce(renames, body, fn {pattern, value}, body ->
        Regex.replace(pattern, body, value)
      end)

    if String.contains?(body, "@nx_iree.vulkan.kernel") do
      raise NxIREE.CompileError, message: "unexpected form of an NxIREE.Vulkan custom call"
    end

    [header, rest] = String.split(body, "\n", parts: 2)

    ~s|#nx_iree_kernel_target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">\n| <>
      header <> "\n" <> Enum.join(executables, "\n") <> "\n" <> rest
  end

  defp dispatches([indent, results, operands, index, operand_types, result_types], blocks) do
    n = String.to_integer(index)
    block = Enum.at(blocks, n)
    prefix = "%nx_iree_kernel#{n}"

    result_types =
      result_types |> String.trim() |> String.trim_leading("(") |> String.trim_trailing(")")

    # The block's inputs, cast to signless tensors where needed
    {values, casts_in} =
      [split(operands), split(operand_types), block.inputs]
      |> Enum.zip()
      |> Enum.with_index()
      |> Enum.map_reduce([], fn {{operand, type, {name, template}}, i}, lines ->
        if unsigned?(type) do
          cast = "#{prefix}_in#{i}"

          line =
            "#{indent}#{cast} = stablehlo.bitcast_convert #{operand} : (#{type}) -> #{tensor_type(template)}"

          {{name, cast}, [line | lines]}
        else
          {{name, operand}, lines}
        end
      end)

    {values, stage_lines} =
      block.stages
      |> Enum.with_index()
      |> Enum.reduce({Map.new(values), []}, fn {stage, j}, {values, lines} ->
        result = "#{prefix}_#{j}"
        count = length(stage.outputs)
        defined = if count == 1, do: result, else: "#{result}:#{count}"
        reads = Enum.map(stage.inputs, &Map.fetch!(values, &1))
        read_types = Enum.map(stage.inputs, &tensor_type(template_of(block, &1)))
        write_types = Enum.map(stage.outputs, fn {_, template} -> tensor_type(template) end)

        line =
          "#{indent}#{defined} = flow.dispatch @nx_iree_kernel#{n}_#{j}::@main[](#{Enum.join(reads, ", ")})" <>
            " : (#{Enum.join(read_types, ", ")}) -> (#{Enum.join(write_types, ", ")})"

        written =
          stage.outputs
          |> Enum.with_index()
          |> Map.new(fn {{name, _}, m} ->
            {name, if(count == 1, do: result, else: "#{result}##{m}")}
          end)

        {Map.merge(values, written), [line | lines]}
      end)

    # The block's outputs, cast back where they are unsigned
    {outputs, casts_out} =
      block.outputs
      |> Enum.zip(split(result_types))
      |> Enum.with_index()
      |> Enum.map_reduce([], fn {{name, type}, i}, lines ->
        value = Map.fetch!(values, name)

        if unsigned?(type) do
          cast = "#{prefix}_out#{i}"

          line =
            "#{indent}#{cast} = stablehlo.bitcast_convert #{value} : (#{tensor_type(template_of(block, name))}) -> #{type}"

          {cast, [line | lines]}
        else
          {value, lines}
        end
      end)

    original = results |> String.split(":") |> hd()

    renames =
      case outputs do
        [value] ->
          [{~r/#{Regex.escape(original)}(?![\w#:])/, value}]

        values ->
          values
          |> Enum.with_index()
          |> Enum.map(fn {value, i} -> {~r/#{Regex.escape(original)}##{i}(?!\d)/, value} end)
      end

    lines = Enum.reverse(casts_in) ++ Enum.reverse(stage_lines) ++ Enum.reverse(casts_out)
    {Enum.join(lines, "\n"), renames}
  end

  defp template_of(block, name) do
    Enum.find_value(block.inputs, fn {input, template} -> input == name && template end) ||
      Enum.find_value(block.stages, fn stage -> Keyword.get(stage.outputs, name) end)
  end

  defp split(""), do: []
  defp split(list), do: list |> String.split(",") |> Enum.map(&String.trim/1)

  defp unsigned?(type), do: Regex.match?(~r/xui\d+>$|<ui\d+>$/, type)

  defp executable(stage, {_index, path}, name, target) do
    {x, y, z} = stage.workgroups

    bindings =
      (List.duplicate("#hal.pipeline.binding<storage_buffer, ReadOnly>", count(stage.inputs)) ++
         List.duplicate("#hal.pipeline.binding<storage_buffer>", count(stage.outputs)))
      |> Enum.join(",\n")

    """
    hal.executable.source private @#{name} <objects = #hal.executable.objects<{
      #{target} = [#hal.executable.object<{path = "#{mlir_string(path)}"}>]
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

  defp count(values) when is_integer(values), do: values
  defp count(values), do: length(values)


  defp tensor_type(%Nx.Tensor{shape: shape, type: type}) do
    dimensions = shape |> Tuple.to_list() |> Enum.map(&Integer.to_string/1)

    element =
      case type do
        {:f, size} -> "f#{size}"
        {:bf, 16} -> "bf16"
        {kind, size} when kind in [:s, :u] -> "i#{size}"
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
