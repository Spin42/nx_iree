defmodule NxIREE.Vulkan.SPIRV do
  @moduledoc false
  # Minimal SPIR-V reading and patching for NxIREE.Vulkan kernels.

  import Bitwise

  @magic 0x07230203
  @op_type_int 21
  @op_type_float 22
  @op_spec_constant_true 48
  @op_spec_constant_false 49
  @op_spec_constant 50
  @op_decorate 71
  @decoration_spec_id 1

  @doc false
  def valid?(binary)
      when is_binary(binary) and byte_size(binary) >= 20 and rem(byte_size(binary), 4) == 0 do
    match?(<<@magic::little-32, _::binary>>, binary)
  end

  def valid?(_), do: false

  @doc false
  # Sets the default values of specialization constants, by SpecId. Shaders
  # created without VkSpecializationInfo use these defaults, so drivers
  # compile them as constants.
  def specialize(binary, values) when map_size(values) == 0, do: binary

  def specialize(<<header::binary-size(20), body::binary>>, values) do
    instructions = instructions(body)

    types =
      for {op, [id, width | _]} <- instructions,
          op in [@op_type_int, @op_type_float],
          into: %{},
          do: {id, {op, width}}

    spec_ids =
      for {@op_decorate, [target, @decoration_spec_id, spec_id]} <- instructions,
          into: %{},
          do: {spec_id, target}

    targets =
      Map.new(values, fn {spec_id, value} ->
        case spec_ids do
          %{^spec_id => target} -> {target, {spec_id, value}}
          _ -> raise ArgumentError, "the shader has no specialization constant #{spec_id}"
        end
      end)

    patched =
      Enum.map(instructions, fn {op, operands} = instruction ->
        case {op, operands} do
          {op, [type, id | _]}
          when op in [@op_spec_constant, @op_spec_constant_true, @op_spec_constant_false] ->
            case targets do
              %{^id => {spec_id, value}} -> constant(op, type, id, value, types, spec_id)
              _ -> instruction
            end

          _ ->
            instruction
        end
      end)

    header <> IO.iodata_to_binary(Enum.map(patched, &encode/1))
  end

  defp constant(op, type, id, value, _types, _spec_id)
       when op in [@op_spec_constant_true, @op_spec_constant_false] and is_boolean(value) do
    {if(value, do: @op_spec_constant_true, else: @op_spec_constant_false), [type, id]}
  end

  defp constant(@op_spec_constant, type, id, value, types, spec_id) do
    word =
      case {types[type], value} do
        {{@op_type_int, 32}, value} when is_integer(value) and value in -0x80000000..0xFFFFFFFF ->
          value &&& 0xFFFFFFFF

        {{@op_type_float, 32}, value} when is_number(value) ->
          <<word::little-32>> = <<value * 1.0::little-float-32>>
          word

        _ ->
          raise ArgumentError,
                "specialization constant #{spec_id} takes a 32-bit integer, float or boolean " <>
                  "matching its type, got: #{inspect(value)}"
      end

    {@op_spec_constant, [type, id, word]}
  end

  defp constant(_op, _type, _id, value, _types, spec_id) do
    raise ArgumentError, "specialization constant #{spec_id} is a boolean, got: #{inspect(value)}"
  end

  defp instructions(<<>>), do: []

  defp instructions(<<opcode::little-16, count::little-16, rest::binary>>) when count >= 1 do
    size = (count - 1) * 4
    <<operands::binary-size(^size), rest::binary>> = rest
    [{opcode, for(<<word::little-32 <- operands>>, do: word)} | instructions(rest)]
  end

  defp encode({opcode, operands}) do
    [
      <<opcode::little-16, length(operands) + 1::little-16>>
      | Enum.map(operands, &<<&1::little-32>>)
    ]
  end
end
