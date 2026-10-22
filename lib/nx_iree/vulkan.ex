defmodule NxIREE.Vulkan do
  @moduledoc """
  Exports precompiled SPIR-V compute shaders and pipelines for NxIREE.

  The shader uses entry point "main" and descriptor set 0. Storage-buffer
  bindings are the inputs in order, followed by the outputs in order.
  Inputs must be read-only and every output element must be written.

  Shapes and dispatch workgroup counts are static. Supported element types
  are s32, u32 and f32. See docs/vulkan-kernels.md for the shader contract.
  """

  import Bitwise, only: [>>>: 2, &&&: 2]

  @default_features "spirv:v1.3,cap:Shader"
  @flags [
    "--iree-hal-target-backends=vulkan-spirv",
    "--iree-execution-model=async-internal",
    "--iree-hal-memoization=false"
  ]

  @doc """
  Compiles a SPIR-V binary into the format accepted by NxIREE.load_exported/2.

  The input templates are a flat list of tensors. The output is one tensor
  template or a nonempty flat tuple of tensor templates. Scalar tensors are
  passed as one-element buffer bindings, not push constants.

  Options:

    * :workgroups — required {x, y, z} dispatch counts, each in 1..65535.
    * :target_features — IREE SPIR-V version, capability and extension list.
      Defaults to "spirv:v1.3,cap:Shader". The caller must select requirements
      supported by the destination device.
    * :external_resources_mappable — request host-visible, device-local,
      mappable external buffers. Defaults to false. This can avoid readback
      staging on shared-memory GPUs, but may slow allocations or access on
      discrete GPUs. Transient intermediate buffers keep IREE's normal policy.

  This host operation requires iree-compile, but not EXLA or a GPU.
  The exported binary runs with NxIREE.load_exported/2 and NxIREE.run/2.
  """
  def export(spirv, input_templates, output_template, opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    opts =
      Keyword.validate!(opts,
        workgroups: nil,
        target_features: @default_features,
        external_resources_mappable: false
      )

    flags = allocation_flags!(opts[:external_resources_mappable])
    workgroups = workgroups!(opts[:workgroups])
    features = target_features!(opts[:target_features])
    validate_spirv!(spirv, features)

    unless is_list(input_templates) do
      raise ArgumentError, "input_templates must be a flat list of tensor templates"
    end

    inputs =
      input_templates
      |> Enum.with_index()
      |> Enum.map(fn {template, index} -> template!(template, "input #{index}") end)

    {output, outputs} = outputs!(output_template)
    inputs = Enum.with_index(inputs) |> Enum.map(fn {t, i} -> {i, t} end)
    outputs = Enum.with_index(outputs, length(inputs)) |> Enum.map(fn {t, i} -> {i, t} end)

    compile(%{
      inputs: inputs,
      stages: [%{spirv: spirv, inputs: inputs, outputs: outputs, workgroups: workgroups}],
      outputs: outputs,
      output: output,
      features: features,
      flags: flags
    })
  end

  @doc """
  Exports several immutable-buffer dispatches as one IREE function.

  Inputs are named tensor templates in a keyword list. Each stage is a keyword
  list with :spirv, :inputs (previously defined names), :outputs (new named
  templates), and :workgroups. Names cannot be overwritten. Return one name
  or a nonempty flat tuple of names. The runtime receives positional inputs
  in the initial keyword order. IREE manages intermediate buffers and barriers.

  Global options are :target_features and :external_resources_mappable, as
  in export/4. Every shader obeys the same read-only-input and fully-written-output
  contract.
  """
  def export_pipeline(input_templates, stages, result_names, opts \\ []) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    opts =
      Keyword.validate!(opts,
        target_features: @default_features,
        external_resources_mappable: false
      )

    flags = allocation_flags!(opts[:external_resources_mappable])
    features = target_features!(opts[:target_features])
    named_inputs = named_templates!(input_templates, "inputs")

    values =
      named_inputs
      |> Enum.with_index()
      |> Map.new(fn {{name, tensor}, id} -> {name, {id, tensor}} end)

    unless is_list(stages) and stages != [],
      do: raise(ArgumentError, "stages must be a nonempty list")

    {stages, values} = Enum.map_reduce(stages, values, &stage!(&1, &2, features))

    names =
      cond do
        is_atom(result_names) -> [result_names]
        is_tuple(result_names) and tuple_size(result_names) > 0 -> Tuple.to_list(result_names)
        true -> raise ArgumentError, "results must be a name or nonempty flat tuple of names"
      end

    outputs = Enum.map(names, &value!(values, &1))
    templates = Enum.map(outputs, &elem(&1, 1))
    output = if is_atom(result_names), do: hd(templates), else: List.to_tuple(templates)

    compile(%{
      inputs: Enum.map(named_inputs, fn {name, _} -> Map.fetch!(values, name) end),
      stages: stages,
      outputs: outputs,
      output: output,
      features: features,
      flags: flags
    })
  end

  defp named_templates!(named, label) do
    unless Keyword.keyword?(named) and length(Keyword.keys(named)) == map_size(Map.new(named)),
      do: raise(ArgumentError, "#{label} must be a keyword list with unique names")

    Enum.map(named, fn {name, tensor} -> {name, template!(tensor, "#{label} #{name}")} end)
  end

  defp value!(values, name) do
    case Map.fetch(values, name) do
      {:ok, value} -> value
      :error -> raise ArgumentError, "undefined buffer #{inspect(name)}"
    end
  end

  defp stage!(stage, values, features) do
    unless Keyword.keyword?(stage), do: raise(ArgumentError, "stage must be a keyword list")
    stage = Keyword.validate!(stage, spirv: nil, inputs: nil, outputs: nil, workgroups: nil)
    validate_spirv!(stage[:spirv], features)
    groups = workgroups!(stage[:workgroups])
    unless is_list(stage[:inputs]), do: raise(ArgumentError, "stage inputs must be a list")
    inputs = Enum.map(stage[:inputs], &value!(values, &1))
    named_outputs = named_templates!(stage[:outputs], "stage outputs")
    if named_outputs == [], do: raise(ArgumentError, "stage outputs must be nonempty")

    {outputs, values} =
      Enum.map_reduce(named_outputs, values, fn {name, template}, values ->
        if Map.has_key?(values, name), do: raise(ArgumentError, "buffer #{name} already defined")
        value = {map_size(values), template}
        {value, Map.put(values, name, value)}
      end)

    {%{spirv: stage[:spirv], inputs: inputs, outputs: outputs, workgroups: groups}, values}
  end

  defp compile(spec) do
    directory = temporary_directory!()

    try do
      module =
        NxIREE.compile(NxIREE.Vulkan.MLIR.render(spec, directory), @flags ++ spec.flags,
          output_container: spec.output
        )

      :erlang.term_to_binary(%{
        format: NxIREE.Exported.format(),
        bytecode: module.bytecode,
        compilation_flags: module.compilation_flags,
        inputs: Enum.map(spec.inputs, fn {_, t} -> {t.shape, t.type} end),
        used_inputs: Enum.map(spec.inputs, &elem(&1, 0)),
        written_inputs: module.written_inputs,
        output: spec.output,
        callback_pid_size: 0
      })
    after
      File.rm_rf!(directory)
    end
  end

  defp allocation_flags!(false), do: []
  defp allocation_flags!(true), do: ["--iree-hal-external-resources-mappable=true"]

  defp allocation_flags!(value) do
    raise ArgumentError,
          ":external_resources_mappable must be a boolean, got #{inspect(value)}"
  end

  defp workgroups!({x, y, z} = counts)
       when is_integer(x) and x in 1..65535 and is_integer(y) and y in 1..65535 and
              is_integer(z) and z in 1..65535,
       do: counts

  defp workgroups!(value) do
    raise ArgumentError,
          ":workgroups must be {x, y, z} with each count in 1..65535, got #{inspect(value)}"
  end

  defp target_features!(features) when is_binary(features) do
    valid =
      Regex.match?(
        ~r/\Aspirv:v1\.[0-6](?:,cap:[A-Za-z][A-Za-z0-9]*|,ext:SPV_[A-Za-z0-9_]+)*\z/,
        features
      ) and "cap:Shader" in String.split(features, ",")

    if valid, do: features, else: invalid_features!(features)
  end

  defp target_features!(value), do: invalid_features!(value)

  defp invalid_features!(value) do
    raise ArgumentError,
          ":target_features must be a SPIR-V version and capability/extension list including cap:Shader, got #{inspect(value)}"
  end

  defp validate_spirv!(
         <<0x07230203::little-32, version::little-32, _generator::little-32, bound::little-32,
           0::little-32, instructions::binary>> = binary,
         "spirv:v1." <> rest
       )
       when rem(byte_size(binary), 4) == 0 and bound > 0 do
    major = version >>> 16
    minor = version >>> 8 &&& 0xFF
    <<target_minor, _::binary>> = rest

    unless major == 1 and minor in 0..6 and (version &&& 0xFF) == 0 do
      raise ArgumentError, "unsupported SPIR-V version word #{inspect(version)}"
    end

    if minor > target_minor - ?0 do
      raise ArgumentError, "SPIR-V 1.#{minor} exceeds the declared target version"
    end

    unless compute_main?(instructions, false) do
      raise ArgumentError, "SPIR-V must contain a GLCompute entry point named main"
    end
  end

  defp validate_spirv!(_binary, _features) do
    raise ArgumentError,
          "spirv must be a little-endian SPIR-V binary with a complete header and 32-bit words"
  end

  defp compute_main?(<<>>, found), do: found

  defp compute_main?(<<word::little-32, rest::binary>>, found) do
    count = word >>> 16
    opcode = word &&& 0xFFFF

    if count == 0 or (count - 1) * 4 > byte_size(rest) do
      raise ArgumentError, "SPIR-V contains a truncated or zero-length instruction"
    end

    size = (count - 1) * 4
    operands = binary_part(rest, 0, size)
    rest = binary_part(rest, size, byte_size(rest) - size)

    main =
      case {opcode, operands} do
        {15, <<5::little-32, _id::little-32, name::binary>>} ->
          match?(["main", _], :binary.split(name, <<0>>))

        _ ->
          false
      end

    compute_main?(rest, found or main)
  end

  defp template!(%Nx.Tensor{shape: shape, type: type, vectorized_axes: []} = tensor, label) do
    unless is_tuple(shape) and Enum.all?(Tuple.to_list(shape), &(is_integer(&1) and &1 > 0)) do
      raise ArgumentError, "#{label} must have a static shape with positive dimensions"
    end

    unless type in [{:s, 32}, {:u, 32}, {:f, 32}] do
      raise ArgumentError, "#{label} must have type s32, u32 or f32, got #{inspect(type)}"
    end

    Nx.to_template(tensor)
  end

  defp template!(_tensor, label) do
    raise ArgumentError, "#{label} must be a non-vectorized Nx tensor template"
  end

  defp outputs!(%Nx.Tensor{} = tensor) do
    output = template!(tensor, "output")
    {output, [output]}
  end

  defp outputs!(tuple) when is_tuple(tuple) and tuple_size(tuple) > 0 do
    outputs =
      tuple
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.map(fn {template, index} -> template!(template, "output #{index}") end)

    {List.to_tuple(outputs), outputs}
  end

  defp outputs!(_value) do
    raise ArgumentError, "output must be a tensor template or a nonempty flat tuple of templates"
  end

  defp temporary_directory! do
    name =
      "nx_iree-vulkan-#{System.pid()}-" <>
        Base.encode16(:crypto.strong_rand_bytes(16), case: :lower)

    path = Path.join(System.tmp_dir!(), name)
    File.mkdir!(path)
    path
  end
end
