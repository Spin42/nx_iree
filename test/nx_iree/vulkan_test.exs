defmodule NxIREE.VulkanTest do
  use ExUnit.Case

  @single_source """
  #version 450
  layout(local_size_x = 1) in;
  layout(set = 0, binding = 0, std430) readonly buffer Input { int input_values[]; };
  layout(set = 0, binding = 1, std430) writeonly buffer Output { int output_values[]; };
  void main() {
    uint i = gl_GlobalInvocationID.x;
    output_values[i] = input_values[i] * 3 + 7;
  }
  """

  @pair_source """
  #version 450
  layout(local_size_x = 1) in;
  layout(set = 0, binding = 0, std430) readonly buffer Input { float input_values[]; };
  layout(set = 0, binding = 1, std430) writeonly buffer Doubled { float doubled[]; };
  layout(set = 0, binding = 2, std430) writeonly buffer Indices { uint indices[]; };
  void main() {
    uint i = gl_GlobalInvocationID.x;
    doubled[i] = input_values[i] * 2.0;
    indices[i] = i;
  }
  """

  # Generated with glslang -V --target-env vulkan1.1 -S comp; both fixtures
  # were validated with spirv-val --target-env vulkan1.1. Embedded binaries
  # keep the export tests independent of the GLSL toolchain.
  @single Base.decode64!(
            """
            AwIjBwADAQALAAgAJgAAAAAAAAARAAIAAQAAAAsABgABAAAAR0xTTC5zdGQuNDUwAAAAAA4AAwAAAAAAAQAAAA8ABgAFAAAABAAA
            AG1haW4AAAAACwAAABAABgAEAAAAEQAAAAEAAAABAAAAAQAAAAMAAwACAAAAwgEAAAUABAAEAAAAbWFpbgAAAAAFAAMACAAAAGkA
            AAAFAAgACwAAAGdsX0dsb2JhbEludm9jYXRpb25JRAAAAAUABAASAAAAT3V0cHV0AAAGAAcAEgAAAAAAAABvdXRwdXRfdmFsdWVz
            AAAABQADABQAAAAAAAAABQAEABgAAABJbnB1dAAAAAYABwAYAAAAAAAAAGlucHV0X3ZhbHVlcwAAAAAFAAMAGgAAAAAAAABHAAQA
            CwAAAAsAAAAcAAAARwAEABEAAAAGAAAABAAAAEcAAwASAAAAAgAAAEgABAASAAAAAAAAABkAAABIAAUAEgAAAAAAAAAjAAAAAAAA
            AEcAAwAUAAAAGQAAAEcABAAUAAAAIQAAAAEAAABHAAQAFAAAACIAAAAAAAAARwAEABcAAAAGAAAABAAAAEcAAwAYAAAAAgAAAEgA
            BAAYAAAAAAAAABgAAABIAAUAGAAAAAAAAAAjAAAAAAAAAEcAAwAaAAAAGAAAAEcABAAaAAAAIQAAAAAAAABHAAQAGgAAACIAAAAA
            AAAARwAEACUAAAALAAAAGQAAABMAAgACAAAAIQADAAMAAAACAAAAFQAEAAYAAAAgAAAAAAAAACAABAAHAAAABwAAAAYAAAAXAAQA
            CQAAAAYAAAADAAAAIAAEAAoAAAABAAAACQAAADsABAAKAAAACwAAAAEAAAArAAQABgAAAAwAAAAAAAAAIAAEAA0AAAABAAAABgAA
            ABUABAAQAAAAIAAAAAEAAAAdAAMAEQAAABAAAAAeAAMAEgAAABEAAAAgAAQAEwAAAAwAAAASAAAAOwAEABMAAAAUAAAADAAAACsA
            BAAQAAAAFQAAAAAAAAAdAAMAFwAAABAAAAAeAAMAGAAAABcAAAAgAAQAGQAAAAwAAAAYAAAAOwAEABkAAAAaAAAADAAAACAABAAc
            AAAADAAAABAAAAArAAQAEAAAAB8AAAADAAAAKwAEABAAAAAhAAAABwAAACsABAAGAAAAJAAAAAEAAAAsAAYACQAAACUAAAAkAAAA
            JAAAACQAAAA2AAUAAgAAAAQAAAAAAAAAAwAAAPgAAgAFAAAAOwAEAAcAAAAIAAAABwAAAEEABQANAAAADgAAAAsAAAAMAAAAPQAE
            AAYAAAAPAAAADgAAAD4AAwAIAAAADwAAAD0ABAAGAAAAFgAAAAgAAAA9AAQABgAAABsAAAAIAAAAQQAGABwAAAAdAAAAGgAAABUA
            AAAbAAAAPQAEABAAAAAeAAAAHQAAAIQABQAQAAAAIAAAAB4AAAAfAAAAgAAFABAAAAAiAAAAIAAAACEAAABBAAYAHAAAACMAAAAU
            AAAAFQAAABYAAAA+AAMAIwAAACIAAAD9AAEAOAABAA==
            """
            |> String.replace("\n", "")
          )

  @pair Base.decode64!(
          """
          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
          """
          |> String.replace("\n", "")
        )

  @glslang System.get_env("GLSLANG") || System.find_executable("glslangValidator") ||
             System.find_executable("glslang")
  @spirv_val System.get_env("SPIRV_VAL") || System.find_executable("spirv-val")

  defp export_single(
         input \\ Nx.template({4}, :s32),
         output \\ Nx.template({4}, :s32),
         opts \\ [workgroups: {4, 1, 1}]
       ) do
    NxIREE.Vulkan.export(@single, [input], output, opts)
  end

  defp exported(binary), do: NxIREE.Exported.decode(binary)

  defp shader_directories do
    Path.wildcard(Path.join(System.tmp_dir!(), "nx_iree-vulkan-#{System.pid()}-*"))
    |> Enum.sort()
  end

  test "exports a static shader in the normal format without callback inputs" do
    before = shader_directories()
    data = export_single() |> exported()
    assert data.format == NxIREE.Exported.format()
    assert is_binary(data.bytecode) and byte_size(data.bytecode) > 0
    assert data.inputs == [{{4}, {:s, 32}}]
    assert data.used_inputs == [0]
    assert data.callback_pid_size == 0
    assert data.output == Nx.template({4}, :s32)
    assert data.written_inputs == []
    assert shader_directories() == before
  end

  test "external resource mapping is an explicit boolean allocation policy" do
    default = export_single() |> exported()
    refute "--iree-hal-external-resources-mappable=true" in default.compilation_flags

    mapped =
      export_single(Nx.template({4}, :s32), Nx.template({4}, :s32),
        workgroups: {4, 1, 1},
        external_resources_mappable: true
      )
      |> exported()

    assert "--iree-hal-external-resources-mappable=true" in mapped.compilation_flags

    for invalid <- [nil, "true", 1] do
      assert_raise ArgumentError, ~r/:external_resources_mappable must be a boolean/, fn ->
        export_single(Nx.template({4}, :s32), Nx.template({4}, :s32),
          workgroups: {4, 1, 1},
          external_resources_mappable: invalid
        )
      end
    end
  end

  test "supports scalar buffers and tuple outputs with signed, unsigned and floating types" do
    scalar = export_single(Nx.template({}, :s32), Nx.template({}, :s32), workgroups: {1, 1, 1})
    assert exported(scalar).inputs == [{{}, {:s, 32}}]

    outputs = {Nx.template({4}, :f32), Nx.template({4}, :u32)}
    pair = NxIREE.Vulkan.export(@pair, [Nx.template({4}, :f32)], outputs, workgroups: {4, 1, 1})
    assert exported(pair).output == outputs
  end

  test "rejects invalid tensors and containers before compilation" do
    for bad <- [
          :not_a_tensor,
          Nx.template({4}, :f64),
          Nx.vectorize(Nx.template({2, 4}, :s32), :batch)
        ] do
      assert_raise ArgumentError, ~r/input 0 must/, fn -> export_single(bad) end
    end

    assert_raise ArgumentError, ~r/static shape/, fn ->
      export_single(%{Nx.template({4}, :s32) | shape: {:dynamic}})
    end

    assert_raise ArgumentError, ~r/flat list/, fn ->
      apply(NxIREE.Vulkan, :export, [
        @single,
        Nx.template({4}, :s32),
        Nx.template({4}, :s32),
        [workgroups: {4, 1, 1}]
      ])
    end

    for bad <- [{}, [], {Nx.template({4}, :s32), :bad}, Nx.template({4}, :s64)] do
      assert_raise ArgumentError, ~r/output/, fn -> export_single(Nx.template({4}, :s32), bad) end
    end
  end

  test "requires bounded static workgroups and validates options and target features" do
    for counts <- [nil, {1, 1}, {0, 1, 1}, {65536, 1, 1}, {1.0, 1, 1}] do
      assert_raise ArgumentError, ~r/:workgroups must/, fn ->
        export_single(Nx.template({4}, :s32), Nx.template({4}, :s32), workgroups: counts)
      end
    end

    assert_raise ArgumentError, ~r/keyword list/, fn ->
      apply(NxIREE.Vulkan, :export, [@single, [], Nx.template({4}, :s32), nil])
    end

    assert_raise ArgumentError, ~r/unknown keys/, fn ->
      export_single(Nx.template({4}, :s32), Nx.template({4}, :s32),
        workgroups: {4, 1, 1},
        entry_point: "other"
      )
    end

    for features <- ["spirv:v1.3", "spirv:v1.3,cap:Shader\"\n}>", ["cap:Shader"]] do
      assert_raise ArgumentError, ~r/:target_features must/, fn ->
        export_single(Nx.template({4}, :s32), Nx.template({4}, :s32),
          workgroups: {4, 1, 1},
          target_features: features
        )
      end
    end
  end

  test "rejects malformed SPIR-V and unsupported entry points" do
    for binary <- [nil, "", "not a shader", binary_part(@single, 0, byte_size(@single) - 1)] do
      assert_raise ArgumentError, ~r/little-endian SPIR-V/, fn ->
        NxIREE.Vulkan.export(binary, [], Nx.template({4}, :s32), workgroups: {4, 1, 1})
      end
    end

    <<header::binary-size(20), _::binary>> = @single

    assert_raise ArgumentError, ~r/zero-length instruction/, fn ->
      NxIREE.Vulkan.export(header <> <<0::32>>, [], Nx.template({4}, :s32), workgroups: {4, 1, 1})
    end

    without_main = :binary.replace(@single, "main", "nope", [:global])

    assert_raise ArgumentError, ~r/GLCompute entry point named main/, fn ->
      NxIREE.Vulkan.export(without_main, [], Nx.template({4}, :s32), workgroups: {4, 1, 1})
    end

    <<magic::binary-size(4), _version::binary-size(4), rest::binary>> = @single
    newer = magic <> <<0x00010500::little-32>> <> rest

    assert_raise ArgumentError, ~r/exceeds the declared target/, fn ->
      NxIREE.Vulkan.export(newer, [], Nx.template({4}, :s32), workgroups: {4, 1, 1})
    end
  end

  test "concurrent exports use separate shader files and clean them" do
    before = shader_directories()

    Enum.each(Task.async_stream(1..4, fn _ -> export_single() end), fn result ->
      assert {:ok, binary} = result
      assert byte_size(exported(binary).bytecode) > 0
    end)

    assert shader_directories() == before
  end

  @tag :tmp_dir
  @tag skip: is_nil(@glslang) or is_nil(@spirv_val)
  test "embedded shader fixtures validate and their GLSL sources regenerate", %{tmp_dir: dir} do
    for {name, source, spirv} <- [
          {"single", @single_source, @single},
          {"pair", @pair_source, @pair}
        ] do
      source_path = Path.join(dir, name <> ".comp")
      binary_path = Path.join(dir, name <> ".spv")
      File.write!(source_path, source)
      File.write!(binary_path, spirv)

      {log, status} =
        System.cmd(@spirv_val, ["--target-env", "vulkan1.1", binary_path], stderr_to_stdout: true)

      assert status == 0, log

      {log, status} =
        System.cmd(
          @glslang,
          ["-V", "--target-env", "vulkan1.1", "-S", "comp", source_path, "-o", binary_path],
          stderr_to_stdout: true
        )

      assert status == 0, log

      {log, status} =
        System.cmd(@spirv_val, ["--target-env", "vulkan1.1", binary_path], stderr_to_stdout: true)

      assert status == 0, log
    end
  end

  @tag :vulkan
  test "one pipeline runs dependent dispatches and retains returned intermediates" do
    tensor = Nx.template({4}, :s32)

    stage = fn input, output ->
      [spirv: @single, inputs: [input], outputs: [{output, tensor}], workgroups: {4, 1, 1}]
    end

    before = shader_directories()

    binary =
      NxIREE.Vulkan.export_pipeline(
        [input: tensor],
        [stage.(:input, :intermediate), stage.(:intermediate, :result)],
        {:result, :intermediate, :input},
        external_resources_mappable: true
      )

    assert shader_directories() == before
    assert exported(binary).written_inputs == []
    {:ok, devices} = NxIREE.list_devices("vulkan")
    match = System.get_env("NX_IREE_DEFAULT_DEVICE", "")
    devices = Enum.filter(devices, &String.contains?(&1.uri, match))
    assert devices != []

    for device <- devices do
      model = NxIREE.load_exported(binary, device: device)

      for values <- [[1, 2, 3, 4], [-3, 0, 10, 20]] do
        input =
          Nx.tensor(values, type: :s32, backend: Nx.BinaryBackend)
          |> Nx.backend_copy({NxIREE.Backend, device: device})

        try do
          results = NxIREE.run(model, [input])

          try do
            {result, intermediate, returned_input} = results
            assert Nx.to_flat_list(result) == Enum.map(values, &(&1 * 9 + 28))
            assert Nx.to_flat_list(intermediate) == Enum.map(values, &(&1 * 3 + 7))
            assert Nx.to_flat_list(returned_input) == values
          after
            for result <- Tuple.to_list(results), do: Nx.backend_deallocate(result)
          end

          assert Nx.to_flat_list(input) == values
        after
          Nx.backend_deallocate(input)
        end
      end
    end
  end

  test "pipeline validates named immutable dataflow before compilation" do
    tensor = Nx.template({4}, :s32)
    stage = [spirv: @single, inputs: [:input], outputs: [result: tensor], workgroups: {4, 1, 1}]

    for {inputs, stages, results, message} <- [
          {[input: tensor, input: tensor], [stage], :result, ~r/unique names/},
          {[input: tensor], [], :result, ~r/nonempty list/},
          {[input: tensor], [Keyword.put(stage, :inputs, [:future])], :result,
           ~r/undefined buffer/},
          {[input: tensor], [Keyword.put(stage, :outputs, input: tensor)], :input,
           ~r/already defined/},
          {[input: tensor], [stage, stage], :result, ~r/already defined/},
          {[input: tensor], [stage], :missing, ~r/undefined buffer/},
          {[input: tensor], [stage], {}, ~r/nonempty flat tuple/},
          {[input: tensor], [Keyword.put(stage, :outputs, [])], :result, ~r/nonempty/},
          {[input: tensor], [Keyword.put(stage, :workgroups, {0, 1, 1})], :result,
           ~r/workgroups/},
          {[input: tensor], [Keyword.put(stage, :spirv, "bad")], :result, ~r/SPIR-V/}
        ] do
      assert_raise ArgumentError, message, fn ->
        NxIREE.Vulkan.export_pipeline(inputs, stages, results)
      end
    end
  end

  @tag :vulkan
  test "normal load and run reuse the shaders and preserve resident inputs" do
    {:ok, devices} = NxIREE.list_devices("vulkan")
    match = System.get_env("NX_IREE_DEFAULT_DEVICE", "")
    devices = Enum.filter(devices, &String.contains?(&1.uri, match))
    assert devices != [], "no Vulkan device matches #{inspect(match)}"

    for device <- devices do
      single = NxIREE.load_exported(export_single(), device: device)
      outputs = {Nx.template({4}, :f32), Nx.template({4}, :u32)}

      pair =
        NxIREE.Vulkan.export(@pair, [Nx.template({4}, :f32)], outputs, workgroups: {4, 1, 1})
        |> NxIREE.load_exported(device: device)

      for values <- [[1, 2, 3, 4], [-3, 0, 10, 20]] do
        input =
          Nx.tensor(values, type: :s32, backend: Nx.BinaryBackend)
          |> Nx.backend_copy({NxIREE.Backend, device: device})

        try do
          result = NxIREE.run(single, [input]) |> Nx.backend_transfer(Nx.BinaryBackend)
          assert Nx.to_flat_list(result) == Enum.map(values, &(&1 * 3 + 7))
          assert Nx.to_flat_list(input) == values
        after
          Nx.backend_deallocate(input)
        end

        float_input = Nx.tensor(values, type: :f32, backend: Nx.BinaryBackend)
        {doubled, indices} = NxIREE.run(pair, [float_input])
        doubled = Nx.backend_transfer(doubled, Nx.BinaryBackend)
        indices = Nx.backend_transfer(indices, Nx.BinaryBackend)
        assert Nx.to_flat_list(doubled) == Enum.map(values, &(&1 * 2.0))
        assert Nx.to_flat_list(indices) == [0, 1, 2, 3]
      end
    end
  end
end
