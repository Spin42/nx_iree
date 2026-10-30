defmodule NxIREE.Vulkan.KernelTest do
  use ExUnit.Case

  alias NxIREE.TestShaders

  # out[i] = in[i] * 3 + 7
  defp single(x) do
    kernel = %NxIREE.Vulkan.Kernel{spirv: TestShaders.single(), workgroups: {4, 1, 1}}

    Nx.block(kernel, [x], Nx.template({4}, :s32), fn _kernel, x ->
      x |> Nx.multiply(3) |> Nx.add(7)
    end)
  end

  # {in[i] * 2.0, i}
  defp pair(x) do
    kernel = %NxIREE.Vulkan.Kernel{spirv: TestShaders.pair(), workgroups: {4, 1, 1}}
    outputs = {Nx.template({4}, :f32), Nx.template({4}, :u32)}

    Nx.block(kernel, [x], outputs, fn _kernel, x ->
      {Nx.multiply(x, 2), Nx.iota({4}, type: :u32)}
    end)
  end

  # A default that differs from the shader, to tell which one ran
  defp single_with_other_default(x) do
    kernel = %NxIREE.Vulkan.Kernel{spirv: TestShaders.single(), workgroups: {4, 1, 1}}

    Nx.block(kernel, [x], Nx.template({4}, :s32), fn _kernel, x ->
      x |> Nx.multiply(3) |> Nx.add(8)
    end)
  end

  @input Nx.tensor([1, 2, 3, -4], type: :s32, backend: Nx.BinaryBackend)

  defp values(container) do
    [container] |> Nx.Defn.Composite.flatten_list() |> Enum.map(&Nx.to_flat_list/1)
  end

  test "other backends run the default implementation" do
    expected = [[10, 13, 16, -5]]

    assert values(Nx.Defn.jit(&single/1, compiler: Nx.Defn.Evaluator).(@input)) == expected
    assert values(Nx.Defn.jit(&single/1).(@input)) == expected

    if Code.ensure_loaded?(EXLA) do
      assert values(Nx.Defn.jit(&single/1, compiler: EXLA).(@input)) == expected
    end
  end

  test "a block with more arguments than the function runs" do
    sum_of_three = fn x ->
      kernel = %NxIREE.Vulkan.Kernel{spirv: TestShaders.single(), workgroups: {4, 1, 1}}

      Nx.block(kernel, [x, Nx.multiply(x, 2), Nx.add(x, 1)], Nx.template({4}, :s32), fn
        _kernel, a, b, c -> a |> Nx.add(b) |> Nx.add(c)
      end)
    end

    assert values(Nx.Defn.jit(sum_of_three).(@input)) == [[5, 9, 13, -15]]
  end

  test "Vulkan compilation validates the kernel" do
    bad = fn x ->
      kernel = %NxIREE.Vulkan.Kernel{spirv: "not spir-v", workgroups: {4, 1, 1}}
      Nx.block(kernel, [x], Nx.template({4}, :s32), fn _kernel, x -> x end)
    end

    assert_raise ArgumentError, ~r/must be a little-endian SPIR-V binary/, fn ->
      NxIREE.Compiler.to_bytecode(bad, [Nx.template({4}, :s32)],
        iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv"]
      )
    end
  end

  test "pipelines validate their dataflow" do
    kernel = %NxIREE.Vulkan.Kernel{spirv: TestShaders.single(), workgroups: {4, 1, 1}}
    t = Nx.template({4}, :s32)
    flags = [iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv"]]

    compile = fn pipeline ->
      fun = fn x -> Nx.block(pipeline, [x], t, fn _, x -> x end) end
      NxIREE.Compiler.to_bytecode(fun, [t], flags)
    end

    assert_raise ArgumentError, ~r/reads :missing before it is defined/, fn ->
      compile.(%NxIREE.Vulkan.Pipeline{
        inputs: [:x],
        stages: [{kernel, [:missing], [y: t]}],
        outputs: [:y]
      })
    end

    assert_raise ArgumentError, ~r/defines :x twice/, fn ->
      compile.(%NxIREE.Vulkan.Pipeline{
        inputs: [:x],
        stages: [{kernel, [:x], [x: t]}],
        outputs: [:x]
      })
    end

    assert_raise ArgumentError, ~r/is not written by any stage/, fn ->
      compile.(%NxIREE.Vulkan.Pipeline{
        inputs: [:x],
        stages: [{kernel, [:x], [y: t]}],
        outputs: [:x]
      })
    end

    assert_raise ArgumentError, ~r/do not match the block's output/, fn ->
      wrong = Nx.template({4}, :f32)

      compile.(%NxIREE.Vulkan.Pipeline{
        inputs: [:x],
        stages: [{kernel, [:x], [y: wrong]}],
        outputs: [:y]
      })
    end
  end

  test "specialization rejects unknown constants and mistyped values" do
    flags = [iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv"]]
    t = Nx.template({4}, :f32)

    compile = fn specialization ->
      kernel = %NxIREE.Vulkan.Kernel{
        spirv: TestShaders.specialized(),
        workgroups: {4, 1, 1},
        specialization: specialization
      }

      NxIREE.Compiler.to_bytecode(
        fn x -> Nx.block(kernel, [x], t, fn _, x -> x end) end,
        [t],
        flags
      )
    end

    assert_raise ArgumentError, ~r/no specialization constant 7/, fn -> compile.(%{7 => 1}) end

    assert_raise ArgumentError, ~r/takes a 32-bit integer, float or boolean/, fn ->
      compile.(%{0 => 1.5})
    end
  end

  describe "on Vulkan devices" do
    @describetag :vulkan

    setup do
      {:ok, devices} = NxIREE.list_devices("vulkan")
      match = System.get_env("NX_IREE_DEFAULT_DEVICE", "")
      devices = Enum.filter(devices, &String.contains?(&1.uri, match))
      assert devices != []
      {:ok, devices: devices}
    end

    defp jit(fun, device) do
      Nx.Defn.jit(fun, compiler: NxIREE.Compiler, iree_runtime_options: [device: device])
    end

    test "a block runs its shader", %{devices: devices} do
      for device <- devices do
        assert values(jit(&single_with_other_default/1, device).(@input)) == [[10, 13, 16, -5]]
      end
    end

    test "kernels mix with Nx operations and feed each other", %{devices: devices} do
      fun = fn x ->
        y = single(Nx.add(x, 1))
        {doubled, indices} = pair(Nx.as_type(single(y), :f32))
        {Nx.add(doubled, 1), indices, y}
      end

      for device <- devices do
        result = values(jit(fun, device).(@input))

        assert result == values(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(@input))
        assert List.last(result) == [13, 16, 19, -2]
      end
    end

    test "a pipeline passes intermediate tensors between its stages", %{devices: devices} do
      single = %NxIREE.Vulkan.Kernel{spirv: TestShaders.single(), workgroups: {4, 1, 1}}
      pair = %NxIREE.Vulkan.Kernel{spirv: TestShaders.pair(), workgroups: {4, 1, 1}}
      s32 = Nx.template({4}, :s32)
      f32 = Nx.template({4}, :f32)
      u32 = Nx.template({4}, :u32)

      pipeline = %NxIREE.Vulkan.Pipeline{
        inputs: [:x],
        stages: [
          {single, [:x], [once: s32]},
          {single, [:once], [twice: s32]}
        ],
        outputs: [:twice, :once]
      }

      doubled = %NxIREE.Vulkan.Pipeline{
        inputs: [:x],
        stages: [{pair, [:x], [doubled: f32, indices: u32]}],
        outputs: [:indices, :doubled]
      }

      fun = fn x ->
        {twice, once} =
          Nx.block(pipeline, [x], {s32, s32}, fn _, x ->
            once = x |> Nx.multiply(3) |> Nx.add(7)
            {once |> Nx.multiply(3) |> Nx.add(7), once}
          end)

        {indices, doubled} =
          Nx.block(doubled, [Nx.as_type(twice, :f32)], {u32, f32}, fn _, x ->
            {Nx.iota({4}, type: :u32), Nx.multiply(x, 2)}
          end)

        {twice, once, indices, doubled}
      end

      for device <- devices do
        result = values(jit(fun, device).(@input))
        assert result == values(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(@input))
        assert Enum.take(result, 2) == [[37, 46, 55, -8], [10, 13, 16, -5]]
      end
    end

    test "specialization constants replace the shader's defaults", %{devices: devices} do
      t = Nx.template({4}, :f32)
      x = Nx.tensor([1.0, 2.0, 3.0, -4.0], backend: Nx.BinaryBackend)

      specialized = fn specialization ->
        kernel = %NxIREE.Vulkan.Kernel{
          spirv: TestShaders.specialized(),
          workgroups: {4, 1, 1},
          specialization: specialization
        }

        fn x -> Nx.block(kernel, [x], t, fn _, x -> x end) end
      end

      for device <- devices do
        assert values(jit(specialized.(%{}), device).(x)) == [[3.5, 6.5, 9.5, -11.5]]

        assert values(jit(specialized.(%{0 => 10, 1 => -1.0}), device).(x)) == [
                 [9.0, 19.0, 29.0, -41.0]
               ]
      end
    end
  end
end
