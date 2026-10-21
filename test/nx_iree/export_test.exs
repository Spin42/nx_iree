defmodule NxIREE.ExportTest do
  use ExUnit.Case

  @cpu_flags ["--iree-hal-target-backends=llvm-cpu", "--iree-input-type=stablehlo_xla"]

  defp cpu, do: NxIREE.Device.find_default_device("local-sync")
  defp host(t), do: Nx.backend_transfer(t, Nx.BinaryBackend)

  defp roundtrip(fun, args, flags \\ @cpu_flags, device \\ nil) do
    binary = NxIREE.export(fun, args, iree_compiler_flags: flags)
    NxIREE.load_exported(binary, device: device || cpu())
  end

  test "runs a function compiled ahead of time" do
    a = Nx.iota({3, 4}, type: :f32)
    b = Nx.iota({4, 2}, type: :f32)
    model = roundtrip(&Nx.dot/2, [Nx.template({3, 4}, :f32), Nx.template({4, 2}, :f32)])

    assert host(NxIREE.run(model, [a, b])) |> Nx.to_flat_list() ==
             Nx.dot(a, b) |> Nx.to_flat_list()
  end

  test "is reusable and accepts its own outputs" do
    model = roundtrip(&Nx.add(&1, 1.0), [Nx.template({2}, :f32)])
    out = Enum.reduce(1..5, Nx.tensor([0.0, 10.0]), fn _, acc -> NxIREE.run(model, [acc]) end)
    assert host(out) |> Nx.to_flat_list() == [5.0, 15.0]
  end

  test "skips arguments the function does not use" do
    model =
      roundtrip(fn a, _unused, c -> Nx.multiply(a, c) end, [
        Nx.template({2}, :f32),
        Nx.template({7}, :s32),
        Nx.template({2}, :f32)
      ])

    result = NxIREE.run(model, [Nx.tensor([2.0, 3.0]), Nx.iota({7}), Nx.tensor([4.0, 5.0])])
    assert host(result) |> Nx.to_flat_list() == [8.0, 15.0]
  end

  test "takes containers and returns containers" do
    fun = fn {a, b}, %{"scale" => s} -> {Nx.add(a, b) |> Nx.multiply(s), Nx.subtract(a, b)} end
    args = [{Nx.template({2}, :f32), Nx.template({2}, :f32)}, %{"scale" => Nx.template({}, :f32)}]
    model = roundtrip(fun, args)

    {sum, diff} =
      NxIREE.run(model, [{Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 5.0])}, %{"scale" => 2.0}])

    assert host(sum) |> Nx.to_flat_list() == [8.0, 14.0]
    assert host(diff) |> Nx.to_flat_list() == [-2.0, -3.0]
  end

  test "survives a file round trip" do
    path =
      Path.join(
        System.tmp_dir!(),
        "nx_iree_export_test_#{System.unique_integer([:positive])}.bin"
      )

    File.write!(
      path,
      NxIREE.export(&Nx.exp/1, [Nx.template({3}, :f32)], iree_compiler_flags: @cpu_flags)
    )

    try do
      model = NxIREE.load_exported(File.read!(path), device: cpu())
      result = NxIREE.run(model, [Nx.tensor([0.0, 1.0, 2.0])]) |> host() |> Nx.to_flat_list()
      assert_in_delta Enum.at(result, 2), :math.exp(2.0), 1.0e-5
    after
      File.rm(path)
    end
  end

  test "runs on Vulkan" do
    {:ok, devices} = NxIREE.list_devices("vulkan")

    for device <- devices do
      flags = ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla"]
      model = roundtrip(&Nx.multiply(&1, &1), [Nx.template({4}, :f32)], flags, device)

      assert host(NxIREE.run(model, [Nx.tensor([1.0, 2.0, 3.0, 4.0])])) |> Nx.to_flat_list() == [
               1.0,
               4.0,
               9.0,
               16.0
             ]
    end
  end

  test "rejects inputs that do not match the export" do
    model = roundtrip(&Nx.exp/1, [Nx.template({3}, :f32)])

    assert_raise ArgumentError, ~r/input 0 must be \{:f, 32\} \{3\}/, fn ->
      NxIREE.run(model, [Nx.tensor([1.0, 2.0])])
    end

    assert_raise ArgumentError, ~r/expected 1 input tensors, got 2/, fn ->
      NxIREE.run(model, [Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([1.0])])
    end
  end

  test "reports compiler errors" do
    # a 3x3 kernel with stride 2 and padding 1 on a 2x2 input fails in IREE 3.13
    model = fn x, w ->
      Nx.conv(x, w,
        strides: [2, 2],
        padding: [{1, 1}, {1, 1}],
        input_permutation: [0, 3, 1, 2],
        kernel_permutation: [3, 2, 0, 1],
        output_permutation: [0, 3, 1, 2]
      )
    end

    error =
      assert_raise NxIREE.CompileError, fn ->
        NxIREE.export(model, [Nx.template({1, 2, 2, 4}, :f32), Nx.template({3, 3, 4, 8}, :f32)],
          iree_compiler_flags: @cpu_flags
        )
      end

    assert error.message =~ "iree-compile exited with status"
    assert error.message =~ "error:"
  end

  test "needs an explicit target backend" do
    assert_raise ArgumentError, ~r/--iree-hal-target-backends/, fn ->
      NxIREE.export(&Nx.exp/1, [Nx.template({3}, :f32)], iree_compiler_flags: [])
    end
  end

  test "rejects binaries that are not exports" do
    assert_raise ArgumentError, ~r/not a binary produced by NxIREE.export/, fn ->
      NxIREE.load_exported(:erlang.term_to_binary(%{hello: :world}))
    end

    assert_raise ArgumentError, ~r/unsupported export format/, fn ->
      NxIREE.load_exported(:erlang.term_to_binary(%{format: 99}))
    end
  end
end
