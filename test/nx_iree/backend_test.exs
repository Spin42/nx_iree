defmodule NxIREE.BackendTest do
  use ExUnit.Case

  @flags ["--iree-input-type=stablehlo_xla", "--iree-execution-model=async-internal"]

  defp cpu, do: NxIREE.Device.find_default_device("local-sync")

  defp jit(fun, device \\ cpu()) do
    Nx.Defn.jit(fun,
      compiler: NxIREE.Compiler,
      iree_compiler_flags: @flags,
      iree_runtime_options: [device: device]
    )
  end

  defp host(t), do: Nx.backend_transfer(t, Nx.BinaryBackend)

  defp assert_same(actual, expected) do
    assert {Nx.type(actual), Nx.shape(actual), Nx.to_flat_list(actual)} ==
             {Nx.type(expected), Nx.shape(expected), Nx.to_flat_list(expected)}
  end

  describe "constants" do
    test "keep the requested type" do
      for type <- [:f32, :f64, :s32, :u8] do
        t = Nx.tensor(1, type: type, backend: {NxIREE.Backend, device: cpu()})
        assert t |> host() |> Nx.to_number() == 1
        assert Nx.type(t) == Nx.Type.normalize!(type)
      end
    end

    test "fill the requested shape" do
      t = Nx.broadcast(Nx.tensor(2.5, backend: {NxIREE.Backend, device: cpu()}), {2, 3})
      assert_same(host(t), Nx.broadcast(Nx.tensor(2.5), {2, 3}))
    end

    test "feed computations with their value" do
      t = Nx.tensor(4.0, backend: {NxIREE.Backend, device: cpu()})
      assert jit(&Nx.sqrt/1).(t) |> host() |> Nx.to_number() == 2.0
    end
  end

  describe "devices" do
    test "unknown devices raise with the available ones" do
      assert_raise ArgumentError, ~r/unknown IREE device "nope:\/\/".*local-sync:\/\//, fn ->
        Nx.backend_transfer(Nx.iota({2}), {NxIREE.Backend, device: "nope://"})
      end
    end

    test "backend_transfer honours :device" do
      device = cpu()
      t = Nx.backend_transfer(Nx.iota({4}), {NxIREE.Backend, device: device})
      assert t.data.device == device.ref
    end

    test "inputs from another device are copied" do
      {:ok, devices} = NxIREE.list_devices("local-task")
      other = hd(devices)
      t = Nx.backend_transfer(Nx.tensor([1.0, 4.0, 9.0]), {NxIREE.Backend, device: other})
      assert_same(jit(&Nx.sqrt/1).(t) |> host(), Nx.tensor([1.0, 2.0, 3.0]))
    end
  end

  describe "f64" do
    test "stays f64 on the CPU backend" do
      x = Nx.tensor([0.5, 2.0], type: :f64, backend: Nx.BinaryBackend)
      result = jit(&Nx.exp/1).(x) |> host()
      assert Nx.type(result) == {:f, 64}
      for {got, want} <- Enum.zip(Nx.to_flat_list(result), [:math.exp(0.5), :math.exp(2.0)]) do
        assert_in_delta got, want, 1.0e-12
      end
    end

    test "is rejected with a clear error where IREE demotes it" do
      fun =
        Nx.Defn.jit(&Nx.exp/1,
          compiler: NxIREE.Compiler,
          iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv" | @flags],
          iree_runtime_options: [device: cpu()]
        )

      assert_raise ArgumentError, ~r/f64 tensors are not supported/, fn ->
        fun.(Nx.tensor([1.0], type: :f64))
      end
    end
  end

  describe "compiled functions" do
    test "are compiled once per input template and reused" do
      f = jit(fn a, b -> Nx.dot(a, b) end)
      a = Nx.iota({8, 8}, type: :f32)

      for _ <- 1..20 do
        assert_same(f.(a, a) |> host(), Nx.dot(a, a))
      end

      # a new shape compiles a new function
      b = Nx.iota({4, 4}, type: :f32)
      assert_same(f.(b, b) |> host(), Nx.dot(b, b))
    end

    test "outputs can be fed back as inputs" do
      f = jit(&Nx.add(&1, 1))
      out = Enum.reduce(1..10, Nx.tensor([0.0, 1.0]), fn _, acc -> f.(acc) end)
      assert_same(host(out), Nx.tensor([10.0, 11.0]))
    end
  end

  describe "deallocated tensors" do
    test "raise when read again" do
      t = Nx.backend_transfer(Nx.iota({4}), {NxIREE.Backend, device: cpu()})
      _ = Nx.backend_transfer(t, Nx.BinaryBackend)
      assert_raise MatchError, fn -> Nx.backend_transfer(t, Nx.BinaryBackend) end
    end

    test "raise when passed to a computation" do
      t = Nx.backend_transfer(Nx.iota({4}, type: :f32), {NxIREE.Backend, device: cpu()})
      _ = Nx.backend_transfer(t, Nx.BinaryBackend)
      assert_raise RuntimeError, ~r/deallocated/, fn -> jit(&Nx.exp/1).(t) end
    end
  end

  test "indexed_put with a list of indices" do
    t = Nx.iota({4}, type: :f32)
    idx = Nx.tensor([[0], [2]])
    upd = Nx.tensor([10.0, 20.0])
    result = jit(&Nx.indexed_put/3).(t, idx, upd) |> host()
    assert_same(result, Nx.tensor([10.0, 1.0, 20.0, 3.0]))
  end
end
