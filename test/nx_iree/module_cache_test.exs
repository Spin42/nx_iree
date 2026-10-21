defmodule NxIREE.ModuleCacheTest.Loop do
  import Nx.Defn

  defn run(x) do
    y = x * 2

    {_i, acc} =
      while {i = 0, acc = y + y}, i < 3 do
        {i + 1, acc + 1}
      end

    acc
  end
end

defmodule NxIREE.ModuleCacheTest do
  use ExUnit.Case

  @flags ["--iree-input-type=stablehlo_xla", "--iree-execution-model=async-internal"]

  setup do
    dir = Path.join(System.tmp_dir!(), "nx_iree_cache_test_#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf!(dir) end)
    %{dir: dir, cpu: NxIREE.Device.find_default_device("local-sync")}
  end

  defp opts(dir, cpu, flags \\ @flags) do
    [
      compiler: NxIREE.Compiler,
      iree_compiler_flags: flags,
      iree_runtime_options: [device: cpu],
      iree_module_cache: dir
    ]
  end

  defp key(fun, templates, flags \\ @flags) do
    NxIREE.ModuleCache.key(Nx.Defn.debug_expr_apply(fun, templates), templates, flags)
  end

  defp host(t), do: t |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

  describe "keys" do
    test "are stable across builds of the same function" do
      t = Nx.template({3}, :f32)

      fun = fn x ->
        x |> Nx.exp() |> Nx.add(Nx.tensor([1.0, 2.0, 3.0], backend: Nx.BinaryBackend)) |> Nx.sum()
      end

      assert key(fun, [t]) == key(fun, [t])
    end

    test "depend on shapes, constants and flags but not flag order" do
      t = Nx.template({3}, :f32)
      add = fn c -> fn x -> Nx.add(x, Nx.tensor(c, backend: Nx.BinaryBackend)) end end
      base = key(add.([1.0, 2.0, 3.0]), [t])

      refute key(add.([1.0, 2.0, 3.0]), [Nx.template({2, 3}, :f32)]) == base
      refute key(add.([1.0, 2.0, 4.0]), [t]) == base
      refute key(add.([1.0, 2.0, 3.0]), [t], ["--iree-vulkan-target=pascal" | @flags]) == base
      assert key(add.([1.0, 2.0, 3.0]), [t], Enum.reverse(@flags)) == base
    end

    test "follow shared subexpressions and while loops" do
      t = Nx.template({4}, :f32)

      assert key(&NxIREE.ModuleCacheTest.Loop.run/1, [t]) ==
               key(&NxIREE.ModuleCacheTest.Loop.run/1, [t])
    end
  end

  test "a compiled function is stored and found again", %{dir: dir, cpu: cpu} do
    x = Nx.tensor([0.0, 1.0])
    exp = Nx.Defn.jit(&Nx.exp/1, opts(dir, cpu))
    assert host(exp.(x)) == host(Nx.exp(Nx.backend_copy(x, Nx.BinaryBackend)))
    assert [entry] = Path.wildcard(Path.join(dir, "*.nxiree"))

    # Swap in another function's bytecode: a new compile of exp (a distinct
    # closure, so not the in-memory jit cache) now runs it,
    # so the module came from the cache, not the compiler
    neg_flags =
      @flags ++ ["--iree-hal-target-backends=llvm-cpu", "--iree-llvmcpu-target-cpu=host"]

    {:ok, neg} =
      NxIREE.Compiler.to_bytecode(&Nx.negate/1, [Nx.template({2}, :f32)],
        iree_compiler_flags: neg_flags
      )

    {:ok, {_module, pid_size}} = NxIREE.ModuleCache.read(entry)
    NxIREE.ModuleCache.write!(entry, neg, pid_size)

    exp_again = Nx.Defn.jit(fn t -> Nx.exp(t) end, opts(dir, cpu))
    assert host(exp_again.(x)) == [-0.0, -1.0]
  end

  test "a cached entry in an older format is recompiled", %{dir: dir, cpu: cpu} do
    f = Nx.Defn.jit(&Nx.multiply(&1, 3), opts(dir, cpu))
    assert host(f.(Nx.tensor([1.0]))) == [3.0]
    [entry] = Path.wildcard(Path.join(dir, "*.nxiree"))
    File.write!(entry, :erlang.term_to_binary(%{format: 0}))

    f = Nx.Defn.jit(&Nx.multiply(&1, 3), opts(dir, cpu))
    assert host(f.(Nx.tensor([2.0]))) == [6.0]
    assert {:ok, _} = NxIREE.ModuleCache.read(entry)
  end
end
