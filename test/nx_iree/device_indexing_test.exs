defmodule NxIREE.DeviceIndexingTest.Defns do
  import Nx.Defn

  defn(slice2(t, i, j), do: Nx.slice(t, [i, j], [2, 3]))
  defn(slice_mixed(t, j), do: Nx.slice(t, [1, j], [2, 2]))
  defn(slice_strided(t, j), do: Nx.slice_along_axis(t, j, 4, axis: 1, strides: 2))

  defn(put2(t, u, i, j), do: Nx.put_slice(t, [i, j], u))
  defn(put_row(t, u, i), do: Nx.put_slice(t, [i, 0], u))

  # Writes one row per step at a moving position, like a KV cache
  defn fill(cache, rows) do
    {_i, cache, _rows} =
      while {i = 0, cache, rows}, i < 3 do
        row = Nx.slice_along_axis(rows, i, 1, axis: 0)
        {i + 1, Nx.put_slice(cache, [i + 1, 0], row), rows}
      end

    cache
  end
end

defmodule NxIREE.DeviceIndexingTest do
  use ExUnit.Case

  alias NxIREE.DeviceIndexingTest.Defns

  @flags ["--iree-input-type=stablehlo_xla", "--iree-execution-model=async-internal"]

  defp iree(fun, args) do
    opts = [
      compiler: NxIREE.Compiler,
      iree_compiler_flags: @flags,
      iree_runtime_options: [device: NxIREE.Device.find_default_device("local-sync")],
      iree_device_indexing: true
    ]

    fun |> Nx.Defn.jit_apply(args, opts) |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  defp reference(fun, args) do
    args = Enum.map(args, &Nx.backend_copy(&1, Nx.BinaryBackend))

    fun
    |> Nx.Defn.jit_apply(args, compiler: Nx.Defn.Evaluator)
    |> Nx.backend_copy(Nx.BinaryBackend)
  end

  defp assert_same(fun, args) do
    want = reference(fun, args)
    got = iree(fun, args)

    assert {got.type, got.shape, Nx.to_flat_list(got)} ==
             {want.type, want.shape, Nx.to_flat_list(want)}
  end

  defp grid(type \\ :f32), do: Nx.iota({4, 6}, type: type, backend: Nx.BinaryBackend)
  defp s(n), do: Nx.tensor(n, type: :s32, backend: Nx.BinaryBackend)

  describe "dynamic slices" do
    test "in range, clamped high and negative starts" do
      for {i, j} <- [{1, 2}, {3, 5}, {-2, -1}, {0, 0}] do
        assert_same(&Defns.slice2/3, [grid(), s(i), s(j)])
      end
    end

    test "with static and strided axes" do
      for j <- [0, 1, 3, 9] do
        assert_same(&Defns.slice_mixed/2, [grid(), s(j)])
        assert_same(&Defns.slice_strided/2, [grid(), s(j)])
      end
    end
  end

  describe "dynamic slice updates" do
    test "in range, clamped high and negative starts" do
      u = Nx.multiply(Nx.iota({2, 3}, backend: Nx.BinaryBackend), -1.0)

      for {i, j} <- [{1, 2}, {3, 5}, {-2, -1}, {0, 0}] do
        assert_same(&Defns.put2/4, [grid(), u, s(i), s(j)])
      end
    end

    test "full-width rows and promoted types" do
      u = Nx.tensor([[100.5, 101.5, 102.5, 103.5, 104.5, 105.5]], backend: Nx.BinaryBackend)

      for i <- [0, 2, 7] do
        assert_same(&Defns.put_row/3, [grid(:s32), u, s(i)])
      end
    end

    test "in a loop at a moving position" do
      cache = Nx.broadcast(Nx.tensor(0.0, backend: Nx.BinaryBackend), {5, 3})
      rows = Nx.iota({3, 3}, type: :f32, backend: Nx.BinaryBackend)
      assert_same(&Defns.fill/2, [cache, rows])
    end
  end

  test "leaves no slice with tensor start indices" do
    expr =
      Nx.Defn.debug_expr_apply(&Defns.fill/2, [
        Nx.template({5, 3}, :f32),
        Nx.template({3, 3}, :f32)
      ])

    {_, ops} =
      Nx.Defn.Composite.traverse(NxIREE.Compiler.DeviceIndexing.rewrite(expr), [], &collect/2)

    assert ops != []

    refute Enum.any?(ops, fn {op, starts} ->
             op in [:slice, :put_slice] and Enum.any?(starts, &(not is_integer(&1)))
           end)
  end

  defp collect(%Nx.Tensor{data: %Nx.Defn.Expr{op: op, args: args}} = t, acc) do
    starts =
      case {op, args} do
        {:slice, [_, starts | _]} -> starts
        {:put_slice, [_, starts, _]} -> starts
        _ -> []
      end

    {_, acc} = Nx.Defn.Tree.apply_args(t, :all, [{op, starts} | acc], &collect/2)
    {t, acc}
  end

  defp collect(other, acc), do: {other, acc}
end
