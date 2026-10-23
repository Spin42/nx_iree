defmodule NxIREE.ModuleCacheSpecTest do
  use ExUnit.Case, async: true

  @flag "--iree-codegen-transform-dialect-library="

  setup do
    dir = Path.join(System.tmp_dir!(), "nx_iree-spec-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, dir: dir}
  end

  test "tuning specs enter cache keys by contents, not path", %{dir: dir} do
    vars = [Nx.template({2}, :f32)]
    a = Path.join(dir, "a.mlir")
    b = Path.join(dir, "b.mlir")
    File.write!(a, "spec")
    File.write!(b, "spec")
    key = fn path -> NxIREE.ModuleCache.key(:expr, vars, ["--x", @flag <> path]) end

    assert key.(a) == key.(b)
    File.write!(b, "other spec")
    refute key.(a) == key.(b)
  end

  test "a missing spec is reported", %{dir: dir} do
    assert_raise ArgumentError, ~r/cannot read tuning spec/, fn ->
      NxIREE.ModuleCache.key(:expr, [], [@flag <> Path.join(dir, "missing.mlir")])
    end
  end
end
