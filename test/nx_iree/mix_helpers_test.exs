defmodule NxIREE.MixHelpersTest do
  use ExUnit.Case, async: true

  setup do
    root = Path.join(System.tmp_dir!(), "nx_iree_compiler_#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf!(root) end)
    %{root: root, priv: Path.join(root, "priv")}
  end

  test "switches an existing compiler link and accepts repeated selection", %{
    root: root,
    priv: priv
  } do
    first = Path.join(root, "first")
    second = Path.join(root, "second")

    for dir <- [first, second] do
      File.mkdir_p!(dir)
      File.write!(Path.join(dir, "iree-compile"), "compiler")
    end

    assert :ok = NxIREE.MixHelpers.link_compiler!(first, priv)
    assert :ok = NxIREE.MixHelpers.link_compiler!(second, priv)
    assert :ok = NxIREE.MixHelpers.link_compiler!(second, priv)
    assert File.read_link!(Path.join(priv, "iree-compile")) == Path.join(second, "iree-compile")
    assert File.read!(Path.join(first, "iree-compile")) == "compiler"
  end

  test "accepts the wheel directory layout", %{root: root, priv: priv} do
    dir = Path.join([root, "compiler", "_mlir_libs"])
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "iree-compile"), "compiler")

    assert :ok = NxIREE.MixHelpers.link_compiler!(root, priv)
    assert File.read_link!(Path.join(priv, "iree-compile")) == Path.join(dir, "iree-compile")
  end

  test "a missing compiler leaves the selected compiler intact", %{root: root, priv: priv} do
    File.mkdir_p!(priv)
    link = Path.join(priv, "iree-compile")
    File.write!(link, "existing compiler")

    assert_raise RuntimeError, ~r/iree-compile not found in NX_IREE_COMPILER_DIR=/, fn ->
      NxIREE.MixHelpers.link_compiler!(Path.join(root, "missing"), priv)
    end

    assert File.read!(link) == "existing compiler"
  end
end
