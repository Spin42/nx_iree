defmodule NxIREE.HostCallsTest.Defns do
  import Nx.Defn

  # Reports each step through a named hook, like Bumblebee's generation loop
  defn steps(x) do
    {_i, x} =
      while {i = 0, x}, i < 3 do
        x = x + 1
        {x, _} = io_call({x, %{step: i}}, :step)
        {i + 1, x}
      end

    x
  end

  defn printed(x) do
    print_value(x * 2)
  end
end

defmodule NxIREE.HostCallsTest do
  use ExUnit.Case

  alias NxIREE.HostCallsTest.Defns

  @flags ["--iree-input-type=stablehlo_xla", "--iree-execution-model=async-internal"]

  defp jit(fun, opts \\ []) do
    Nx.Defn.jit(
      fun,
      [
        compiler: NxIREE.Compiler,
        iree_compiler_flags: @flags,
        iree_runtime_options: [device: NxIREE.Device.find_default_device("local-sync")]
      ] ++ opts
    )
  end

  defp host(t), do: t |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

  test "named io_calls without a hook are removed" do
    assert host(jit(&Defns.steps/1).(Nx.tensor([1.0, 2.0]))) == [4.0, 5.0]
  end

  test "named io_calls with a hook raise" do
    assert_raise ArgumentError, ~r/:step hook is not supported/, fn ->
      jit(&Defns.steps/1, hooks: %{step: &IO.inspect/1}).(Nx.tensor([1.0]))
    end
  end

  test "io_calls with a function raise" do
    assert_raise ArgumentError, ~r/print_value/, fn ->
      jit(&Defns.printed/1).(Nx.tensor([1.0]))
    end
  end

  test "runtime calls raise" do
    fun = fn x -> Nx.runtime_call(x, x, &__MODULE__.double/2) end

    assert_raise ArgumentError, ~r/runtime_call/, fn ->
      jit(fun).(Nx.tensor([1.0]))
    end
  end

  def double(t, _opts), do: Nx.multiply(t, 2)
end
