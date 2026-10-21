defmodule NxIREE.TuningTest do
  use ExUnit.Case

  @vulkan ["--iree-hal-target-backends=vulkan-spirv"]
  @t Nx.template({128, 128}, :f32)

  defp exported(opts) do
    NxIREE.export(&Nx.dot/2, [@t, @t], [iree_compiler_flags: @vulkan] ++ opts)
    |> :erlang.binary_to_term()
  end

  test "a tuned export records the spec's name, not its file" do
    tuned = exported(iree_tuning: :maxwell_matmul)
    plain = exported([])

    assert "--nx-iree-tuning=maxwell_matmul" in tuned.compilation_flags
    refute Enum.any?(tuned.compilation_flags, &(&1 =~ "transform-dialect-library"))
    refute tuned.bytecode == plain.bytecode
  end

  test "unknown specs and non-Vulkan backends raise" do
    assert_raise ArgumentError, ~r/unknown :iree_tuning :nope/, fn ->
      exported(iree_tuning: :nope)
    end

    assert_raise ArgumentError, ~r/vulkan-spirv backend, got llvm-cpu/, fn ->
      NxIREE.export(&Nx.dot/2, [@t, @t],
        iree_compiler_flags: ["--iree-hal-target-backends=llvm-cpu"],
        iree_tuning: :maxwell_matmul
      )
    end
  end

  test "a tuned matmul matches the CPU on a Vulkan device" do
    case NxIREE.list_devices("vulkan") do
      {:ok, [gpu | _]} ->
        a = Nx.iota({128, 128}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(1000)

        got =
          Nx.Defn.jit(&Nx.dot/2,
            compiler: NxIREE.Compiler,
            iree_runtime_options: [device: gpu],
            iree_tuning: :maxwell_matmul
          ).(a, a)
          |> Nx.backend_transfer(Nx.BinaryBackend)

        want = Nx.dot(a, a)
        assert Nx.subtract(got, want) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() < 1.0e-2

      _ ->
        :ok
    end
  end
end
