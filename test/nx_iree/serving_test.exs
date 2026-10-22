defmodule NxIREE.ServingTest do
  use ExUnit.Case

  setup_all do
    binary =
      NxIREE.export(
        fn a, b -> %{sum: Nx.add(a, b), original: a} end,
        [Nx.template({2}, :f32), Nx.template({2}, :f32)],
        iree_compiler_flags: [
          "--iree-hal-target-backends=llvm-cpu",
          "--iree-input-type=stablehlo_xla"
        ]
      )

    %{model: NxIREE.load_exported(binary, device: "local-sync://")}
  end

  test "serves exported functions without a compiler and preserves borrowed inputs", %{
    model: model
  } do
    a = Nx.tensor([1.0, 2.0], backend: {NxIREE.Backend, device: model.device})
    b = Nx.tensor([10.0, 20.0], backend: Nx.BinaryBackend)
    previous = Nx.Defn.default_options()
    Nx.Defn.default_options(iree_compiler_flags: ["--invalid-serving-compile"])

    try do
      output = Nx.Serving.run(NxIREE.Serving.new(model), Nx.Batch.stack([{a, b}]))
      assert Nx.shape(output.sum) == {1, 2}
      assert output.sum.data.device == model.device.ref
      assert Nx.to_flat_list(output.sum) == [11.0, 22.0]
      Nx.backend_deallocate(output.original)
      assert Nx.to_flat_list(a) == [1.0, 2.0]
      assert Nx.to_flat_list(b) == [10.0, 20.0]
      Nx.backend_deallocate(output.sum)
    after
      Nx.Defn.default_options(previous)
      Nx.backend_deallocate(a)
    end
  end

  test "existing Nx.Serving streams schedule changing inputs in order", %{model: model} do
    serving = NxIREE.Serving.new(model) |> Nx.Serving.streaming()
    start_supervised!({Nx.Serving, name: __MODULE__, serving: serving, batch_timeout: 0})

    batches =
      Stream.map(1..8, fn n ->
        Nx.Batch.stack([
          {Nx.tensor([n * 1.0, n * 2.0], backend: Nx.BinaryBackend),
           Nx.tensor([3.0, 4.0], backend: Nx.BinaryBackend)}
        ])
      end)

    results =
      Nx.Serving.batched_run(__MODULE__, batches)
      |> Enum.map(fn {:batch, output, _metadata} ->
        values = Nx.to_flat_list(output.sum)
        Nx.backend_deallocate(output.sum)
        Nx.backend_deallocate(output.original)
        values
      end)

    assert results == for(n <- 1..8, do: [n * 1.0 + 3.0, n * 2.0 + 4.0])
  end
end
