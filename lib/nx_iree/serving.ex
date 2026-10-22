defmodule NxIREE.Serving do
  @moduledoc """
  Runs an exported IREE function through `Nx.Serving` without compilation.

  Each `Nx.Batch.stack/1` entry represents one complete invocation of the
  exported function. Use a tensor for one argument or a tuple of tensors
  for multiple arguments. Input shapes exclude the added serving batch axis.
  Results remain on the IREE device with that axis added back.

      serving = NxIREE.Serving.new(model) |> Nx.Serving.streaming()
      {:ok, _pid} = Nx.Serving.start_link(name: MyServing, serving: serving)
      batches = Stream.map(inputs, &Nx.Batch.stack([&1]))
      results = Nx.Serving.batched_run(MyServing, batches)

  The maximum batch size is one invocation. Nx.Serving provides scheduling
  and backpressure for each input stream. A stream of results avoids
  concatenating multiple invocations into one tensor. The loaded model uses
  one IREE context; this adapter does not create additional GPU contexts.
  Caller-owned input tensors remain valid after serving.
  """

  @spec new(NxIREE.Exported.t()) :: Nx.Serving.t()
  def new(%NxIREE.Exported{} = model) do
    Nx.Serving.new(fn _options -> fn batch -> execute(model, batch) end end)
    |> Nx.Serving.batch_size(1)
  end

  defp execute(model, batch) do
    {inputs, nil} =
      Nx.LazyContainer.traverse(batch, nil, fn _template, materialize, acc ->
        {Nx.squeeze(materialize.(), axes: [0]), acc}
      end)

    device_ref = model.device.ref

    {args, owned} =
      inputs
      |> then(&Nx.Defn.Composite.flatten_list([&1]))
      |> Enum.map_reduce([], fn
        %Nx.Tensor{data: %NxIREE.Backend{device: ^device_ref}} = tensor, owned ->
          {tensor, owned}

        tensor, owned ->
          copy = Nx.backend_copy(tensor, {NxIREE.Backend, device: model.device})
          {copy, [copy | owned]}
      end)

    try do
      model
      |> NxIREE.run(args)
      |> Nx.Defn.Composite.traverse(fn tensor ->
        output = Nx.new_axis(tensor, 0)
        Nx.backend_deallocate(tensor)
        output
      end)
    after
      Enum.each(owned, &Nx.backend_deallocate/1)
    end
  end
end
