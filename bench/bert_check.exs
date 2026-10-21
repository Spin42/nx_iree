# Per-layer Vulkan/EXLA difference in BERT-tiny's hidden states, with and
# without padding in the attention mask.
Logger.configure(level: :warning)
{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, "GPU-00000000030000"))
opts = [compiler: NxIREE.Compiler, iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla"], iree_runtime_options: [device: gpu]]
repo = {:hf, "hf-internal-testing/tiny-random-BertModel"}
{:ok, spec} = Bumblebee.load_spec(repo)

{:ok, %{model: model, params: params}} = Bumblebee.load_model(repo, spec: spec)
{_, predict} = Axon.build(model, global_layer_options: [output_hidden_states: true, output_attentions: true])
ids = Nx.tensor([[101, 7592, 1010, 2088, 102, 0, 0, 0]])

for {label, mask} <- [padded: Nx.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]), full: Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])] do
  inputs = %{"input_ids" => ids, "attention_mask" => mask}
  want = Nx.Defn.jit(predict, compiler: EXLA).(params, inputs)|> then(&Tuple.to_list(&1[String.to_atom(System.get_env("FIELD", "hidden_states"))]))
  got = Nx.Defn.jit(predict, opts).(params, inputs)|> then(&Tuple.to_list(&1[String.to_atom(System.get_env("FIELD", "hidden_states"))]))

  errs =
    Enum.zip_with(got, want, fn g, w ->
      g = Nx.backend_transfer(g, Nx.BinaryBackend)
      w = Nx.backend_transfer(w, Nx.BinaryBackend)
      (Nx.subtract(g, w) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()) /
        (w |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number())
    end)

  IO.puts("#{label}: per layer (embeddings first) #{inspect(Enum.map(errs, &Float.round(&1, 6)))}")
end
