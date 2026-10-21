# Prints BERT-tiny's first-layer attention weights (head 0) on Vulkan and EXLA.
Logger.configure(level: :warning)
{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")))
opts = [compiler: NxIREE.Compiler, iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla"] ++ String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true), iree_runtime_options: [device: gpu]]
repo = {:hf, "hf-internal-testing/tiny-random-BertModel"}
{:ok, %{model: model, params: params, spec: spec}} = Bumblebee.load_model(repo)
IO.puts("hidden #{spec.hidden_size}, heads #{spec.num_attention_heads}, blocks #{spec.num_blocks}")
{_, predict} = Axon.build(model, global_layer_options: [output_attentions: true])
inputs = %{"input_ids" => Nx.tensor([[101, 7592, 1010, 2088, 102, 0, 0, 0]]), "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])}
want = Nx.Defn.jit(predict, compiler: EXLA).(params, inputs).attentions |> elem(0) |> Nx.backend_transfer(Nx.BinaryBackend)
got = Nx.Defn.jit(predict, opts).(params, inputs).attentions |> elem(0) |> Nx.backend_transfer(Nx.BinaryBackend)
IO.puts("shape #{inspect(Nx.shape(got))}")
for row <- 0..2 do
  IO.puts("row #{row} got  #{inspect(got[[0, 0, row]] |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 4)))}")
  IO.puts("row #{row} want #{inspect(want[[0, 0, row]] |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 4)))}")
end
