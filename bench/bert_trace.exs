# Runs BERT-tiny once on one IREE device with per-dispatch tensor tracing so
# the stderr log can be diffed between devices.
#   NXIREE_BACKEND=cpu            -> llvm-cpu on local-sync
#   NXIREE_BACKEND=vulkan         -> vulkan-spirv on NXIREE_DEVICE_MATCH
Logger.configure(level: :warning)
extra = String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)

{flags, device} =
  case System.get_env("NXIREE_BACKEND", "vulkan") do
    "cpu" ->
      {["--iree-hal-target-backends=llvm-cpu"], NxIREE.Device.find_default_device("local-sync")}

    "vulkan" ->
      {:ok, vk} = NxIREE.list_devices("vulkan")
      match = System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")
      {["--iree-hal-target-backends=vulkan-spirv"], Enum.find(vk, &String.contains?(&1.uri, match))}
  end

flags = flags ++ ["--iree-input-type=stablehlo_xla", "--iree-flow-trace-dispatch-tensors"] ++ extra
opts = [compiler: NxIREE.Compiler, iree_compiler_flags: flags, iree_runtime_options: [device: device]]
repo = {:hf, "hf-internal-testing/tiny-random-BertModel"}
{:ok, %{model: model, params: params}} = Bumblebee.load_model(repo)
{_, predict} = Axon.build(model, global_layer_options: [output_attentions: true])
inputs = %{"input_ids" => Nx.tensor([[101, 7592, 1010, 2088, 102, 0, 0, 0]]), "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])}
got = Nx.Defn.jit(predict, opts).(params, inputs).attentions |> elem(0) |> Nx.backend_transfer(Nx.BinaryBackend)
IO.puts("row 0 #{inspect(got[[0, 0, 0]] |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 4)))}")
