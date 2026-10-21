# Runs Axon and Bumblebee models through nx_iree and compares with EXLA.
#
#   mix run models.exs                      # all models, CPU and Vulkan
#   MODELS="mlp bert" TARGETS=cpu mix run models.exs
#
# Bumblebee models are HuggingFace's tiny random test checkpoints, so this
# checks operator coverage and numerics, not accuracy.
Logger.configure(level: :warning)

only_models = System.get_env("MODELS") |> then(&(&1 && String.split(&1)))
only_targets = System.get_env("TARGETS") |> then(&(&1 && String.split(&1)))
{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")))
cpu = NxIREE.Device.find_default_device("local-sync")
base = ["--iree-input-type=stablehlo_xla"]

targets = [
  {"cpu", [compiler: NxIREE.Compiler, iree_compiler_flags: ["--iree-hal-target-backends=llvm-cpu" | base], iree_runtime_options: [device: cpu]]},
  {"vulkan", [compiler: NxIREE.Compiler, iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv" | base], iree_runtime_options: [device: gpu]]}
]

key = Nx.Random.key(11)
uniform = fn shape -> shape |> then(&Nx.Random.uniform(key, shape: &1)) |> elem(0) end

axon = fn model, input ->
  {init_fn, predict_fn} = Axon.build(model)
  params = init_fn.(Nx.to_template(input), Axon.ModelState.empty())
  {predict_fn, params, input}
end

bumblebee = fn repo, inputs ->
  {:ok, %{model: model, params: params}} = Bumblebee.load_model({:hf, repo})
  {_init, predict_fn} = Axon.build(model)
  {predict_fn, params, inputs}
end

models = [
  {"mlp", fn ->
     Axon.input("x", shape: {nil, 784})
     |> Axon.dense(128, activation: :relu)
     |> Axon.dense(10, activation: :softmax)
     |> axon.(uniform.({32, 784}))
   end},
  {"cnn", fn ->
     Axon.input("x", shape: {nil, 28, 28, 1})
     |> Axon.conv(16, kernel_size: 3, activation: :relu)
     |> Axon.max_pool(kernel_size: 2)
     |> Axon.conv(32, kernel_size: 3, activation: :relu)
     |> Axon.global_avg_pool()
     |> Axon.dense(10)
     |> axon.(uniform.({8, 28, 28, 1}))
   end},
  {"norms", fn ->
     Axon.input("x", shape: {nil, 16, 16, 3})
     |> Axon.conv(8, kernel_size: 3, padding: :same)
     |> Axon.batch_norm()
     |> Axon.relu()
     |> Axon.flatten()
     |> Axon.dense(32)
     |> Axon.layer_norm()
     |> Axon.dropout(rate: 0.5)
     |> Axon.dense(4)
     |> axon.(uniform.({4, 16, 16, 3}))
   end},
  {"bert", fn ->
     bumblebee.("hf-internal-testing/tiny-random-BertModel", %{
       "input_ids" => Nx.tensor([[101, 7592, 1010, 2088, 102, 0, 0, 0]]),
       "attention_mask" => if(System.get_env("BERT_NO_PAD"), do: Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]), else: Nx.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]))
     })
   end},
  {"gpt2", fn ->
     bumblebee.("hf-internal-testing/tiny-random-GPT2Model", %{
       "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60]]),
       "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1]])
     })
   end},
  {"resnet", fn ->
     bumblebee.("hf-internal-testing/tiny-random-ResNetModel", %{"pixel_values" => uniform.({1, 64, 64, 3})})
   end},
  {"vit", fn ->
     bumblebee.("hf-internal-testing/tiny-random-ViTModel", %{"pixel_values" => uniform.({1, 30, 30, 3})})
   end}
]

leaves = fn out -> Nx.Defn.Composite.flatten_list([out]) end

for {name, build} <- models, is_nil(only_models) or name in only_models do
  case (try do {:ok, build.()} rescue e -> {:error, e} end) do
    {:error, e} ->
      IO.puts("#{name}: could not build: #{Exception.message(e) |> String.slice(0, 160)}")

    {:ok, {predict_fn, params, input}} ->
      want =
        Nx.Defn.jit(predict_fn, compiler: EXLA).(params, input)
        |> leaves.()
        |> Enum.map(&Nx.backend_transfer(&1, Nx.BinaryBackend))

      for {tname, opts} <- targets, is_nil(only_targets) or tname in only_targets do
        result =
          try do
            {us, got} = :timer.tc(fn -> Nx.Defn.jit(predict_fn, opts).(params, input) |> leaves.() end)

            errs =
              Enum.zip_with(got, want, fn g, w ->
                g = Nx.backend_transfer(g, Nx.BinaryBackend)
                w = Nx.backend_transfer(w, Nx.BinaryBackend)
                scale = max(w |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number(), 1.0e-6)
                (Nx.subtract(g, w) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()) / scale
              end)

            worst = Enum.max(errs)
            if System.get_env("PER_OUTPUT"), do: IO.puts("  per output: #{inspect(errs)}")
            verdict = if worst < 1.0e-3, do: "ok", else: "MISMATCH"
            "#{verdict} (#{length(errs)} outputs, worst relative error #{:erlang.float_to_binary(worst, decimals: 6)}, first call #{div(us, 1000)} ms)"
          rescue
            e -> "ERROR #{Exception.message(e) |> String.replace("\n", " ") |> String.slice(0, 220)}"
          end

        IO.puts("#{String.pad_trailing(name, 7)} #{String.pad_trailing(tname, 7)} #{result}")
      end
  end
end
