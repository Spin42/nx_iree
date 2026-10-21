# Runs pretrained HuggingFace models through Bumblebee servings on EXLA (the
# reference) and on nx_iree, and compares what applications see: labels,
# scores, predicted tokens and generated text.
#
#   IMAGE=cats.ppm mix run real_models.exs
#   CHECKS="sentiment gpt2" TARGETS=vulkan mix run real_models.exs
#
# IMAGE is a binary PPM (P6), e.g. converted from COCO val2017 000000039769.
# Checkpoints are downloaded to Bumblebee's cache on first use (about 1.3 GB).
Logger.configure(level: :warning)

only_checks = System.get_env("CHECKS") |> then(&(&1 && String.split(&1)))
only_targets = System.get_env("TARGETS") |> then(&(&1 && String.split(&1)))

{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")))
cpu = NxIREE.Device.find_default_device("local-task")
extra = String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)

targets = [
  {"cpu", [compiler: NxIREE.Compiler, iree_compiler_flags: extra, iree_runtime_options: [device: cpu]]},
  {"vulkan", [compiler: NxIREE.Compiler, iree_compiler_flags: extra, iree_runtime_options: [device: gpu]]}
]

defmodule Check do
  def ppm!(path) do
    data = File.read!(path)
    [header, w, h] = Regex.run(~r/\AP6\s+(\d+)\s+(\d+)\s+255\s/, data)
    {w, h} = {String.to_integer(w), String.to_integer(h)}

    data
    |> binary_part(byte_size(header), w * h * 3)
    |> Nx.from_binary(:u8)
    |> Nx.reshape({h, w, 3})
  end

  # Starts the serving as a process, as applications do, and runs it twice:
  # the first call follows compilation, the second is timed
  def run(serving, input) do
    name = :"serving_#{System.unique_integer([:positive])}"
    {:ok, pid} = Nx.Serving.start_link(serving: serving, name: name, batch_timeout: 10)
    first = Nx.Serving.batched_run(name, input)
    {us, second} = :timer.tc(fn -> Nx.Serving.batched_run(name, input) end)
    Process.unlink(pid)
    Process.exit(pid, :shutdown)
    {first, second, us}
  end

  def predictions(%{predictions: preds}), do: preds
  def predictions(list) when is_list(list), do: Enum.flat_map(list, &predictions/1)
  def predictions(_other), do: []

  # Labels must match in order; scores within tolerance
  def compare_predictions(got, want, key) do
    got = predictions(got)
    want = predictions(want)
    same_labels? = Enum.map(got, & &1[key]) == Enum.map(want, & &1[key])

    worst =
      Enum.zip_with(got, want, fn g, w -> abs(g.score - w.score) end)
      |> Enum.max(fn -> 0.0 end)

    cond do
      not same_labels? ->
        {:mismatch, "labels differ: got #{inspect(Enum.map(got, & &1[key]))}, want #{inspect(Enum.map(want, & &1[key]))}"}

      worst > 1.0e-3 ->
        {:mismatch, "max score difference #{Float.round(worst, 6)}"}

      true ->
        {:ok, "max score difference #{:erlang.float_to_binary(worst, decimals: 6)}"}
    end
  end

  def compare_text(%{results: [%{text: got} | _]}, %{results: [%{text: want} | _]}) do
    if got == want,
      do: {:ok, "identical text (#{String.length(want)} chars)"},
      else: {:mismatch, "got #{inspect(got)}, want #{inspect(want)}"}
  end
end

repo = fn name -> {:hf, name} end

checks = [
  {"sentiment",
   fn defn_options ->
     name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
     {:ok, model} = Bumblebee.load_model(repo.(name))
     # the fine-tuned repo has no tokenizer.json; the base model's is the same
     {:ok, tokenizer} = Bumblebee.load_tokenizer(repo.("distilbert/distilbert-base-uncased"))

     serving =
       Bumblebee.Text.text_classification(model, tokenizer,
         top_k: 2,
         compile: [batch_size: 4, sequence_length: 32],
         defn_options: defn_options,
         preallocate_params: true
       )

     # different lengths, so the batch is padded
     input = [
       "The food was delicious and the staff were lovely.",
       "Terrible.",
       "I expected more from this film, the plot was thin and the acting wooden, although the music was nice.",
       "Not bad at all"
     ]

     {serving, input, &Check.compare_predictions(&1, &2, :label)}
   end},
  {"fill_mask",
   fn defn_options ->
     name = "google-bert/bert-base-uncased"
     {:ok, model} = Bumblebee.load_model(repo.(name))
     {:ok, tokenizer} = Bumblebee.load_tokenizer(repo.(name))

     serving =
       Bumblebee.Text.fill_mask(model, tokenizer,
         top_k: 5,
         compile: [batch_size: 1, sequence_length: 32],
         defn_options: defn_options,
         preallocate_params: true
       )

     {serving, "The capital of France is [MASK].", &Check.compare_predictions(&1, &2, :token)}
   end},
  {"gpt2",
   fn defn_options ->
     name = "openai-community/gpt2"
     {:ok, model} = Bumblebee.load_model(repo.(name))
     {:ok, tokenizer} = Bumblebee.load_tokenizer(repo.(name))
     {:ok, config} = Bumblebee.load_generation_config(repo.(name))
     config = Bumblebee.configure(config, max_new_tokens: 24, strategy: %{type: :greedy_search})

     serving =
       Bumblebee.Text.generation(model, tokenizer, config,
         compile: [batch_size: 1, sequence_length: 32],
         defn_options: defn_options,
         preallocate_params: true
       )

     {serving, "The Jetson Nano is a small computer that", &Check.compare_text/2}
   end},
  {"resnet50",
   fn defn_options ->
     name = "microsoft/resnet-50"
     {:ok, model} = Bumblebee.load_model(repo.(name))
     {:ok, featurizer} = Bumblebee.load_featurizer(repo.(name))

     serving =
       Bumblebee.Vision.image_classification(model, featurizer,
         top_k: 5,
         compile: [batch_size: 1],
         defn_options: defn_options,
         preallocate_params: true
       )

     {serving, Check.ppm!(System.fetch_env!("IMAGE")), &Check.compare_predictions(&1, &2, :label)}
   end},
  {"vit",
   fn defn_options ->
     name = "google/vit-base-patch16-224"
     {:ok, model} = Bumblebee.load_model(repo.(name))
     {:ok, featurizer} = Bumblebee.load_featurizer(repo.(name))

     serving =
       Bumblebee.Vision.image_classification(model, featurizer,
         top_k: 5,
         compile: [batch_size: 1],
         defn_options: defn_options,
         preallocate_params: true
       )

     {serving, Check.ppm!(System.fetch_env!("IMAGE")), &Check.compare_predictions(&1, &2, :label)}
   end}
]

for {name, build} <- checks, is_nil(only_checks) or name in only_checks do
  # Preprocessing such as image resizing runs defn eagerly, through the global
  # options, so it is compiled for the same target as the model
  Nx.Defn.global_default_options(compiler: EXLA)
  {serving, input, compare} = build.(compiler: EXLA)
  {_, want, exla_us} = Check.run(serving, input)
  IO.puts("\n#{name}: exla #{div(exla_us, 1000)} ms per call; reference #{inspect(Check.predictions(want) |> Enum.take(2), limit: 4) |> String.slice(0, 160)}")
  if match?(%{results: _}, want), do: IO.puts("  reference text: #{inspect(hd(want.results).text)}")

  for {tname, opts} <- targets, is_nil(only_targets) or tname in only_targets do
    result =
      try do
        Nx.Defn.global_default_options(opts)
        {serving, input, _} = build.(opts)
        {first, second, us} = Check.run(serving, input)

        with {:ok, msg} <- compare.(first, want),
             {:ok, _} <- compare.(second, want) do
          "ok   #{msg}, #{div(us, 1000)} ms per call"
        else
          {:mismatch, msg} -> "MISMATCH #{msg}"
        end
      rescue
        e -> "ERROR #{Exception.message(e) |> String.replace("\n", " ") |> String.slice(0, 300)}"
      end

    IO.puts("  #{String.pad_trailing(tname, 7)} #{result}")
  end
end
