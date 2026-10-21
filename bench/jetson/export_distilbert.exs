# Exports DistilBERT SST-2 for the Jetson and saves what the board needs to
# run it without Axon or Bumblebee: the exported forward pass per Vulkan
# target, the parameters as a plain map of tensors, tokenized inputs and
# EXLA's reference logits.
#
#   OUT=/tmp/distilbert mix run jetson/export_distilbert.exs
Logger.configure(level: :warning)
out = System.get_env("OUT", "distilbert")
File.mkdir_p!(out)

name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
{:ok, %{model: model, params: params, spec: spec}} = Bumblebee.load_model({:hf, name})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "distilbert/distilbert-base-uncased"})
{_init, predict} = Axon.build(model)

texts = [
  "The food was delicious and the staff were lovely.",
  "Terrible.",
  "I expected more from this film, the plot was thin and the acting wooden.",
  "Not bad at all"
]

tokenizer = Bumblebee.configure(tokenizer, length: 32)
inputs = Bumblebee.apply_tokenizer(tokenizer, texts) |> Map.take(["input_ids", "attention_mask"])

data = params.data
forward = fn data, inputs -> predict.(Axon.ModelState.new(data), inputs).logits end

reference = Nx.Defn.jit(forward, compiler: EXLA).(data, inputs) |> Nx.backend_copy(Nx.BinaryBackend)

nvk_features =
  Enum.join(
    ~w(spirv:v1.3 cap:Shader cap:StorageBuffer8BitAccess cap:Int8 ext:SPV_KHR_8bit_storage
       cap:GroupNonUniform cap:GroupNonUniformArithmetic cap:GroupNonUniformShuffle
       cap:GroupNonUniformBallot cap:Int64 cap:Int16 cap:Float16 cap:StorageBuffer16BitAccess
       ext:SPV_KHR_16bit_storage),
    ","
  )

targets = [
  {"generic", []},
  {"nvk", ["--iree-vulkan-target-features=#{nvk_features}"]}
]

templates = [Nx.to_template(data), Nx.to_template(inputs)]

for {target, extra} <- targets do
  flags = ["--iree-hal-target-backends=vulkan-spirv" | extra]
  bin = NxIREE.export(forward, templates, iree_compiler_flags: flags)
  File.write!(Path.join(out, "distilbert-#{target}.bin"), bin)
  IO.puts("#{target}: #{byte_size(bin)} bytes")
end

host = fn c -> Nx.Defn.Composite.traverse(c, &Nx.backend_copy(&1, Nx.BinaryBackend)) end
File.write!(Path.join(out, "params.nx"), Nx.serialize(host.(data)))
File.write!(Path.join(out, "inputs.nx"), Nx.serialize(host.(inputs)))
File.write!(Path.join(out, "reference.nx"), Nx.serialize(reference))
File.write!(Path.join(out, "labels.etf"), :erlang.term_to_binary(spec.id_to_label))

IO.puts("reference labels: #{inspect(reference |> Nx.argmax(axis: 1) |> Nx.to_flat_list())}")
IO.puts("wrote #{out}")
