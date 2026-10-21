# Writes the StableHLO module EXLA produces for BERT-tiny with attentions
# enabled (and the Vulkan bytecode) to the directory in OUT.
Logger.configure(level: :warning)
out = System.get_env("OUT", ".")
repo = {:hf, "hf-internal-testing/tiny-random-BertModel"}
{:ok, %{model: model, params: params}} = Bumblebee.load_model(repo)
{_, predict} = Axon.build(model, global_layer_options: [output_attentions: true])
inputs = %{"input_ids" => Nx.tensor([[101, 7592, 1010, 2088, 102, 0, 0, 0]]), "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])}
flags = ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla"]
{:ok, mod} = NxIREE.Compiler.to_bytecode(predict, [params, inputs], iree_compiler_flags: flags)
File.write!(Path.join(out, "bert.mlir"), mod.mlir_module)
File.write!(Path.join(out, "bert.vmfb"), mod.bytecode)
IO.puts("wrote #{byte_size(mod.mlir_module)} bytes of MLIR")
