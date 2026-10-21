# Softmax over masked scores (as attention masks produce) on Vulkan vs EXLA.
Logger.configure(level: :warning)
{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, "GPU-00000000030000"))
opts = [compiler: NxIREE.Compiler, iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla"], iree_runtime_options: [device: gpu]]
scores = Nx.tensor([[0.3, -1.2, 2.0, 0.5, 0.1, 0.9, -0.4, 1.1]])
mask = Nx.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])

for {name, fill} <- [neg_inf: Nx.Constants.neg_infinity(:f32), min_finite: Nx.Constants.min_finite(:f32), minus_1e9: Nx.tensor(-1.0e9)] do
  fun = fn s, m -> Axon.Activations.softmax(Nx.select(m, s, fill)) end
  want = Nx.Defn.jit(fun, compiler: EXLA).(scores, mask) |> Nx.backend_transfer(Nx.BinaryBackend)
  got = Nx.Defn.jit(fun, opts).(scores, mask) |> Nx.backend_transfer(Nx.BinaryBackend)
  IO.puts("#{name}:\n  got  #{inspect(Nx.to_flat_list(got))}\n  want #{inspect(Nx.to_flat_list(want))}")
end
