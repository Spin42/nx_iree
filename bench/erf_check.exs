# Accuracy of erf-based ops on Vulkan and CPU against EXLA.
Logger.configure(level: :warning)
{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, "GPU-00000000030000"))
cpu = NxIREE.Device.find_default_device("local-sync")
x = Nx.linspace(-4.0, 4.0, n: 2001)

for {name, fun} <- [erf: &Nx.erf/1, gelu: &Axon.Activations.gelu/1, tanh: &Nx.tanh/1, exp: &Nx.exp/1, softmax: &Axon.Activations.softmax/1],
    {tname, device, backend} <- [{"cpu", cpu, "llvm-cpu"}, {"vulkan", gpu, "vulkan-spirv"}] do
  opts = [compiler: NxIREE.Compiler, iree_compiler_flags: ["--iree-hal-target-backends=#{backend}", "--iree-input-type=stablehlo_xla"], iree_runtime_options: [device: device]]
  want = Nx.Defn.jit(fun, compiler: EXLA).(x) |> Nx.backend_transfer(Nx.BinaryBackend)
  got = Nx.Defn.jit(fun, opts).(x) |> Nx.backend_transfer(Nx.BinaryBackend)
  err = Nx.subtract(got, want) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  IO.puts("#{String.pad_trailing(to_string(name), 8)} #{String.pad_trailing(tname, 7)} max abs error #{:erlang.float_to_binary(err, decimals: 9)}")
end
