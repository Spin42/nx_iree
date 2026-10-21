# 1xK . KxN on each device and IREE Vulkan target, against EXLA.
#   CASE="vulkan pascal" mix run tuning/matvec_check.exs   (one case)
Logger.configure(level: :warning)
{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")))
cpu = NxIREE.Device.find_default_device("local-sync")
{a, k} = Nx.Random.uniform(Nx.Random.key(3), shape: {1, 256})
{b, _} = Nx.Random.uniform(k, shape: {256, 256})
want = Nx.Defn.jit(&Nx.dot/2, compiler: EXLA).(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
only = System.get_env("CASE")

for {label, device, extra} <- [
      {"cpu", cpu, []},
      {"vulkan default target", gpu, []},
      {"vulkan pascal", gpu, ["--iree-vulkan-target=pascal"]},
      {"vulkan rdna3", gpu, ["--iree-vulkan-target=rdna3"]}
    ],
    is_nil(only) or label == only do
  flags =
    ["--iree-hal-target-backends=#{device.compiler_target_backend}", "--iree-input-type=stablehlo_xla"] ++ extra

  opts = [compiler: NxIREE.Compiler, iree_compiler_flags: flags, iree_runtime_options: [device: device]]
  got = Nx.Defn.jit(&Nx.dot/2, opts).(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
  err = Nx.subtract(got, want) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  IO.puts("#{label}: max err #{err}  first #{inspect(Enum.take(Nx.to_flat_list(got), 3))}  want #{inspect(Enum.take(Nx.to_flat_list(want), 3))}")
end
