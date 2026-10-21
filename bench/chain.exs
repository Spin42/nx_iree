# Breaks the 4M-float chain down: transfers, elementwise part, reduction.
Logger.configure(level: :warning)

{:ok, devices} = NxIREE.list_devices("vulkan")
match = System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")
device = Enum.find(devices, &String.contains?(&1.uri, match)) || hd(devices)

flags = [
  "--iree-hal-target-backends=vulkan-spirv",
  "--iree-input-type=stablehlo_xla",
  "--iree-execution-model=async-internal"
] ++ String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)

opts = [compiler: NxIREE.Compiler, iree_compiler_flags: flags, iree_runtime_options: [device: device]]

time = fn f, reps ->
  f.()
  t0 = System.monotonic_time(:microsecond)
  for _ <- 1..reps, do: f.()
  Float.round((System.monotonic_time(:microsecond) - t0) / reps / 1000, 2)
end

{big, _} = Nx.Random.uniform(Nx.Random.key(42), shape: {4_000_000}, type: :f32)
on_dev = Nx.backend_transfer(big, {NxIREE.Backend, device: device})

elementwise = fn t -> t |> Nx.multiply(1.5) |> Nx.add(0.25) |> Nx.exp() |> Nx.multiply(t) end
reduce = fn t -> Nx.sum(t) end
chain = fn t -> t |> elementwise.() |> Nx.sum() end
ident = fn t -> Nx.add(t, 0.0) end

host = fn t -> Nx.backend_transfer(t, Nx.BinaryBackend) end

for {name, f} <- [ident: ident, elementwise: elementwise, reduce: reduce, chain: chain] do
  j = Nx.Defn.jit(f, opts)
  from_host = time.(fn -> j.(big) |> host.() end, 5)
  from_dev = time.(fn -> j.(on_dev) |> host.() end, 5)
  IO.puts("#{name}: host input #{from_host} ms, device input #{from_dev} ms")
end

IO.puts("upload only: #{time.(fn -> Nx.backend_transfer(big, {NxIREE.Backend, device: device}) end, 5)} ms")
IO.puts("readback only: #{time.(fn -> Nx.backend_copy(on_dev, Nx.BinaryBackend) end, 5)} ms")
