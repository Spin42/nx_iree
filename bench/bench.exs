# nx_iree on the host GPU through IREE's Vulkan backend: matmul throughput,
# compile time and a correctness check against EXLA (CPU).
Logger.configure(level: :warning)

IO.puts("drivers: #{inspect(NxIREE.list_drivers())}")
{:ok, devices} = NxIREE.list_devices("vulkan")
IO.puts("vulkan devices: #{inspect(devices, limit: :infinity)}")
# The device URI carries the PCI bus; NXIREE_DEVICE_MATCH picks the card (bus 03 is the discrete Radeon)
match = System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")
device = Enum.find(devices, &String.contains?(&1.uri, match)) || hd(devices)
IO.puts("using #{device.uri}")

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
  (System.monotonic_time(:microsecond) - t0) / reps
end

key = Nx.Random.key(42)
dot = fn a, b -> Nx.dot(a, b) end

IO.puts("\n== matmul f32 (GFLOPS)")
IO.puts(String.pad_trailing("n", 6) <> String.pad_leading("nx_iree", 12) <> String.pad_leading("exla_cpu", 12) <> String.pad_leading("compile ms", 12))

for n <- [128, 256, 512, 1024, 2048] do
  {a, key2} = Nx.Random.uniform(key, shape: {n, n}, type: :f32)
  {b, _} = Nx.Random.uniform(key2, shape: {n, n}, type: :f32)

  t0 = System.monotonic_time(:millisecond)
  jitted = Nx.Defn.jit(dot, opts)
  first = jitted.(a, b)
  compile_ms = System.monotonic_time(:millisecond) - t0

  exla = Nx.Defn.jit(dot, compiler: EXLA)
  reps = if n <= 512, do: 20, else: 5
  readback = fn t -> t |> Nx.backend_transfer(Nx.BinaryBackend) end
  # The timed call includes copying the inputs in and the result back
  ti = time.(fn -> jitted.(a, b) |> readback.() end, reps)
  te = time.(fn -> exla.(a, b) |> readback.() end, reps)
  gf = fn us -> 2 * n * n * n / us / 1000 end

  IO.puts(
    String.pad_trailing("#{n}", 6) <>
      String.pad_leading(:erlang.float_to_binary(gf.(ti), decimals: 1), 12) <>
      String.pad_leading(:erlang.float_to_binary(gf.(te), decimals: 1), 12) <>
      String.pad_leading("#{compile_ms}", 12)
  )

  if n == 512 do
    diff = Nx.subtract(readback.(first), readback.(exla.(a, b))) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    IO.puts("   max |iree - exla| at n=512: #{diff}")
  end
end

IO.puts("\n== small fused graph: x * 1.5 + 0.25 |> exp |> * x |> sum on 4M floats (ms)")
{big, _} = Nx.Random.uniform(key, shape: {4_000_000}, type: :f32)
chain = fn t -> t |> Nx.multiply(1.5) |> Nx.add(0.25) |> Nx.exp() |> Nx.multiply(t) |> Nx.sum() end
ji = Nx.Defn.jit(chain, opts)
je = Nx.Defn.jit(chain, compiler: EXLA)
li = time.(fn -> ji.(big) |> Nx.backend_transfer(Nx.BinaryBackend) end, 10)
le = time.(fn -> je.(big) |> Nx.backend_transfer(Nx.BinaryBackend) end, 10)
IO.puts("nx_iree #{Float.round(li / 1000, 2)}  exla_cpu #{Float.round(le / 1000, 2)}")
