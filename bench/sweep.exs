# Reduction time vs size, with host inputs and with a device-resident input
# produced by a previous IREE call.
Logger.configure(level: :warning)

{:ok, devices} = NxIREE.list_devices("vulkan")
match = System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")
device = Enum.find(devices, &String.contains?(&1.uri, match)) || hd(devices)

flags =
  ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla"] ++
    String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)

opts = [compiler: NxIREE.Compiler, iree_compiler_flags: flags, iree_runtime_options: [device: device]]

time = fn f, reps ->
  f.()
  t0 = System.monotonic_time(:microsecond)
  for _ <- 1..reps, do: f.()
  Float.round((System.monotonic_time(:microsecond) - t0) / reps / 1000, 2)
end

for n <- [250_000, 1_000_000, 4_000_000, 16_000_000] do
  {big, _} = Nx.Random.uniform(Nx.Random.key(1), shape: {n}, type: :f32)
  copy = Nx.Defn.jit(fn t -> Nx.add(t, 0.0) end, opts)
  # Output of an IREE call: lives in a device buffer
  resident = copy.(big)
  sum = Nx.Defn.jit(&Nx.sum/1, opts)
  th = time.(fn -> sum.(big) |> Nx.to_number() end, 5)
  td = time.(fn -> sum.(resident) |> Nx.to_number() end, 5)
  IO.puts("n=#{n}: sum host input #{th} ms, device-resident input #{td} ms")
end
