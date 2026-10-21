# Per-call latency of a tiny jitted add, input already on the device.
Logger.configure(level: :warning)

{:ok, devices} = NxIREE.list_devices(System.get_env("NXIREE_DRIVER", "vulkan"))
match = System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")
device = Enum.find(devices, &String.contains?(&1.uri, match)) || hd(devices)
IO.puts("using #{device.uri}")

flags =
  ["--iree-hal-target-backends=#{device.compiler_target_backend}", "--iree-input-type=stablehlo_xla"] ++
    String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)

opts = [compiler: NxIREE.Compiler, iree_compiler_flags: flags, iree_runtime_options: [device: device]]

x = Nx.iota({256}, type: :f32) |> Nx.backend_transfer({NxIREE.Backend, device: device})
j = Nx.Defn.jit(if(System.get_env("OP") == "sum", do: &Nx.sum/1, else: fn t -> Nx.add(t, t) end), opts)
j.(x)

reps = 50
t0 = System.monotonic_time(:microsecond)
for _ <- 1..reps, do: j.(x)
IO.puts("call only: #{Float.round((System.monotonic_time(:microsecond) - t0) / reps, 1)} us")

t0 = System.monotonic_time(:microsecond)
for _ <- 1..reps, do: j.(x) |> Nx.to_binary()
IO.puts("call + readback: #{Float.round((System.monotonic_time(:microsecond) - t0) / reps, 1)} us")
