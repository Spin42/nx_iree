# EXLA 1.0 through IREE's Vulkan PJRT plugin: plain `compiler: EXLA` code
# running on a Vulkan GPU.
Logger.configure(level: :warning)

plugin = System.fetch_env!("PJRT_PLUGIN")
device_id = String.to_integer(System.get_env("PJRT_DEVICE_ID", "0"))

Application.put_env(:exla, :clients,
  Application.fetch_env!(:exla, :clients) ++
    [iree_vulkan: [platform: :pjrt_plugin, device_type: "iree_vulkan", plugin_path: plugin, default_device_id: device_id]]
)

client = EXLA.Client.fetch!(:iree_vulkan)
IO.puts("PJRT client: #{client.device_count} devices, using #{client.default_device_id}")

time = fn f, reps ->
  f.()
  t0 = System.monotonic_time(:microsecond)
  for _ <- 1..reps, do: f.()
  (System.monotonic_time(:microsecond) - t0) / reps
end

key = Nx.Random.key(42)
host = fn t -> Nx.backend_transfer(t, Nx.BinaryBackend) end

for n <- [256, 512, 1024, 2048] do
  {a, k2} = Nx.Random.uniform(key, shape: {n, n}, type: :f32)
  {b, _} = Nx.Random.uniform(k2, shape: {n, n}, type: :f32)
  gpu = EXLA.jit(&Nx.dot/2, client: :iree_vulkan)
  cpu = EXLA.jit(&Nx.dot/2, client: :host)

  if n == 512 do
    diff = Nx.subtract(host.(gpu.(a, b)), host.(cpu.(a, b))) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    IO.puts("max |pjrt - host| at 512: #{diff}")
  end

  reps = if n >= 1024, do: 5, else: 20
  tg = time.(fn -> gpu.(a, b) |> host.() end, reps)
  tc = time.(fn -> cpu.(a, b) |> host.() end, reps)
  gf = fn us -> Float.round(2 * n * n * n / us / 1000, 1) end
  IO.puts("matmul #{n}: pjrt vulkan #{gf.(tg)} GFLOPS, exla host #{gf.(tc)} GFLOPS")
end
