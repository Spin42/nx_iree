# Loads AOT modules on NVK running against the nouveau drm-shim, so NAK
# compiles their shaders for the emulated chip (NOUVEAU_CHIPSET) without the
# hardware. With NAK_DEBUG=print the shaders are dumped to stderr.
#
#   AOT_MODULES=jetson-modules-rdna3.bin AOT_KEYS="matmul 1024" mix run shim_load.exs
Logger.configure(level: :warning)

%{modules: modules} = System.fetch_env!("AOT_MODULES") |> File.read!() |> :erlang.binary_to_term()

keys =
  case System.get_env("AOT_KEYS") do
    nil ->
      Map.keys(modules)

    spec ->
      [op, n] = String.split(spec)
      [{String.to_atom(op), String.to_integer(n)}]
  end

{:ok, devices} = NxIREE.list_devices("vulkan")
IO.puts(:stderr, "devices: #{Enum.map_join(devices, ", ", & &1.uri)}")
device = hd(devices)

for key <- keys do
  IO.puts(:stderr, "=== loading #{inspect(key)}")
  {:ok, _} = NxIREE.load(modules[key], device: device)
end
