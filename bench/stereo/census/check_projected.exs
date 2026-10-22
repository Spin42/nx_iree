if root = System.get_env("STEREO_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx nx_iree),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:nx)
bundle = System.fetch_env!("STEREO_BUNDLE") |> File.read!() |> :erlang.binary_to_term()
{:ok, devices} = NxIREE.list_devices("vulkan")
uri = System.get_env("NXIREE_DEVICE_URI")
device = Enum.find(devices, fn device -> is_nil(uri) or device.uri == uri end)
unless device, do: raise("no matching Vulkan device")
model = NxIREE.load_exported(bundle.models["consistency_projected"], device: device)
directory = System.fetch_env!("VIDEO_DATA")
bytes = Enum.product(Tuple.to_list(bundle.shape)) * 4

read = fn name ->
  data = File.read!(Path.join(directory, "consistency_edge_#{name}.npy"))
  <<0x93, "NUMPY", _::binary>> = data
  binary_part(data, byte_size(data) - bytes, bytes)
end

inputs =
  for {name, type} <- [{"left", :f32}, {"winners", :u32}] do
    read.(name)
    |> Nx.from_binary(type)
    |> Nx.reshape(bundle.shape)
    |> Nx.backend_copy({NxIREE.Backend, device: device})
  end

try do
  result = NxIREE.run(model, inputs)

  try do
    unless Nx.to_binary(result) == read.("expected"), do: raise("projection edge mismatch")
  after
    Nx.backend_deallocate(result)
  end
after
  Enum.each(inputs, &Nx.backend_deallocate/1)
end

IO.puts("PROJECTED_EDGES_PASS")
