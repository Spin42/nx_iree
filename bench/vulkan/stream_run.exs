if root = System.get_env("STEREO_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx nx_iree),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:nx)

{:ok, devices} = NxIREE.list_devices("vulkan")

device =
  case {System.get_env("NXIREE_DEVICE_URI"), devices} do
    {nil, [device]} ->
      device

    {uri, _} when is_binary(uri) ->
      Enum.find(devices, &(&1.uri == uri)) || raise("device not found")

    _ ->
      raise("set NXIREE_DEVICE_URI")
  end

IO.puts("STREAM_DEVICE=#{device.uri}")

models = System.fetch_env!("VULKAN_BUNDLE") |> File.read!() |> :erlang.binary_to_term()

for {words, binary} <- Enum.sort(models) do
  model = NxIREE.load_exported(binary, device: device)
  pattern = for value <- [0, 1, 0xFFFFFFFF, 0x12345678], into: <<>>, do: <<value::little-32>>
  input_binary = :binary.copy(pattern, div(words, 4))

  expected_pattern =
    for <<value::little-32 <- pattern>>,
      into: <<>>,
      do: <<Bitwise.bxor(value, 0x5A5A5A5A)::little-32>>

  expected = :binary.copy(expected_pattern, div(words, 4))

  input =
    input_binary |> Nx.from_binary(:u32) |> Nx.backend_copy({NxIREE.Backend, device: device})

  try do
    first = NxIREE.run(model, [input]) |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_binary()
    if first != expected, do: raise("stream output mismatch")

    times =
      for index <- 1..23 do
        for path <- Path.wildcard("/sys/class/thermal/thermal_zone*/temp") do
          if File.read!(path) |> String.trim() |> String.to_integer() >= 70_000,
            do: raise("thermal limit")
        end

        {us, output} = :timer.tc(fn -> NxIREE.run(model, [input]) end)

        try do
          if index == 23 and Nx.to_binary(output) != expected,
            do: raise("repeated stream output mismatch")
        after
          Nx.backend_deallocate(output)
        end

        us / 1000
      end
      |> Enum.drop(3)

    if Nx.to_binary(input) != input_binary, do: raise("caller input changed")
    median = times |> Enum.sort() |> Enum.at(9)
    gbps = 2 * words * 4 / (median * 1_000_000)

    IO.puts(
      "STREAM_RESULT words=#{words} bytes_read_written=#{2 * words * 4} median_ms=#{median} GBps=#{gbps} ms=#{inspect(times)} includes=runtime_allocation_completed_dispatch excludes=upload_readback_release"
    )
  after
    Nx.backend_deallocate(input)
  end
end

IO.puts("VULKAN_STREAM_PASS")
