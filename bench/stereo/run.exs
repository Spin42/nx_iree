root = System.get_env("STEREO_RUNTIME_ROOT")

if root do
  for app <- ~w(complex telemetry nx nx_iree),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

bundle = System.get_env("STEREO_BUNDLE", Path.join(root || __DIR__, "stereo.bin"))

{:ok, _} = Application.ensure_all_started(:nx)

defmodule StereoPathBench do
  def run(bundle) do
    %{models: exports, fixtures: fixtures, benchmarks: benchmarks} =
      bundle |> File.read!() |> :erlang.binary_to_term()

    {:ok, devices} = NxIREE.list_devices("vulkan")

    device =
      case {System.get_env("NXIREE_DEVICE_URI"), devices} do
        {nil, [device]} ->
          device

        {uri, _} when is_binary(uri) ->
          Enum.find(devices, &(&1.uri == uri)) || raise("Vulkan device #{uri} not found")

        _ ->
          raise("set NXIREE_DEVICE_URI to one of #{inspect(Enum.map(devices, & &1.uri))}")
      end

    IO.puts("DEVICE=#{device.uri}")

    models =
      Map.new(exports, fn {shape, binary} ->
        model = NxIREE.load_exported(binary, device: device)

        IO.puts(
          "MODEL shape=#{inspect(shape)} written_inputs=#{inspect(model.module.written_inputs)}"
        )

        {shape, model}
      end)

    for fixture <- fixtures, do: check(fixture, models, device, false)
    for fixture <- benchmarks, do: check(fixture, models, device, true)
    IO.puts("STEREO_PATH_PASS")
  end

  defp check({name, shape, input, expected, p1, p2}, models, device, benchmark?) do
    guard_resources!()

    args = [
      Nx.from_binary(input, :s32) |> Nx.reshape(shape),
      Nx.tensor(p1, type: :s32),
      Nx.tensor(p2, type: :s32)
    ]

    resident = Enum.map(args, &Nx.backend_copy(&1, {NxIREE.Backend, device: device}))

    run = fn ->
      {invoke_us, output} = :timer.tc(fn -> NxIREE.run(Map.fetch!(models, shape), resident) end)

      {readback_us, binary} =
        :timer.tc(fn -> output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_binary() end)

      {binary, invoke_us / 1000, readback_us / 1000}
    end

    try do
      {result, _, _} = run.()
      if result != expected, do: raise("#{name}: GPU path costs differ from scalar reference")
      IO.puts("PASS #{name}: #{div(byte_size(expected), 4)} exact values")

      if benchmark? do
        repeats = System.get_env("STEREO_REPEATS", "10") |> String.to_integer()
        if repeats < 1, do: raise("STEREO_REPEATS must be positive")
        pause_ms = System.get_env("STEREO_PAUSE_MS", "0") |> String.to_integer()
        if pause_ms < 0, do: raise("STEREO_PAUSE_MS must be nonnegative")

        times =
          for _ <- 1..repeats do
            guard_resources!()
            {us, {got, invoke_ms, readback_ms}} = :timer.tc(run)
            if got != expected, do: raise("#{name}: repeated invocation changed results")
            if pause_ms > 0, do: Process.sleep(pause_ms)
            {us / 1000, invoke_ms, readback_ms}
          end

        IO.puts(
          "PATH_BENCH shape=#{inspect(shape)} ms=#{inspect(Enum.map(times, &elem(&1, 0)))} pause_ms=#{pause_ms} input_bytes=#{byte_size(input)} output_bytes=#{byte_size(expected)} includes=invoke_and_readback"
        )

        IO.puts(
          "PATH_PHASES shape=#{inspect(shape)} invoke_ms=#{inspect(Enum.map(times, &elem(&1, 1)))} readback_ms=#{inspect(Enum.map(times, &elem(&1, 2)))}"
        )
      end

      [resident_input, resident_p1, resident_p2] = resident

      unless Nx.to_binary(resident_input) == input and Nx.to_number(resident_p1) == p1 and
               Nx.to_number(resident_p2) == p2,
             do: raise("#{name}: caller-owned input changed")

      IO.puts("INPUTS_UNCHANGED #{name}")
    after
      Enum.each(resident, &Nx.backend_deallocate/1)
    end
  end

  defp guard_resources! do
    for path <- Path.wildcard("/sys/class/thermal/thermal_zone*") do
      if File.read!(path <> "/temp") |> String.trim() |> String.to_integer() >= 70_000 do
        raise("thermal limit reached at #{path}")
      end
    end

    if minimum = System.get_env("STEREO_MIN_AVAILABLE_KB") do
      [_, available] = Regex.run(~r/MemAvailable:\s+(\d+)/, File.read!("/proc/meminfo"))
      IO.puts("MEM_AVAILABLE_KB=#{available}")

      if String.to_integer(available) < String.to_integer(minimum),
        do: raise("memory headroom guard")
    end
  end
end

StereoPathBench.run(bundle)
