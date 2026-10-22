if root = System.get_env("STEREO_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx nx_iree),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:nx)

defmodule PipelineThroughput do
  def run do
    bundle = System.fetch_env!("STEREO_BUNDLE") |> File.read!() |> :erlang.binary_to_term()
    inflight = System.get_env("VIDEO_CONCURRENCY", "1") |> String.to_integer()
    unless inflight in [1, 2, 4], do: raise("use one, two or four independent contexts")
    repeats = System.get_env("VIDEO_REPEATS", "1") |> String.to_integer()
    unless repeats in 1..20, do: raise("VIDEO_REPEATS must be 1..20")
    {:ok, devices} = NxIREE.list_devices("vulkan")
    uri = System.get_env("NXIREE_DEVICE_URI")
    device = Enum.find(devices, &(is_nil(uri) or &1.uri == uri)) || raise("no matching GPU")

    models =
      for _ <- 1..inflight, do: NxIREE.load_exported(bundle.models["pipeline"], device: device)

    pairs =
      System.fetch_env!("VIDEO_DATA")
      |> Path.join("*.left.bin")
      |> Path.wildcard()
      |> Enum.sort()
      |> Enum.map(fn file ->
        prefix = String.replace_suffix(file, ".left.bin", "")

        pair =
          for side <- ~w(left right),
              do:
                File.read!(prefix <> "." <> side <> ".bin")
                |> Nx.from_binary(:s32)
                |> Nx.reshape(bundle.shape)

        {Path.basename(prefix), pair}
      end)

    50 = length(pairs)
    frame_count = 50 * repeats

    IO.puts(
      "PIPELINE_DEVICE=#{device.uri} inflight=#{inflight} frames=#{frame_count} repeats=#{repeats}"
    )

    for model <- models, _ <- 1..3, do: process(elem(hd(pairs), 1), model, device)
    pairs = Enum.flat_map(1..repeats, fn _ -> pairs end)
    chunks = Enum.chunk_every(pairs, ceil(length(pairs) / inflight))

    parent = self()
    pulse = spawn_link(fn -> heartbeat(parent, System.monotonic_time(:microsecond), []) end)
    receive do: ({:heartbeat_ready, ^pulse} -> :ok)

    work = fn ->
      cpu_start = cpu_ticks()

      {wall_us, results} =
        :timer.tc(fn ->
          Enum.zip(models, chunks)
          |> Task.async_stream(
            fn {model, chunk} ->
              for {frame, pair} <- chunk do
                for path <- Path.wildcard("/sys/class/thermal/thermal_zone*/temp") do
                  if String.to_integer(String.trim(File.read!(path))) >= 70_000,
                    do: raise("thermal limit")
                end

                if minimum = System.get_env("STEREO_MIN_AVAILABLE_KB") do
                  [_, available] =
                    Regex.run(~r/MemAvailable:\s+(\d+)/, File.read!("/proc/meminfo"))

                  if String.to_integer(available) < String.to_integer(minimum),
                    do: raise("memory headroom guard")
                end

                {us, binary} = :timer.tc(fn -> process(pair, model, device) end)
                {frame, binary, us / 1000}
              end
            end,
            max_concurrency: inflight,
            ordered: false,
            timeout: 60_000
          )
          |> Enum.flat_map(fn {:ok, frames} -> frames end)
        end)

      cpu_end = cpu_ticks()

      IO.puts(
        "PIPELINE_CPU_TICKS=#{inspect(if(cpu_start && cpu_end, do: cpu_end - cpu_start))} source=linux_proc_self_stat_user_plus_system"
      )

      {wall_us, results}
    end

    {wall_us, results} =
      case System.get_env("STEREO_PROFILE") do
        nil -> work.()
        path -> NxIREE.Profiling.capture(device, path, work)
      end

    send(pulse, :stop)
    intervals = receive do: ({:heartbeat_intervals, ^pulse, values} -> Enum.sort(values))

    IO.puts(
      "PIPELINE_HEARTBEAT count=#{length(intervals)} p95_us=#{Enum.at(intervals, ceil(length(intervals) * 0.95) - 1)} max_us=#{List.last(intervals)}"
    )

    ^frame_count = length(results)
    output = System.fetch_env!("VIDEO_OUTPUT")
    File.mkdir_p!(output)

    for {frame, copies} <- Enum.group_by(results, &elem(&1, 0)) do
      [{_, binary, ms} | _] = copies

      unless Enum.all?(copies, &(elem(&1, 1) == binary)),
        do: raise("repeated output disagreement: #{frame}")

      File.write!(Path.join(output, frame <> ".disparity.bin"), binary)
      IO.puts("PIPELINE_FRAME=#{frame} latency_ms=#{ms} occurrences=#{length(copies)}")
    end

    IO.puts("PIPELINE_REPEAT_OUTPUTS_PASS frames=#{frame_count} unique=50")

    times = results |> Enum.map(&elem(&1, 2)) |> Enum.sort()

    IO.puts(
      "PIPELINE_RESULT inflight=#{inflight} fps=#{frame_count * 1_000_000 / wall_us} wall_ms=#{wall_us / 1000} latency_p50_ms=#{Enum.at(times, ceil(frame_count * 0.5) - 1)} latency_p95_ms=#{Enum.at(times, ceil(frame_count * 0.95) - 1)} includes=upload_graph_readback_synchronization_pool_scheduling_guards excludes=disk_io_input_preparation warmup_per_context=3"
    )

    IO.puts("PIPELINE_THROUGHPUT_PASS")
  end

  defp process(pair, model, device) do
    inputs = Enum.map(pair, &Nx.backend_copy(&1, {NxIREE.Backend, device: device}))

    try do
      output = NxIREE.run(model, inputs)

      try do
        Nx.to_binary(output)
      after
        Nx.backend_deallocate(output)
      end
    after
      Enum.each(inputs, &Nx.backend_deallocate/1)
    end
  end

  defp cpu_ticks do
    case File.read("/proc/self/stat") do
      {:ok, stat} ->
        [_, fields] = String.split(stat, ") ", parts: 2)
        fields = String.split(fields)
        String.to_integer(Enum.at(fields, 11)) + String.to_integer(Enum.at(fields, 12))

      {:error, _} ->
        nil
    end
  end

  defp heartbeat(parent, previous, intervals) do
    if intervals == [], do: send(parent, {:heartbeat_ready, self()})

    receive do
      :stop -> send(parent, {:heartbeat_intervals, self(), intervals})
    after
      1 ->
        now = System.monotonic_time(:microsecond)
        heartbeat(parent, now, [now - previous | intervals])
    end
  end
end

PipelineThroughput.run()
