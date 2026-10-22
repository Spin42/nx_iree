if root = System.get_env("STEREO_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx nx_iree),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:nx)

defmodule StereoServingBench do
  def run do
    repeats = System.get_env("VIDEO_REPEATS", "20") |> String.to_integer()
    fps = System.get_env("VIDEO_FPS", "100") |> String.to_integer()
    unless repeats in 1..20 and fps in 1..240, do: raise("invalid repeat count or frame rate")
    bundle = System.fetch_env!("STEREO_BUNDLE") |> File.read!() |> :erlang.binary_to_term()
    {:ok, devices} = NxIREE.list_devices("vulkan")
    uri = System.get_env("NXIREE_DEVICE_URI")
    device = Enum.find(devices, &(is_nil(uri) or &1.uri == uri)) || raise("no matching GPU")
    model = NxIREE.load_exported(bundle.models["pipeline"], device: device)

    pairs =
      System.fetch_env!("VIDEO_DATA")
      |> Path.join("*.left.bin")
      |> Path.wildcard()
      |> Enum.sort()
      |> Enum.map(fn file ->
        prefix = String.replace_suffix(file, ".left.bin", "")

        [left, right] =
          for side <- ~w(left right) do
            File.read!(prefix <> "." <> side <> ".bin")
            |> Nx.from_binary(:s32, backend: Nx.BinaryBackend)
            |> Nx.reshape(bundle.shape)
          end

        {Path.basename(prefix), {left, right}}
      end)

    50 = length(pairs)
    plain = NxIREE.Serving.new(model)

    for _ <- 1..3 do
      output = Nx.Serving.run(plain, Nx.Batch.stack([elem(hd(pairs), 1)]))
      Nx.to_binary(output)
      Nx.backend_deallocate(output)
    end

    {:ok, supervisor} =
      Nx.Serving.start_link(
        name: __MODULE__,
        serving: Nx.Serving.streaming(plain),
        batch_timeout: 0
      )

    try do
      measure(pairs, repeats, fps, device.uri)
    after
      Supervisor.stop(supervisor)
    end
  end

  defp measure(pairs, repeats, fps, device_uri) do
    owner = self()
    period = 1_000_000 / fps
    count = length(pairs) * repeats
    started = now()
    initial_cpu = cpu_ticks()

    batches =
      Stream.cycle(pairs)
      |> Stream.take(count)
      |> Stream.with_index()
      |> Stream.map(fn {{frame, pair}, index} ->
        target = started + round(index * period)
        remaining = target - now()
        if remaining > 0, do: Process.sleep(ceil(remaining / 1000))
        generated = now()
        guard!()
        send(owner, {:frame_arrived, index, frame, target, generated})
        Nx.Batch.stack([pair])
      end)

    {metrics, predictions} =
      Nx.Serving.batched_run(__MODULE__, batches)
      |> Stream.with_index()
      |> Enum.reduce({[], %{}}, fn {{:batch, output, _metadata}, index}, {metrics, predictions} ->
        binary =
          try do
            Nx.to_binary(output)
          after
            Nx.backend_deallocate(output)
          end

        completed = now()

        receive do
          {:frame_arrived, ^index, frame, target, generated} ->
            if previous = predictions[frame] do
              unless previous == binary, do: raise("repeated output disagreement: #{frame}")
            end

            metric = {generated - target, completed - generated, completed - target}
            {[metric | metrics], Map.put(predictions, frame, binary)}
        after
          1000 -> raise "missing frame arrival"
        end
      end)

    wall = now() - started
    used_cpu = cpu_ticks() - initial_cpu
    ^count = length(metrics)
    50 = map_size(predictions)
    output = System.fetch_env!("VIDEO_OUTPUT")
    File.mkdir_p!(output)

    for {frame, binary} <- predictions,
        do: File.write!(Path.join(output, frame <> ".disparity.bin"), binary)

    rows =
      metrics
      |> Enum.reverse()
      |> Enum.with_index()
      |> Enum.map(fn {{arrival, latency, scheduled}, index} ->
        "#{index},#{arrival},#{latency},#{scheduled}\n"
      end)

    File.write!(Path.join(output, "timing.csv"), [
      "frame_index,arrival_lateness_us,latency_us,scheduled_latency_us\n" | rows
    ])

    reference = System.get_env("VIDEO_REFERENCE")

    if reference do
      for {frame, binary} <- predictions do
        unless File.read!(Path.join(reference, frame <> ".disparity.bin")) == binary,
          do: raise("reference output disagreement: #{frame}")
      end
    end

    IO.puts("SERVING_DEVICE=#{device_uri} frames=#{count} target_fps=#{fps}")

    IO.puts(
      "SERVING_RESULT fps=#{count * 1_000_000 / wall} wall_ms=#{wall / 1000} cpu_ticks=#{used_cpu} " <>
        "cpu_cores=#{used_cpu * 10_000 / wall} completed_within_period=#{Enum.count(metrics, &(elem(&1, 2) <= period))}"
    )

    for {name, index} <- [{"arrival_lateness", 0}, {"latency", 1}, {"scheduled_latency", 2}] do
      values = metrics |> Enum.map(&elem(&1, index)) |> Enum.sort()

      IO.puts(
        "SERVING_#{name} p50_ms=#{Enum.at(values, ceil(count * 0.5) - 1) / 1000} " <>
          "p95_ms=#{Enum.at(values, ceil(count * 0.95) - 1) / 1000} max_ms=#{List.last(values) / 1000}"
      )
    end

    IO.puts("SERVING_OUTPUTS_PASS total=#{count} unique=50 reference=#{not is_nil(reference)}")
    IO.puts("STEREO_SERVING_PASS")
  end

  defp guard! do
    for path <- Path.wildcard("/sys/class/thermal/thermal_zone*/temp") do
      if String.to_integer(String.trim(File.read!(path))) >= 70_000,
        do: raise("thermal limit")
    end

    if minimum = System.get_env("STEREO_MIN_AVAILABLE_KB") do
      [_, available] = Regex.run(~r/MemAvailable:\s+(\d+)/, File.read!("/proc/meminfo"))

      if String.to_integer(available) < String.to_integer(minimum),
        do: raise("memory headroom guard")
    end
  end

  defp now, do: System.monotonic_time(:microsecond)

  defp cpu_ticks do
    [_, fields] = String.split(File.read!("/proc/self/stat"), ") ", parts: 2)
    fields = String.split(fields)
    String.to_integer(Enum.at(fields, 11)) + String.to_integer(Enum.at(fields, 12))
  end
end

StereoServingBench.run()
