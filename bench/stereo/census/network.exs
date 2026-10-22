for {variable, apps} <- [
      {"STEREO_RUNTIME_ROOT", ~w(complex telemetry nx nx_iree)},
      {"EVISION_RUNTIME_ROOT", ~w(castore evision)}
    ],
    root = System.get_env(variable),
    app <- apps,
    do: Code.append_path(Path.join([root, "lib", app, "ebin"]))

{:ok, _} = Application.ensure_all_started(:nx)

defmodule NetworkStereoBench do
  def run do
    count = System.get_env("VIDEO_FRAMES", "600") |> String.to_integer()
    fps = System.get_env("VIDEO_FPS", "60") |> String.to_integer()
    concurrency = System.get_env("VIDEO_CONCURRENCY", "1") |> String.to_integer()
    port = System.get_env("VIDEO_PORT", "8765") |> String.to_integer()
    matching = System.get_env("VIDEO_MATCH", "true")
    unless matching in ["true", "false"], do: raise("VIDEO_MATCH must be true or false")
    match? = matching == "true"

    unless count in 50..1200 and fps in 1..240 and port in 1..65535 and concurrency in 1..4,
      do: raise("invalid stream configuration")

    {:ok, address} =
      :inet.parse_address(String.to_charlist(System.get_env("VIDEO_BIND", "0.0.0.0")))

    bundle = System.fetch_env!("STEREO_BUNDLE") |> File.read!() |> :erlang.binary_to_term()
    {height, width} = bundle.shape

    servers =
      if match? do
        Enum.take(
          [
            __MODULE__,
            NetworkStereoBench.SecondContext,
            NetworkStereoBench.ThirdContext,
            NetworkStereoBench.FourthContext
          ],
          concurrency
        )
      end

    expected =
      System.fetch_env!("VIDEO_EXPECTED")
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(fn line ->
        [frame, input, output] = String.split(line)
        {frame, input, if(output == "-", do: nil, else: output)}
      end)

    50 = length(expected)

    supervisor =
      if servers do
        children =
          Enum.map(servers, fn name ->
            plain = serving(bundle)
            plain = if concurrency == 1, do: Nx.Serving.streaming(plain), else: plain

            Supervisor.child_spec({Nx.Serving, [name: name, serving: plain, batch_timeout: 0]},
              id: name
            )
          end)

        {:ok, pid} = Supervisor.start_link(children, strategy: :one_for_one)
        pid
      end

    try do
      {:ok, listener} =
        :gen_tcp.listen(port, [:binary, ip: address, active: false, reuseaddr: true])

      try do
        IO.puts(
          "NETWORK_LISTEN_READY port=#{port} width=#{width * 2} height=#{height} format=gray fps=#{fps} frames=#{count}"
        )

        {:ok, socket} = :gen_tcp.accept(listener, 15_000)

        try do
          IO.inspect(:inet.peername(socket), label: "NETWORK_PEER")
          measure(socket, servers, expected, count, fps, height, width)
        after
          :gen_tcp.close(socket)
        end
      after
        :gen_tcp.close(listener)
      end
    after
      if supervisor, do: Supervisor.stop(supervisor)
    end
  end

  defp serving(bundle) do
    {:ok, _} = Application.ensure_all_started(:evision)
    Evision.setNumThreads(1)
    {:ok, devices} = NxIREE.list_devices("vulkan")
    uri = System.get_env("NXIREE_DEVICE_URI")
    device = Enum.find(devices, &(is_nil(uri) or &1.uri == uri)) || raise("no matching GPU")
    model = NxIREE.load_exported(bundle.models["pipeline"], device: device)
    plain = NxIREE.Serving.new(model)
    zero = Nx.broadcast(Nx.tensor(0, type: :s32, backend: Nx.BinaryBackend), bundle.shape)

    for _ <- 1..3 do
      output = Nx.Serving.run(plain, Nx.Batch.stack([{zero, zero}]))
      Nx.to_binary(output)
      Nx.backend_deallocate(output)
    end

    plain
  end

  defp measure(socket, servers, expected, count, fps, height, width) do
    owner = self()
    started = now()
    initial_cpu = cpu_ticks()

    batches =
      Stream.map(0..(count - 1), fn index ->
        {:ok, frame} = :gen_tcp.recv(socket, height * width * 2, 5000)
        arrived = now()
        {name, input_hash, output_hash} = Enum.at(expected, rem(index, length(expected)))
        unless hash(frame) == input_hash, do: raise("network input disagreement: #{index}")
        guard!()
        send(owner, {:arrival, index, arrived, name, output_hash})

        if servers do
          mat = Evision.Mat.from_binary(frame, :u8, height, width * 2, 1)

          [left, right] =
            for x <- [0, width] do
              mat
              |> Evision.Mat.roi({x, 0, width, height})
              |> Evision.Mat.as_type(:s32)
              |> Evision.Mat.to_nx(Nx.BinaryBackend)
              |> Nx.reshape({height, width})
            end

          if index == 0,
            do:
              IO.inspect(Enum.map([left, right], &hash(Nx.to_binary(&1))),
                label: "NETWORK_CONVERTED_INPUT_HASHES"
              )

          Nx.Batch.stack([{left, right}])
        else
          :transport
        end
      end)

    results =
      case servers do
        nil ->
          batches

        [server] ->
          Nx.Serving.batched_run(server, batches)

        servers ->
          batches
          |> Stream.with_index()
          |> Task.async_stream(
            fn {batch, index} ->
              server = Enum.at(servers, rem(index, length(servers)))
              {:batch, Nx.Serving.batched_run(server, batch), nil}
            end,
            max_concurrency: length(servers),
            ordered: true,
            timeout: 15_000
          )
          |> Stream.map(fn {:ok, result} -> result end)
      end

    {metrics, predictions} =
      results
      |> Stream.with_index()
      |> Enum.reduce({[], %{}}, fn {result, index}, {metrics, predictions} ->
        binary =
          case result do
            {:batch, output, _} ->
              try do
                Nx.to_binary(output)
              after
                Nx.backend_deallocate(output)
              end

            :transport ->
              <<>>
          end

        completed = now()

        receive do
          {:arrival, ^index, arrived, name, expected_hash} ->
            if not is_nil(servers) and not is_nil(expected_hash) and hash(binary) != expected_hash do
              output = System.fetch_env!("VIDEO_OUTPUT")
              File.mkdir_p!(output)
              File.write!(Path.join(output, "mismatch-#{index}.disparity.bin"), binary)
              raise("GPU output disagreement: #{index}")
            end

            {[{index, arrived, completed} | metrics], Map.put(predictions, name, binary)}
        after
          1000 -> raise "missing arrival"
        end
      end)

    wall = now() - started
    used_cpu = cpu_ticks() - initial_cpu
    metrics = Enum.reverse(metrics)
    ^count = length(metrics)
    [{0, first, first_complete} | _] = metrics
    {_, last, last_complete} = List.last(metrics)
    period = 1_000_000 / fps
    output = System.fetch_env!("VIDEO_OUTPUT")
    File.mkdir_p!(output)

    for {name, binary} <- predictions,
        servers,
        do: File.write!(Path.join(output, name <> ".disparity.bin"), binary)

    rows =
      Enum.map(metrics, fn {index, arrived, completed} ->
        "#{index},#{arrived - first},#{completed - arrived},#{arrived - first - round(index * period)}\n"
      end)

    File.write!(Path.join(output, "timing.csv"), [
      "frame_index,arrival_relative_us,processing_latency_us,arrival_lateness_us\n" | rows
    ])

    for {name, values} <- [
          {"processing", Enum.map(metrics, fn {_, a, c} -> c - a end)},
          {"scheduled_completion",
           Enum.map(metrics, fn {i, _, c} -> c - first - round(i * period) end)},
          {"arrival_lateness",
           Enum.map(metrics, fn {i, a, _} -> a - first - round(i * period) end)}
        ] do
      values = Enum.sort(values)

      IO.puts(
        "NETWORK_#{name} p50_ms=#{Enum.at(values, ceil(count * 0.5) - 1) / 1000} p95_ms=#{Enum.at(values, ceil(count * 0.95) - 1) / 1000} max_ms=#{List.last(values) / 1000}"
      )
    end

    IO.puts(
      "NETWORK_RESULT matching=#{not is_nil(servers)} contexts=#{if servers, do: length(servers), else: 0} frames=#{count} unique=#{map_size(predictions)} fps=#{count * 1_000_000 / wall} arrival_fps=#{(count - 1) * 1_000_000 / (last - first)} completion_fps=#{(count - 1) * 1_000_000 / (last_complete - first_complete)} wall_ms=#{wall / 1000} cpu_cores=#{used_cpu * 10_000 / wall} cpu_ticks=#{used_cpu}"
    )

    hashes_checked? =
      not is_nil(servers) and Enum.all?(expected, fn {_, _, hash} -> is_binary(hash) end)

    IO.puts("NETWORK_INPUTS_PASS frames=#{count} gpu_outputs_checked=#{hashes_checked?}")
    IO.puts("NETWORK_STEREO_PASS")
  end

  defp hash(data), do: :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)
  defp now, do: System.monotonic_time(:microsecond)

  defp cpu_ticks do
    [_, rest] = String.split(File.read!("/proc/self/stat"), ") ", parts: 2)
    fields = String.split(rest)
    String.to_integer(Enum.at(fields, 11)) + String.to_integer(Enum.at(fields, 12))
  end

  defp guard! do
    for path <- Path.wildcard("/sys/class/thermal/thermal_zone*/temp") do
      if String.to_integer(String.trim(File.read!(path))) >= 70_000, do: raise("thermal limit")
    end

    if minimum = System.get_env("STEREO_MIN_AVAILABLE_KB") do
      [_, available] = Regex.run(~r/MemAvailable:\s+(\d+)/, File.read!("/proc/meminfo"))

      if String.to_integer(available) < String.to_integer(minimum),
        do: raise("memory headroom guard")
    end
  end
end

NetworkStereoBench.run()
