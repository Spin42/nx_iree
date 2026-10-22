if root = System.get_env("STEREO_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx nx_iree),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:nx)

defmodule CensusStereoBench do
  def run do
    bundle = System.fetch_env!("STEREO_BUNDLE") |> File.read!() |> :erlang.binary_to_term()
    {:ok, devices} = NxIREE.list_devices("vulkan")
    uri = System.get_env("NXIREE_DEVICE_URI")

    device =
      case {uri, devices} do
        {nil, [device]} ->
          device

        {uri, _} when is_binary(uri) ->
          Enum.find(devices, &(&1.uri == uri)) || raise("device not found: #{uri}")

        _ ->
          raise("choose NXIREE_DEVICE_URI")
      end

    Process.put(:stereo_bundle, bundle)
    models = Map.new(bundle.models, fn {k, v} -> {k, NxIREE.load_exported(v, device: device)} end)

    prefixes =
      System.fetch_env!("VIDEO_DATA")
      |> Path.join("*.left.bin")
      |> Path.wildcard()
      |> Enum.sort()
      |> Enum.map(&String.replace_suffix(&1, ".left.bin", ""))

    if prefixes == [], do: raise("no input frames")
    output = System.fetch_env!("VIDEO_OUTPUT")
    File.mkdir_p!(output)

    IO.puts(
      "CENSUS_DEVICE=#{device.uri} shape=#{inspect(bundle.shape)} parameters=#{inspect(Map.take(bundle, [:paths, :disparities, :p1, :p2, :uniqueness, :block, :selector, :select_lanes, :schedule, :census, :execution, :projection]))}"
    )

    first = read_pair(hd(prefixes), bundle.shape)

    for prefix <- prefixes, File.exists?(prefix <> ".output.oracle.bin") do
      process(read_pair(prefix, bundle.shape), {bundle, models, device}, prefix)
      IO.puts("CENSUS_ORACLE_PASS #{Path.basename(prefix)}")
    end

    for _ <- 1..3, do: process(first, {bundle, models, device})

    times =
      for prefix <- prefixes do
        guard_resources!()
        pair = read_pair(prefix, bundle.shape)
        {us, {binary, phases}} = :timer.tc(fn -> process(pair, {bundle, models, device}) end)
        File.write!(Path.join(output, Path.basename(prefix) <> ".disparity.bin"), binary)

        IO.puts(
          "CENSUS_FRAME=#{Path.basename(prefix)} total_ms=#{us / 1000} phases_ms=#{inspect(phases)}"
        )

        us / 1000
      end

    File.write!(
      Path.join(output, "timings_ms.txt"),
      Enum.map_join(times, "\n", &to_string/1) <> "\n"
    )

    sorted = Enum.sort(times)

    IO.puts(
      "CENSUS_RESULT fps=#{1000 * length(times) / Enum.sum(times)} p50_ms=#{Enum.at(sorted, ceil(length(times) * 0.5) - 1)} p95_ms=#{Enum.at(sorted, ceil(length(times) * 0.95) - 1)} includes=upload_census_paths_selection_median_consistency_readback excludes=decode_disk_io warmup=3"
    )

    IO.puts("CENSUS_STEREO_PASS")
  end

  defp read_pair(prefix, shape) do
    Map.new(["left", "right"], fn side ->
      binary = File.read!(prefix <> ".#{side}.bin")

      for <<value::native-signed-32 <- binary>> do
        unless value in 0..255, do: raise("expected grayscale 0..255")
      end

      {side, binary |> Nx.from_binary(:s32) |> Nx.reshape(shape)}
    end)
  end

  defp process(pair, {bundle, models, device}, oracle \\ nil) do
    {upload, inputs} =
      :timer.tc(fn ->
        Map.new(pair, fn {k, v} ->
          {k, Nx.backend_copy(v, {NxIREE.Backend, device: device})}
        end)
      end)

    # Keep ownership explicit even when a downstream dispatch fails.
    try do
      result =
        if bundle[:execution] == "graph" and is_nil(oracle) do
          graph(models["pipeline"], inputs, %{upload: upload / 1000})
        else
          pipeline(bundle.pipeline, {inputs, %{upload: upload / 1000}, oracle}, models)
        end

      if not is_nil(oracle) and bundle[:execution] == "graph" do
        {expected, _} = result
        {actual, _} = graph(models["pipeline"], inputs, %{})
        if actual != expected, do: raise("graph and separate stages disagree")
      end

      if oracle do
        for {name, tensor} <- inputs do
          if Nx.to_binary(tensor) != Nx.to_binary(pair[name]), do: raise("caller input changed")
        end
      end

      result
    after
      Enum.each(inputs, fn {_, tensor} -> Nx.backend_deallocate(tensor) end)
    end
  end

  defp graph(model, inputs, phases) do
    {us, output} = :timer.tc(fn -> NxIREE.run(model, [inputs["left"], inputs["right"]]) end)

    try do
      {readback, binary} = :timer.tc(fn -> Nx.to_binary(output) end)
      {binary, phases |> Map.put(:graph, us / 1000) |> Map.put(:readback, readback / 1000)}
    after
      Nx.backend_deallocate(output)
    end
  end

  defp pipeline([], {values, phases, _oracle}, _models) do
    {readback, binary} = :timer.tc(fn -> Nx.to_binary(values["output"]) end)
    {binary, Map.put(phases, :readback, readback / 1000)}
  end

  defp pipeline([{name, inputs, outputs} | rest], {values, phases, oracle}, models) do
    bundle = Process.get(:stereo_bundle)
    {us, result} =
      :timer.tc(fn -> NxIREE.run(models[name], Enum.map(inputs, &Map.fetch!(values, &1))) end)

    tensors = if is_tuple(result), do: Tuple.to_list(result), else: [result]

    try do
      if oracle do
        for {output, tensor} <- Enum.zip(outputs, tensors) do
          expected =
            if output == "features" do
              # The interleaved feature oracle is the pair of per-image oracles.
              {h, w} = bundle.shape
              words = if bundle.census == "classic", do: 2, else: 1

              ["left_features", "right_features"]
              |> Enum.map(fn side ->
                File.read!(oracle <> ".#{side}.oracle.bin")
                |> Nx.from_binary(:u32)
                |> Nx.reshape({h, w, words})
              end)
              |> Nx.concatenate(axis: 2)
              |> Nx.to_binary()
            else
              File.read!(oracle <> ".#{output}.oracle.bin")
            end

          actual = Nx.to_binary(tensor)

          expected =
            if output == "paths", do: binary_part(expected, 0, byte_size(actual)), else: expected

          # Row-tiled feature and byte volumes are compared in plain row-major order.
          actual =
            case {bundle[:volume_tile], Nx.shape(tensor)} do
              {tile, {rows, columns, words}}
              when is_integer(tile) and rem(rows, tile) == 0 and rows >= elem(bundle.shape, 0) and
                     rows < elem(bundle.shape, 0) + tile ->
                tensor
                |> Nx.reshape({div(rows, tile), columns, tile, words})
                |> Nx.transpose(axes: [0, 2, 1, 3])
                |> Nx.reshape({rows, columns, words})
                |> Nx.slice_along_axis(0, elem(bundle.shape, 0), axis: 0)
                |> Nx.to_binary()

              _ ->
                actual
            end

          if Nx.type(tensor) == {:f, 32} do
            lhs = for <<v::native-float-32 <- expected>>, do: v
            rhs = for <<v::native-float-32 <- actual>>, do: v

            differences =
              Enum.zip(lhs, rhs)
              |> Enum.with_index()
              |> Enum.filter(fn {{a, b}, _} -> abs(a - b) >= 0.00001 end)

            unless length(lhs) == length(rhs) and differences == [],
              do:
                raise(
                  "oracle disagreement: #{output} first=#{inspect(Enum.take(differences, 5))}"
                )
          else
            unless expected == actual, do: raise("oracle disagreement: #{output}")
          end
        end
      end

      values = Map.merge(values, Map.new(Enum.zip(outputs, tensors)))
      pipeline(rest, {values, Map.put(phases, name, us / 1000), oracle}, models)
    after
      Enum.each(tensors, &Nx.backend_deallocate/1)
    end
  end

  defp guard_resources! do
    for path <- Path.wildcard("/sys/class/thermal/thermal_zone*/temp") do
      if File.read!(path) |> String.trim() |> String.to_integer() >= 70_000,
        do: raise("thermal limit: #{path}")
    end

    if minimum = System.get_env("STEREO_MIN_AVAILABLE_KB") do
      [_, available] = Regex.run(~r/MemAvailable:\s+(\d+)/, File.read!("/proc/meminfo"))

      if String.to_integer(available) < String.to_integer(minimum),
        do: raise("memory headroom guard")
    end
  end
end

CensusStereoBench.run()
