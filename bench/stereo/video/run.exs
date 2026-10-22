if root = System.get_env("STEREO_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx nx_iree),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:nx)

defmodule StereoVideoBench do
  def run do
    bundle = System.fetch_env!("STEREO_BUNDLE") |> File.read!() |> :erlang.binary_to_term()
    directory = System.fetch_env!("VIDEO_DATA")
    output = System.fetch_env!("VIDEO_OUTPUT")
    File.mkdir_p!(output)
    {:ok, devices} = NxIREE.list_devices("vulkan")

    device =
      case {System.get_env("NXIREE_DEVICE_URI"), devices} do
        {nil, [device]} ->
          device

        {uri, _} when is_binary(uri) ->
          Enum.find(devices, &(&1.uri == uri)) || raise("device not found")

        _ ->
          raise("choose NXIREE_DEVICE_URI from #{inspect(devices)}")
      end

    models =
      Map.new(bundle.models, fn {name, binary} ->
        {name, NxIREE.load_exported(binary, device: device)}
      end)

    prefixes =
      directory
      |> Path.join("*.left.bin")
      |> Path.wildcard()
      |> Enum.sort()
      |> Enum.map(&String.replace_suffix(&1, ".left.bin", ""))

    if prefixes == [], do: raise("no stereo frames in #{directory}")

    IO.puts(
      "VIDEO_DEVICE=#{device.uri} shape=#{inspect(bundle.shape)} disparities=#{bundle.disparities} frames=#{length(prefixes)}"
    )

    first = hd(prefixes)
    first_pair = read_pair(first, bundle.shape)

    for prefix <- prefixes,
        File.exists?(prefix <> ".cost_left.bin"),
        do: check_intermediates(prefix, {models, device, bundle})

    # Pipeline creation, initial allocations and frequency ramp are warmup costs.
    for _ <- 1..3, do: process(first_pair, models, device)

    times =
      for prefix <- prefixes do
        guard_resources!()
        pair = read_pair(prefix, bundle.shape)
        {us, {binary, phases}} = :timer.tc(fn -> process(pair, models, device) end)
        validate!(prefix, binary)
        File.write!(Path.join(output, Path.basename(prefix) <> ".disparity.bin"), binary)

        IO.puts(
          "VIDEO_FRAME=#{Path.basename(prefix)} total_ms=#{us / 1000} phases_ms=#{inspect(phases)}"
        )

        us / 1000
      end

    File.write!(
      Path.join(output, "timings_ms.txt"),
      Enum.map_join(times, "\n", &to_string/1) <> "\n"
    )

    sorted = Enum.sort(times)
    percentile = fn p -> Enum.at(sorted, max(0, ceil(p * length(sorted)) - 1)) end

    IO.puts(
      "VIDEO_RESULT fps=#{1000 * length(times) / Enum.sum(times)} p50_ms=#{percentile.(0.5)} p95_ms=#{percentile.(0.95)} p99_ms=#{percentile.(0.99)} includes=upload_cost_16paths_select_consistency_readback excludes=decode_disk_io_validation warmup=3"
    )

    IO.puts("STEREO_PATH_PASS")
  end

  defp read_pair(prefix, shape) do
    for side <- ["left", "right"] do
      binary = File.read!(prefix <> ".#{side}.bin")

      for <<value::native-signed-32 <- binary>> do
        unless value in 0..255, do: raise("stereo inputs must contain grayscale values 0..255")
      end

      binary |> Nx.from_binary(:s32) |> Nx.reshape(shape)
    end
  end

  defp process(pair, models, device) do
    {upload, inputs} =
      :timer.tc(fn -> Enum.map(pair, &Nx.backend_copy(&1, {NxIREE.Backend, device: device})) end)

    try do
      {left, lt} = view(inputs, models, "left")

      try do
        {right, rt} = view(inputs, models, "right")

        try do
          {check, checked} = :timer.tc(fn -> NxIREE.run(models["consistent"], [left, right]) end)

          {readback, binary} =
            :timer.tc(fn ->
              checked |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_binary()
            end)

          {binary,
           %{
             upload: upload / 1000,
             left: lt,
             right: rt,
             consistency: check / 1000,
             readback: readback / 1000
           }}
        after
          Nx.backend_deallocate(right)
        end
      after
        Nx.backend_deallocate(left)
      end
    after
      Enum.each(inputs, &Nx.backend_deallocate/1)
    end
  end

  defp view(inputs, models, side) do
    {cost_us, cost} = :timer.tc(fn -> NxIREE.run(models["cost_#{side}"], inputs) end)

    try do
      {path_us, total} =
        :timer.tc(fn ->
          first = NxIREE.run(models["path0"], [cost, cost])

          last = if Map.has_key?(models, "path7_sum"), do: 6, else: 7

          Enum.reduce(1..last, first, fn index, previous ->
            try do
              NxIREE.run(models["path#{index}"], [cost, previous])
            after
              Nx.backend_deallocate(previous)
            end
          end)
        end)

      try do
        {select_us, disparity} =
          :timer.tc(fn ->
            if Map.has_key?(models, "path7_sum"),
              do: NxIREE.run(models["path7"], [cost, total]),
              else: NxIREE.run(models["select"], [total])
          end)

        {disparity, %{cost: cost_us / 1000, paths: path_us / 1000, select: select_us / 1000}}
      after
        Nx.backend_deallocate(total)
      end
    after
      Nx.backend_deallocate(cost)
    end
  end

  defp validate!(prefix, binary) do
    if File.exists?(prefix <> ".expected.bin") do
      expected =
        File.read!(prefix <> ".expected.bin") |> Nx.from_binary(:f32) |> Nx.to_flat_list()

      actual = binary |> Nx.from_binary(:f32) |> Nx.to_flat_list()

      unless length(expected) == length(actual) and
               Enum.zip(expected, actual) |> Enum.all?(fn {a, b} -> abs(a - b) < 0.00001 end),
             do: raise("#{prefix}: CPU oracle disagreement")

      IO.puts("VIDEO_ORACLE_PASS #{Path.basename(prefix)}")
    end
  end

  defp check_intermediates(prefix, {models, device, bundle}) do
    shape = bundle.shape

    inputs =
      Enum.map(read_pair(prefix, shape), &Nx.backend_copy(&1, {NxIREE.Backend, device: device}))

    try do
      for side <- ["left", "right"] do
        cost = NxIREE.run(models["cost_#{side}"], inputs)

        try do
          if oracle_binary(cost, bundle, 0) != File.read!(prefix <> ".cost_#{side}.bin"),
            do: raise("cost oracle mismatch: #{side}")

          first = NxIREE.run(models["path0"], [cost, cost])

          total =
            Enum.reduce(1..7, first, fn index, previous ->
              try do
                model =
                  if index == 7,
                    do: models["path7_sum"] || models["path7"],
                    else: models["path#{index}"]

                NxIREE.run(model, [cost, previous])
              after
                Nx.backend_deallocate(previous)
              end
            end)

          try do
            if oracle_binary(total, bundle, 8) != File.read!(prefix <> ".aggregate_#{side}.bin"),
              do: raise("eight-path oracle mismatch: #{side}")
          after
            Nx.backend_deallocate(total)
          end
        after
          Nx.backend_deallocate(cost)
        end
      end

      Enum.zip(inputs, read_pair(prefix, shape))
      |> Enum.each(fn {resident, original} ->
        if Nx.to_binary(resident) != Nx.to_binary(original), do: raise("caller input changed")
      end)

      IO.puts("VIDEO_INTERMEDIATES_PASS #{Path.basename(prefix)}")
    after
      Enum.each(inputs, &Nx.backend_deallocate/1)
    end
  end

  defp oracle_binary(tensor, %{storage: "packed16", disparities: disparities}, paths) do
    words_per_pixel = div(disparities + 1, 2)

    tensor
    |> Nx.to_binary()
    |> then(fn binary ->
      for <<pixel::binary-size(words_per_pixel * 4) <- binary>>, into: <<>> do
        values = for <<value::native-unsigned-16 <- pixel>>, do: value

        for value <- Enum.take(values, disparities), into: <<>> do
          decoded =
            cond do
              paths == 0 and value == 65_535 -> 1_000_000
              paths > 0 and value >= 32_768 -> paths * 1_000_000 + value - 32_768
              true -> value
            end

          <<decoded::native-signed-32>>
        end
      end
    end)
  end

  defp oracle_binary(tensor, _bundle, _paths), do: Nx.to_binary(tensor)

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

StereoVideoBench.run()
