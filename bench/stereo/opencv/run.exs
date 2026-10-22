if root = System.get_env("EVISION_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx castore evision),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:evision)

defmodule OpenCVStereoBench do
  @width System.get_env("VIDEO_WIDTH", "320") |> String.to_integer()
  @height System.get_env("VIDEO_HEIGHT", "136") |> String.to_integer()

  def run do
    unless @width > 64 and @height > 0, do: raise("invalid stereo dimensions")
    data = System.fetch_env!("VIDEO_DATA")
    output = System.fetch_env!("VIDEO_OUTPUT")
    threads = System.get_env("OPENCV_THREADS", "4") |> String.to_integer()
    repeats = System.get_env("VIDEO_REPEATS", "1") |> String.to_integer()
    unless repeats in 1..20, do: raise("VIDEO_REPEATS must be 1..20")
    Evision.setNumThreads(threads)

    pairs =
      data
      |> Path.join("*.left.bin")
      |> Path.wildcard()
      |> Enum.sort()
      |> Enum.map(fn file ->
        prefix = String.replace_suffix(file, ".left.bin", "")
        {Path.basename(prefix), gray(prefix <> ".left.bin"), gray(prefix <> ".right.bin")}
      end)

    unless length(pairs) == 50, do: raise("expected the complete 50-frame sequence")
    frame_count = 50 * repeats
    pairs = Enum.flat_map(1..repeats, fn _ -> pairs end)
    IO.puts("OPENCV_THREADS=#{Evision.getNumThreads()} shape=#{@height}x#{@width} disparities=64")
    IO.puts("OPENCV_FRAMES=#{frame_count} repeats=#{repeats}")

    modes = [{"sgbm", 0}, {"hh", 1}, {"3way", 2}, {"hh4", 3}]
    requested = System.get_env("OPENCV_MODES", "sgbm,hh,3way,hh4") |> String.split(",")

    unless requested != [] and Enum.all?(requested, &List.keymember?(modes, &1, 0)),
      do: raise("unknown OpenCV mode")

    for {name, mode} <- Enum.filter(modes, &(elem(&1, 0) in requested)) do
      directory = Path.join(output, name)
      File.mkdir_p!(directory)

      %Evision.StereoSGBM{} =
        matcher =
        Evision.StereoSGBM.create(
          minDisparity: 0,
          numDisparities: 64,
          blockSize: 3,
          P1: 72,
          P2: 288,
          disp12MaxDiff: 1,
          preFilterCap: 31,
          uniquenessRatio: 10,
          speckleWindowSize: 0,
          speckleRange: 0,
          mode: mode
        )

      # The published 1.0.0 binding ignores uppercase constructor penalty keys.
      Evision.StereoSGBM.setP1(matcher, 72)
      Evision.StereoSGBM.setP2(matcher, 288)

      for {function, expected} <- [
            getP1: 72,
            getP2: 288,
            getNumDisparities: 64,
            getBlockSize: 3,
            getMode: mode
          ] do
        actual = apply(Evision.StereoSGBM, function, [matcher])
        unless actual == expected, do: raise("#{function}: expected #{expected}, got #{actual}")
      end

      for _ <- 1..3, do: process(hd(pairs), matcher)

      cpu_start = cpu_ticks()

      {wall_us, results} =
        :timer.tc(fn ->
          for {frame, _, _} = pair <- pairs do
            guard_resources!()
            {us, binary} = :timer.tc(fn -> process(pair, matcher) end)
            unless byte_size(binary) == @width * @height * 4, do: raise("incorrect output shape")
            {frame, binary, us / 1000}
          end
        end)

      cpu_end = cpu_ticks()

      IO.puts(
        "OPENCV_CPU_TICKS mode=#{name} ticks=#{inspect(if(cpu_start && cpu_end, do: cpu_end - cpu_start))} source=linux_proc_self_stat_user_plus_system"
      )

      ^frame_count = length(results)

      for {frame, copies} <- Enum.group_by(results, &elem(&1, 0)) do
        [{_, binary, ms} | _] = copies

        unless Enum.all?(copies, &(elem(&1, 1) == binary)),
          do: raise("repeated output disagreement: #{frame}")

        File.write!(Path.join(directory, frame <> ".disparity.bin"), binary)

        IO.puts(
          "OPENCV_FRAME mode=#{name} frame=#{frame} total_ms=#{ms} occurrences=#{length(copies)}"
        )
      end

      IO.puts("OPENCV_REPEAT_OUTPUTS_PASS mode=#{name} frames=#{frame_count} unique=50")

      times = Enum.map(results, &elem(&1, 2))

      File.write!(
        Path.join(directory, "timings_ms.txt"),
        Enum.map_join(times, "\n", &to_string/1) <> "\n"
      )

      sorted = Enum.sort(times)

      IO.puts(
        "OPENCV_RESULT mode=#{name} threads=#{threads} fps=#{frame_count * 1000 / Enum.sum(times)} wall_fps=#{frame_count * 1_000_000 / wall_us} wall_ms=#{wall_us / 1000} p50_ms=#{Enum.at(sorted, ceil(frame_count * 0.5) - 1)} p95_ms=#{Enum.at(sorted, ceil(frame_count * 0.95) - 1)} includes=mat_creation_native_sgbm_f32_conversion_binary_copy wall_includes=guards excludes=decode_disk_io warmup=3"
      )
    end

    IO.puts("EVISION_STEREO_PASS")
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

  defp gray(path) do
    binary = File.read!(path)
    unless byte_size(binary) == @width * @height * 4, do: raise("incorrect input shape")

    for <<value::little-signed-32 <- binary>>, into: <<>> do
      unless value in 0..255, do: raise("expected grayscale 0..255")
      <<value>>
    end
  end

  defp process({_frame, left, right}, matcher) do
    %Evision.Mat{} = left = Evision.Mat.from_binary(left, :u8, @height, @width, 1)
    %Evision.Mat{} = right = Evision.Mat.from_binary(right, :u8, @height, @width, 1)
    %Evision.Mat{} = disparity = Evision.StereoSGBM.compute(matcher, left, right)

    %Evision.Mat{} =
      result = Evision.multiply(disparity, {0.0625, 0.0625, 0.0625, 0.0625}, dtype: 5)

    binary = Evision.Mat.to_binary(result)
    unless is_binary(binary), do: raise("failed to copy disparity")
    binary
  end

  defp guard_resources! do
    for path <- Path.wildcard("/sys/class/thermal/thermal_zone*/temp") do
      if File.read!(path) |> String.trim() |> String.to_integer() >= 70_000,
        do: raise("thermal limit")
    end

    [_, available] = Regex.run(~r/MemAvailable:\s+(\d+)/, File.read!("/proc/meminfo"))
    if String.to_integer(available) < 2_000_000, do: raise("memory headroom guard")
  end
end

OpenCVStereoBench.run()
