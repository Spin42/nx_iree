if root = System.get_env("EVISION_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx castore evision),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:evision)

defmodule OpenCVUMatBench do
  alias Evision.Constant, as: C

  def run do
    true = Evision.OCL.haveOpenCL()
    true = Evision.OCL.useOpenCL()
    device = Evision.OCL.Device.getDefault() |> Evision.OCL.Device.name()
    threads = System.get_env("OPENCV_THREADS", "4") |> String.to_integer()
    iterations = System.get_env("UMAT_ITERATIONS", "20") |> String.to_integer()
    unless iterations > 0, do: raise("UMAT_ITERATIONS must be positive")
    Evision.setNumThreads(threads)

    IO.puts("EVISION_UMAT_DEVICE=#{device} cpu_threads=#{Evision.getNumThreads()}")
    :rand.seed(:exsss, {42, 17, 93})
    input = :rand.bytes(640 * 272 * 3)
    mat = Evision.Mat.from_binary(input, :u8, 272, 640, 3)
    umat = Evision.UMat.uMat(mat)
    map_x = for _y <- 0..135, x <- 0..319, into: <<>>, do: <<x + 0.25::float-native-32>>
    map_y = for y <- 0..135, _x <- 0..319, into: <<>>, do: <<y + 0.5::float-native-32>>

    cpu_maps =
      {Evision.Mat.from_binary(map_x, :f32, 136, 320, 1),
       Evision.Mat.from_binary(map_y, :f32, 136, 320, 1)}

    gpu_maps = cpu_maps |> Tuple.to_list() |> Enum.map(&Evision.UMat.uMat/1) |> List.to_tuple()

    cpu_stages = stages(mat, cpu_maps)
    gpu_stages = stages(umat, gpu_maps)

    for {{name, cpu}, {name, gpu}} <- Enum.zip(cpu_stages, gpu_stages) do
      {max_abs, changed} = difference(pixels(cpu), pixels(gpu), 0, 0)
      IO.puts("EVISION_UMAT_CHECK stage=#{name} max_abs=#{max_abs} changed_pixels=#{changed}")
      unless max_abs <= 1, do: raise("#{name} differs by more than one intensity level")
    end

    unless pixels(mat) == input and pixels(umat) == input,
      do: raise("pipeline modified its input")

    for {mode, maps} <- [{:cpu, cpu_maps}, {:gpu, gpu_maps}] do
      for _ <- 1..3, do: process(input, mode, maps)

      times =
        for _ <- 1..iterations do
          {us, output} = :timer.tc(fn -> process(input, mode, maps) end)
          unless byte_size(output) == 320 * 136, do: raise("incorrect output size")
          us / 1000
        end

      sorted = Enum.sort(times)

      IO.puts(
        "EVISION_UMAT_RESULT mode=#{mode} frames=#{iterations} " <>
          "mean_ms=#{Enum.sum(times) / iterations} " <>
          "p50_ms=#{Enum.at(sorted, div(iterations - 1, 2))} " <>
          "p95_ms=#{Enum.at(sorted, ceil(iterations * 0.95) - 1)} " <>
          "includes=mat_creation_upload_standard_calls_final_readback_binary_copy " <>
          "excludes=map_creation_decode_startup_compile warmup=3"
      )
    end

    IO.puts("EVISION_UMAT_NUMERICAL_PASS dispatch_evidence=required_separately")
  end

  defp stages(src, {map_x, map_y}) do
    gray = Evision.cvtColor(src: src, code: C.cv_COLOR_BGR2GRAY())
    resized = Evision.resize(src: gray, dsize: {320, 136}, interpolation: C.cv_INTER_LINEAR())
    filtered = Evision.medianBlur(src: resized, ksize: 3)

    rectified =
      Evision.remap(
        src: filtered,
        map1: map_x,
        map2: map_y,
        interpolation: C.cv_INTER_LINEAR(),
        borderMode: C.cv_BORDER_CONSTANT()
      )

    [{:gray, gray}, {:resize, resized}, {:median, filtered}, {:remap, rectified}]
  end

  defp process(input, mode, maps) do
    mat = Evision.Mat.from_binary(input, :u8, 272, 640, 3)
    src = if mode == :gpu, do: Evision.UMat.uMat(mat), else: mat
    stages(src, maps) |> List.last() |> elem(1) |> pixels()
  end

  defp pixels(%Evision.UMat{} = umat), do: umat |> Evision.UMat.get() |> pixels()
  defp pixels(%Evision.Mat{} = mat), do: Evision.Mat.to_binary(mat)

  defp difference(<<>>, <<>>, maximum, changed), do: {maximum, changed}

  defp difference(<<a, rest_a::binary>>, <<b, rest_b::binary>>, maximum, changed),
    do: difference(rest_a, rest_b, max(maximum, abs(a - b)), changed + if(a == b, do: 0, else: 1))
end

OpenCVUMatBench.run()
