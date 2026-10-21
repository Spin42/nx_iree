# Runs functions exported with NxIREE.export/3 (bench/aot.exs) on the Jetson
# GPU through NVK, with only Nx and nx_iree's runtime on the board.
#
# The nx_iree application is not started (its .app lists exla); devices and
# the VM instance are created on first use anyway. Helpers live in a module so
# they run compiled rather than through the evaluator.
root = Path.dirname(__ENV__.file)
for app <- ~w(complex telemetry nx nx_iree), do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
{:ok, _} = Application.ensure_all_started(:nx)

defmodule Bench do
  def log(msg), do: File.write!("/root/nxiree.log", "#{msg}\n", [:append])

  def uniform(shape) do
    n = shape |> Tuple.to_list() |> Enum.product()
    bin = for _ <- 1..n, into: <<>>, do: <<:rand.uniform()::float-32-native>>
    bin |> Nx.from_binary(:f32) |> Nx.reshape(shape)
  end

  def time(f, reps) do
    f.()
    t0 = System.monotonic_time(:microsecond)
    for _ <- 1..reps, do: f.()
    (System.monotonic_time(:microsecond) - t0) / reps
  end

  def run(root) do
    for file <- root |> Path.join("jetson-modules*.bin") |> Path.wildcard() |> Enum.sort() do
      run_file(file)
    end
  end

  def run_file(file) do
    %{models: exported, target: target} = file |> File.read!() |> :erlang.binary_to_term()
    {:ok, [device | _]} = NxIREE.list_devices("vulkan")
    log("device #{device.uri}, #{Path.basename(file)} compiled for #{target}")

    models = Map.new(exported, fn {k, bin} -> {k, NxIREE.load_exported(bin, device: device)} end)
    to_dev = fn t -> Nx.backend_transfer(t, {NxIREE.Backend, device: device}) end

    x = to_dev.(Nx.iota({256}, type: :f32))
    lat = time(fn -> NxIREE.run(models[{:add, 256}], [x]) end, 50)
    lat_rb = time(fn -> NxIREE.run(models[{:add, 256}], [x]) |> Nx.to_binary() end, 50)
    log("tiny add: #{Float.round(lat, 1)} us per call, #{Float.round(lat_rb, 1)} us with readback")

    a = uniform({256, 256})
    b = uniform({256, 256})
    gpu = NxIREE.run(models[{:matmul, 256}], [a, b]) |> Nx.backend_transfer(Nx.BinaryBackend)
    diff = Nx.subtract(gpu, Nx.dot(a, b)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    log("matmul 256 max |gpu - cpu| = #{diff}")

    # NXIREE_SIZES="256:512:1024" keeps runs short; the board's supply trips
    # on long full-load bursts
    sizes =
      System.get_env("NXIREE_SIZES", "256:512:1024")
      |> String.split(":", trim: true)
      |> Enum.map(&String.to_integer/1)

    for n <- sizes do
      da = to_dev.(uniform({n, n}))
      db = to_dev.(uniform({n, n}))
      reps = if n >= 1024, do: 3, else: 10
      us = time(fn -> NxIREE.run(models[{:matmul, n}], [da, db]) end, reps)
      log("matmul #{n}: #{Float.round(2 * n * n * n / us / 1000, 1)} GFLOPS (#{Float.round(us / 1000, 2)} ms)")
    end

    if System.get_env("NXIREE_CHAIN") do
      dx = to_dev.(uniform({4_000_000}))
      us = time(fn -> NxIREE.run(models[{:chain, 4_000_000}], [dx]) |> Nx.to_number() end, 5)
      log("chain on 4M floats: #{Float.round(us / 1000, 2)} ms")
    end
  end
end

Bench.run(root)
