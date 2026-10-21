# Runs a model exported by export_distilbert.exs on the Jetson GPU with only
# Nx and nx_iree, and compares its logits with EXLA's on the host.
#
#   MODEL_DIR=/root/model elixir run_model.exs   (code paths as in run.exs)
root = Path.dirname(__ENV__.file)
for app <- ~w(complex telemetry nx nx_iree), do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
{:ok, _} = Application.ensure_all_started(:nx)

defmodule ModelBench do
  def log(msg), do: File.write!("/root/nxiree.log", "#{msg}\n", [:append])

  def run(dir) do
    {:ok, [device | _]} = NxIREE.list_devices("vulkan")
    read = fn f -> dir |> Path.join(f) |> File.read!() |> Nx.deserialize() end
    params = read.("params.nx")
    inputs = read.("inputs.nx")
    reference = read.("reference.nx")
    labels = dir |> Path.join("labels.etf") |> File.read!() |> :erlang.binary_to_term()

    to_dev = fn c -> Nx.Defn.Composite.traverse(c, &Nx.backend_transfer(&1, {NxIREE.Backend, device: device})) end
    {us, params} = :timer.tc(fn -> to_dev.(params) end)
    log("params on device in #{div(us, 1000)} ms")
    inputs = to_dev.(inputs)

    for file <- dir |> Path.join("distilbert-*.bin") |> Path.wildcard() |> Enum.sort() do
      model = NxIREE.load_exported(File.read!(file), device: device)
      run = fn -> NxIREE.run(model, [params, inputs]) |> Nx.backend_copy(Nx.BinaryBackend) end

      {first_us, logits} = :timer.tc(run)
      reps = 10
      {us, _} = :timer.tc(fn -> for _ <- 1..reps, do: run.() end)

      diff = Nx.subtract(logits, reference) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      got = logits |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> Enum.map(&labels[&1])
      want = reference |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> Enum.map(&labels[&1])

      log("#{Path.basename(file)}: labels #{inspect(got)} (#{if got == want, do: "same as EXLA", else: "EXLA #{inspect(want)}"}), " <>
            "max |logit - EXLA| #{Float.round(diff, 6)}, first call #{div(first_us, 1000)} ms, " <>
            "#{Float.round(us / reps / 1000, 1)} ms per batch of 4")
    end
  end
end

ModelBench.run(System.get_env("MODEL_DIR", "/root/model"))
