# Checks that a Bumblebee serving runs from NxIREE's module cache on a node
# without EXLA, as on a device: this node compiles into a fresh cache, then a
# separate VM whose code path has every dependency but EXLA runs the same
# serving code from that cache and must produce the same predictions.
#
#   mix run offline_cache.exs                  # Vulkan (default) or
#   NXIREE_DRIVER=local-task mix run offline_cache.exs
Logger.configure(level: :warning)

defmodule OfflineServing do
  @texts ["A wonderful little device.", "It broke after a day.", "Fine I guess", "Never again."]

  def texts, do: @texts

  def build(device, cache) do
    name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    {:ok, model} = Bumblebee.load_model({:hf, name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "distilbert/distilbert-base-uncased"})

    Bumblebee.Text.text_classification(model, tokenizer,
      compile: [batch_size: 4, sequence_length: 32],
      preallocate_params: true,
      defn_options: [
        compiler: NxIREE.Compiler,
        iree_runtime_options: [device: device],
        iree_module_cache: cache
      ]
    )
  end

  def run(device, cache) do
    {:ok, _} = Nx.Serving.start_link(serving: build(device, cache), name: __MODULE__, batch_timeout: 10)
    Nx.Serving.batched_run(__MODULE__, @texts)
    |> Enum.map(fn %{predictions: [top | _]} -> {top.label, Float.round(top.score, 4)} end)
  end

  def device do
    driver = System.get_env("NXIREE_DRIVER", "vulkan")
    match = System.get_env("NXIREE_DEVICE_MATCH", "")
    {:ok, devices} = NxIREE.list_devices(driver)
    Enum.find(devices, &String.contains?(&1.uri, match)) || raise "no #{driver} device"
  end
end

case System.argv() do
  ["device", cache, expected] ->
    false = Code.ensure_loaded?(EXLA)
    got = OfflineServing.run(OfflineServing.device(), cache)
    want = expected |> File.read!() |> :erlang.binary_to_term()
    IO.puts("device node without EXLA: #{inspect(got)}")
    IO.puts(if got == want, do: "MATCH", else: "MISMATCH, host had #{inspect(want)}")

  [] ->
    cache = Path.join(System.tmp_dir!(), "nx_iree_offline_#{System.unique_integer([:positive])}")
    expected = cache <> ".expected"
    want = OfflineServing.run(OfflineServing.device(), cache)
    File.write!(expected, :erlang.term_to_binary(want))
    IO.puts("host compiled #{length(File.ls!(cache))} modules: #{inspect(want)}")

    lib = Path.join(Mix.Project.build_path(), "lib")

    paths =
      for app <- File.ls!(lib), app not in ["exla", "xla"], ebin = Path.join([lib, app, "ebin"]), File.dir?(ebin),
          do: ["-pa", ebin]

    boot = """
    Application.put_env(:bumblebee, :offline, true)
    {:ok, _} = Application.ensure_all_started([:nx_iree, :bumblebee])
    Code.require_file(#{inspect(__ENV__.file)})
    """

    {out, status} =
      System.cmd("elixir", List.flatten(paths) ++ ["-e", boot, "--", "device", cache, expected],
        env: [{"BUMBLEBEE_OFFLINE", "true"}],
        stderr_to_stdout: true
      )

    IO.puts(out |> String.split("\n") |> Enum.reject(&String.contains?(&1, "[debug]")) |> Enum.join("\n"))
    IO.puts("device VM exited with #{status}")
    File.rm_rf!(cache)
    File.rm(expected)
end
