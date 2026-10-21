flags = [
  "--iree-input-type=stablehlo_xla",
  "--iree-execution-model=async-internal"
]

driver = System.get_env("NX_IREE_DEFAULT_DRIVER") || "local-sync"

# NX_IREE_DEFAULT_DEVICE picks a device of that driver by URI substring
device =
  case System.get_env("NX_IREE_DEFAULT_DEVICE") do
    nil ->
      NxIREE.Device.find_default_device(driver)

    match ->
      {:ok, devices} = NxIREE.Device.list(driver)

      Enum.find(devices, &String.contains?(&1.uri, match)) ||
        raise "no #{driver} device matches #{match}"
  end

runtime_options = [device: device]

Nx.Defn.global_default_options(
  compiler: NxIREE.Compiler,
  iree_compiler_flags: flags,
  iree_runtime_options: runtime_options
)

Nx.global_default_backend(NxIREE.Backend)

if System.get_env("DEBUG") do
  IO.gets("Press Enter to continue - PID: #{System.pid()}")
end

ExUnit.start()
