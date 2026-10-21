# Checks that sorting an nx_iree tensor leaves the input untouched.
# NXIREE_DEVICE_MATCH picks the device by URI substring (default: local-sync).
match = System.get_env("NXIREE_DEVICE_MATCH", "local-sync")
{:ok, devices} = NxIREE.list_devices()
device = Enum.find(devices, &String.contains?(&1.uri, match)) || raise "no device matches #{match}"
IO.puts("device #{device.uri}")

opts = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]

fresh = fn ->
  Nx.tensor([[[4, 5, 2], [2, 5, 3], [5, 0, 2]], [[1, 9, 8], [2, 1, 3], [2, 1, 4]]])
  |> Nx.backend_transfer({NxIREE.Backend, device: device})
end

before = Nx.to_flat_list(fresh.())

for {name, fun} <- [
      sort_axis0: &Nx.sort(&1, axis: 0),
      sort_axis1: &Nx.sort(&1, axis: 1),
      argsort: &Nx.argsort(&1, axis: 0),
      plus_sort: &Nx.sort(Nx.add(&1, 0), axis: 0)
    ] do
  t = fresh.()
  _ = Nx.Defn.jit_apply(fun, [t], opts) |> Nx.to_flat_list()
  intact = Nx.to_flat_list(t) == before
  IO.puts("#{name}: input #{if intact, do: "intact", else: "MODIFIED #{inspect(Nx.to_flat_list(t))}"}")
end
