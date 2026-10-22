Code.require_file("path.exs", __DIR__)
Code.require_file("reference.exs", __DIR__)
Logger.configure(level: :warning)
cpu = NxIREE.Device.find_default_device("local-sync")

for {name, shape, values, p1, p2} <- StereoPathReference.fixtures() do
  input = Nx.tensor(values, type: :s32) |> Nx.reshape(shape)
  want = StereoPathReference.aggregate(values, shape, p1, p2)
  args = [input, Nx.tensor(p1, type: :s32), Nx.tensor(p2, type: :s32)]

  for {label, options} <- [
        {"Nx evaluator", [compiler: Nx.Defn.Evaluator]},
        {"IREE CPU", [compiler: NxIREE.Compiler, iree_runtime_options: [device: cpu]]}
      ] do
    got = Nx.Defn.jit_apply(&StereoPath.aggregate/3, args, options) |> Nx.to_flat_list()
    if got != want, do: raise("#{label}: #{name} differs from scalar reference")
    IO.puts("PASS #{label}: #{name} (#{length(want)} exact values)")
  end
end
