# Bumblebee-style self-attention from a hidden state (dense q/k projections,
# head split, scaled scores, mask bias, softmax) on Vulkan vs EXLA. Steps are
# toggled to find the smallest graph that goes wrong.
Logger.configure(level: :warning)
{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")))
opts = [compiler: NxIREE.Compiler, iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla"], iree_runtime_options: [device: gpu]]
key = Nx.Random.key(5)
{h, key} = Nx.Random.normal(key, shape: {1, 8, 32})
{wq, key} = Nx.Random.normal(key, 0.0, 0.2, shape: {32, 32})
{wk, _} = Nx.Random.normal(key, 0.0, 0.2, shape: {32, 32})
mask = Nx.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])

heads = fn x -> x |> Nx.reshape({1, 8, 4, 8}) |> Nx.transpose(axes: [0, 2, 1, 3]) end
scores = fn h, wq, wk ->
  q = h |> Nx.dot(wq) |> heads.()
  k = h |> Nx.dot(wk) |> heads.()
  Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(1 / :math.sqrt(8))
end
bias = fn m -> Nx.select(m |> Nx.new_axis(1) |> Nx.new_axis(1), 0.0, Nx.Constants.min_finite(:f32)) end

cases = [
  {"q projection only", fn h, wq, _wk, _m -> Nx.dot(h, wq) end},
  {"scores", fn h, wq, wk, _m -> scores.(h, wq, wk) end},
  {"scores + bias", fn h, wq, wk, m -> Nx.add(scores.(h, wq, wk), bias.(m)) end},
  {"softmax(scores)", fn h, wq, wk, _m -> Axon.Activations.softmax(scores.(h, wq, wk), axis: -1) end},
  {"softmax(scores + bias)", fn h, wq, wk, m -> Axon.Activations.softmax(Nx.add(scores.(h, wq, wk), bias.(m)), axis: -1) end}
]

for {name, fun} <- cases do
  want = Nx.Defn.jit(fun, compiler: EXLA).(h, wq, wk, mask) |> Nx.backend_transfer(Nx.BinaryBackend)
  got = Nx.Defn.jit(fun, opts).(h, wq, wk, mask) |> Nx.backend_transfer(Nx.BinaryBackend)
  finite = Nx.select(Nx.abs(want) |> Nx.less(1.0e30), Nx.subtract(got, want) |> Nx.abs(), 0.0)
  err = finite |> Nx.reduce_max() |> Nx.to_number()
  scale = Nx.select(Nx.abs(want) |> Nx.less(1.0e30), Nx.abs(want), 0.0) |> Nx.reduce_max() |> Nx.to_number()
  IO.puts("#{String.pad_trailing(name, 24)} max abs error #{:erlang.float_to_binary(err, decimals: 7)} (scale #{Float.round(scale, 4)})")
end
