# Strided channels-last convolutions (as Axon/Bumblebee emit them) through
# nx_iree on the CPU, against EXLA.
Logger.configure(level: :warning)
cpu = NxIREE.Device.find_default_device("local-sync")
opts = [compiler: NxIREE.Compiler, iree_compiler_flags: ["--iree-hal-target-backends=llvm-cpu", "--iree-input-type=stablehlo_xla"], iree_runtime_options: [device: cpu]]
key = Nx.Random.key(1)

for size <- [2, 3, 4, 5, 7, 8], {k, pad} <- [{3, 1}, {1, 0}], cin <- [4] do
  model =
    Axon.input("x", shape: {nil, size, size, cin})
    |> Axon.conv(8, kernel_size: k, strides: 2, padding: [{pad, pad}, {pad, pad}], use_bias: false)

  {init_fn, predict_fn} = Axon.build(model)
  {x, _} = Nx.Random.uniform(key, shape: {1, size, size, cin})
  params = init_fn.(Nx.to_template(x), Axon.ModelState.empty())
  want = Nx.Defn.jit(predict_fn, compiler: EXLA).(params, x) |> Nx.backend_transfer(Nx.BinaryBackend)

  status =
    try do
      got = Nx.Defn.jit(predict_fn, opts).(params, x) |> Nx.backend_transfer(Nx.BinaryBackend)
      if Nx.all_close(got, want) |> Nx.to_number() == 1, do: "ok #{inspect(Nx.shape(got))}", else: "WRONG"
    rescue
      _ -> "COMPILE FAILED (EXLA shape #{inspect(Nx.shape(want))})"
    end

  IO.puts("in #{size}x#{size}x#{cin} kernel #{k} pad #{pad} stride 2: #{status}")
end
