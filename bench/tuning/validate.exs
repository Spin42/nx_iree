# Checks matmul tuning specs for correctness: compiles dot products of many
# shapes with each spec for a Vulkan target and compares with EXLA on the CPU.
#
#   NXIREE_SPECS="a.mlir b.mlir" NXIREE_VULKAN_TARGET=pascal mix run tuning/validate.exs
#
# An empty spec list checks the preset configuration only.
Logger.configure(level: :warning)

{:ok, devices} = NxIREE.list_devices("vulkan")
match = System.get_env("NXIREE_DEVICE_MATCH", "GPU-00000000030000")
device = Enum.find(devices, &String.contains?(&1.uri, match)) || hd(devices)
target = System.get_env("NXIREE_VULKAN_TARGET", "pascal")
specs = [nil | String.split(System.get_env("NXIREE_SPECS", ""), " ", trim: true)]

cases = [
  {"square 64", fn k -> {Nx.Random.uniform(k, shape: {64, 64}), Nx.Random.uniform(k, shape: {64, 64})} end, &Nx.dot/2},
  {"square 512", fn k -> {Nx.Random.uniform(k, shape: {512, 512}), Nx.Random.uniform(k, shape: {512, 512})} end, &Nx.dot/2},
  {"square 1024", fn k -> {Nx.Random.uniform(k, shape: {1024, 1024}), Nx.Random.uniform(k, shape: {1024, 1024})} end, &Nx.dot/2},
  {"wide 128x512 . 512x96", fn k -> {Nx.Random.uniform(k, shape: {128, 512}), Nx.Random.uniform(k, shape: {512, 96})} end, &Nx.dot/2},
  {"odd 37x53 . 53x91", fn k -> {Nx.Random.uniform(k, shape: {37, 53}), Nx.Random.uniform(k, shape: {53, 91})} end, &Nx.dot/2},
  {"odd 100x200 . 200x50", fn k -> {Nx.Random.uniform(k, shape: {100, 200}), Nx.Random.uniform(k, shape: {200, 50})} end, &Nx.dot/2},
  {"vector 1x256 . 256x256", fn k -> {Nx.Random.uniform(k, shape: {1, 256}), Nx.Random.uniform(k, shape: {256, 256})} end, &Nx.dot/2},
  {"dense layer 32x784 . 784x128", fn k -> {Nx.Random.uniform(k, shape: {32, 784}), Nx.Random.uniform(k, shape: {784, 128})} end, &Nx.dot/2},
  {"transposed a^T . b", fn k -> {Nx.Random.uniform(k, shape: {96, 64}), Nx.Random.uniform(k, shape: {96, 80})} end,
   fn a, b -> Nx.dot(a, [0], b, [0]) end},
  {"batched 4x64x32 . 4x32x48", fn k -> {Nx.Random.uniform(k, shape: {4, 64, 32}), Nx.Random.uniform(k, shape: {4, 32, 48})} end,
   fn a, b -> Nx.dot(a, [2], [0], b, [1], [0]) end},
  {"s32 128x128", fn k ->
     {a, k2} = Nx.Random.randint(k, -8, 8, shape: {128, 128}, type: :s32)
     {b, _} = Nx.Random.randint(k2, -8, 8, shape: {128, 128}, type: :s32)
     {a, b}
   end, &Nx.dot/2}
]

key = Nx.Random.key(3)
exla = fn fun -> Nx.Defn.jit(fun, compiler: EXLA) end

results =
  for spec <- specs, {name, gen, fun} <- cases do
    {a, b} =
      case gen.(key) do
        {{a, _}, {b, _}} -> {a, b}
        {a, b} -> {a, b}
      end

    flags =
      ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla",
       "--iree-vulkan-target=#{target}"] ++
        if(spec, do: ["--iree-codegen-transform-dialect-library=#{spec}"], else: [])

    opts = [compiler: NxIREE.Compiler, iree_compiler_flags: flags, iree_runtime_options: [device: device]]
    want = exla.(fun).(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)

    status =
      try do
        got = Nx.Defn.jit(fun, opts).(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
        err = Nx.subtract(got, want) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        scale = want |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        rel = err / max(scale, 1.0e-30)
        if Nx.shape(got) == Nx.shape(want) and rel < 1.0e-5, do: :ok, else: {:wrong, rel}
      rescue
        e -> {:error, Exception.message(e) |> String.slice(0, 100)}
      end

    label = if spec, do: Path.basename(spec, ".mlir"), else: "preset #{target}"
    IO.puts("#{String.pad_trailing(label, 22)} #{String.pad_trailing(name, 32)} #{inspect(status)}")
    status
  end

bad = Enum.count(results, &(&1 != :ok))
IO.puts("#{length(results) - bad}/#{length(results)} ok")
