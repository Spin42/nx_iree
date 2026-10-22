Code.require_file("path.exs", __DIR__)
Code.require_file("reference.exs", __DIR__)
Logger.configure(level: :warning)

flags = [
  "--iree-hal-target-backends=vulkan-spirv",
  "--iree-vulkan-target-features=spirv:v1.3,cap:Shader,cap:GroupNonUniform,cap:GroupNonUniformArithmetic,cap:GroupNonUniformShuffle,cap:GroupNonUniformBallot"
]

encode = fn values -> Nx.tensor(values, type: :s32) |> Nx.to_binary() end

prepare = fn {name, shape, values, p1, p2} ->
  {name, shape, encode.(values), encode.(StereoPathReference.aggregate(values, shape, p1, p2)),
   p1, p2}
end

fixtures = Enum.map(StereoPathReference.fixtures(), prepare)

benchmarks =
  for shape <- [{32, 24, 32}, {160, 120, 64}] do
    prepare.({inspect(shape), shape, StereoPathReference.values(shape, 13), 8, 32})
  end

models =
  (fixtures ++ benchmarks)
  |> Enum.map(&elem(&1, 1))
  |> Enum.uniq()
  |> Map.new(fn shape ->
    binary =
      NxIREE.export(
        &StereoPath.aggregate/3,
        [Nx.template(shape, :s32), Nx.template({}, :s32), Nx.template({}, :s32)],
        iree_compiler_flags: flags
      )

    IO.puts("export #{inspect(shape)}: #{byte_size(binary)} bytes")
    {shape, binary}
  end)

out = System.fetch_env!("STEREO_OUT")

File.write!(
  out,
  :erlang.term_to_binary(%{models: models, fixtures: fixtures, benchmarks: benchmarks})
)

IO.puts("wrote #{out}")
