Code.require_file("reference.exs", __DIR__)
Logger.configure(level: :warning)

variant = System.get_env("STEREO_KERNEL", "subgroup")

unless variant in ["shared", "subgroup", "subgroup_shared"],
  do: raise("STEREO_KERNEL must be shared, subgroup, or subgroup_shared")

glslang =
  System.get_env("GLSLANG") || System.find_executable("glslang") ||
    System.find_executable("glslangValidator") ||
    raise("install glslang or set GLSLANG to its executable path")

validator =
  System.get_env("SPIRV_VAL") || System.find_executable("spirv-val") ||
    raise("install SPIRV-Tools or set SPIRV_VAL to its spirv-val executable")

output_path = System.fetch_env!("STEREO_OUT")

shader =
  Path.join(__DIR__, if(variant == "shared", do: "path_shared.comp", else: "path_subgroup.comp"))

features =
  if variant == "shared",
    do: "spirv:v1.3,cap:Shader",
    else:
      "spirv:v1.3,cap:Shader,cap:GroupNonUniform,cap:GroupNonUniformArithmetic,cap:GroupNonUniformShuffle"

encode = fn values ->
  Nx.tensor(values, type: :s32, backend: Nx.BinaryBackend) |> Nx.to_binary()
end

prepare = fn {name, shape, values, p1, p2} ->
  {name, shape, encode.(values), encode.(StereoPathReference.aggregate(values, shape, p1, p2)),
   p1, p2}
end

fixtures = Enum.map(StereoPathReference.fixtures(), prepare)

benchmarks =
  for shape <- [{32, 24, 32}, {160, 120, 64}] do
    prepare.({inspect(shape), shape, StereoPathReference.values(shape, 13), 8, 32})
  end

directory =
  Path.join(
    System.tmp_dir!(),
    "nxiree-stereo-" <> Base.url_encode64(:crypto.strong_rand_bytes(18), padding: false)
  )

File.mkdir!(directory)

try do
  models =
    (fixtures ++ benchmarks)
    |> Enum.map(&elem(&1, 1))
    |> Enum.uniq()
    |> Map.new(fn {steps, lines, disparities} = shape ->
      unless disparities in 1..128, do: raise("benchmark kernels support 1–128 disparities")
      width = Stream.iterate(1, &(&1 * 2)) |> Enum.find(&(&1 >= disparities))
      spirv = Path.join(directory, "path.spv")

      args = [
        "-V",
        "--target-env",
        "vulkan1.1",
        "-S",
        "comp",
        "-DWIDTH=#{width}",
        "-DSTEPS=#{steps}",
        "-DLINES=#{lines}",
        "-DDISPARITIES=#{disparities}",
        "-DFORCE_SHARED=#{if variant == "subgroup_shared", do: 1, else: 0}",
        shader,
        "-o",
        spirv
      ]

      for {tool, args} <- [{glslang, args}, {validator, ["--target-env", "vulkan1.1", spirv]}] do
        case System.cmd(tool, args, stderr_to_stdout: true) do
          {_, 0} -> :ok
          {log, status} -> raise("#{tool} exited #{status}:\n#{log}")
        end
      end

      template = Nx.template(shape, :s32)

      binary =
        NxIREE.Vulkan.export(
          File.read!(spirv),
          [template, Nx.template({}, :s32), Nx.template({}, :s32)],
          template,
          workgroups: {lines, 1, 1},
          target_features: features
        )

      IO.puts("export #{variant} #{inspect(shape)}: #{byte_size(binary)} bytes")
      {shape, binary}
    end)

  File.write!(
    output_path,
    :erlang.term_to_binary(%{models: models, fixtures: fixtures, benchmarks: benchmarks})
  )

  IO.puts("wrote #{output_path}")
after
  File.rm_rf!(directory)
end
