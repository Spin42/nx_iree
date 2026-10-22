Logger.configure(level: :warning)

width = System.get_env("VIDEO_WIDTH", "320") |> String.to_integer()
height = System.get_env("VIDEO_HEIGHT", "136") |> String.to_integer()
disparities = System.get_env("VIDEO_DISPARITIES", "64") |> String.to_integer()
unless width > 0 and height > 0 and disparities in 3..128, do: raise("invalid video shape")
output = System.fetch_env!("STEREO_OUT")
glslang = System.fetch_env!("GLSLANG")
validator = System.fetch_env!("SPIRV_VAL")
selection = System.get_env("VIDEO_SELECT", "coalesced")
storage = System.get_env("VIDEO_STORAGE", "s32")
unless storage in ["s32", "packed16"], do: raise("VIDEO_STORAGE must be s32 or packed16")
packed = storage == "packed16"
fuse_select = System.get_env("VIDEO_FUSE_SELECT", "0") == "1"
if fuse_select and not packed, do: raise("VIDEO_FUSE_SELECT requires packed16")
lines_per_group = System.get_env("VIDEO_LINES_PER_GROUP", "1") |> String.to_integer()
unless lines_per_group in [1, 2, 4], do: raise("VIDEO_LINES_PER_GROUP must be 1, 2 or 4")
pixels_per_group = System.get_env("VIDEO_PIXELS_PER_GROUP", "1") |> String.to_integer()
unless pixels_per_group in [1, 2, 4], do: raise("VIDEO_PIXELS_PER_GROUP must be 1, 2 or 4")

unless selection in ["serial", "coalesced", "coalesced_shared"],
  do: raise("VIDEO_SELECT must be serial, coalesced or coalesced_shared")

if packed and selection == "serial", do: raise("packed16 requires a coalesced selector")

directions = [{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}]

volume =
  if packed,
    do: Nx.template({height, width, div(disparities + 1, 2)}, :u32),
    else: Nx.template({height, width, disparities}, :s32)

image = Nx.template({height, width}, :s32)
disparity = Nx.template({height, width}, :f32)
pixel_groups = {div(width * height + 127, 128), 1, 1}
volume_groups = {div(Nx.size(volume) + 127, 128), 1, 1}
base = ["-DWIDTH=#{width}", "-DHEIGHT=#{height}", "-DDISPARITIES=#{disparities}"]
features = "spirv:v1.3,cap:Shader"

subgroup =
  features <> ",cap:GroupNonUniform,cap:GroupNonUniformArithmetic,cap:GroupNonUniformShuffle"

directory =
  Path.join(System.tmp_dir!(), "stereo-video-" <> Base.encode16(:crypto.strong_rand_bytes(12)))

File.mkdir!(directory)

compile = fn {name, source, defines, inputs, result, groups, capabilities} ->
  spirv = Path.join(directory, name <> ".spv")

  args =
    ["-V", "--target-env", "vulkan1.1", "-S", "comp"] ++
      base ++ defines ++ [Path.join(__DIR__, source), "-o", spirv]

  for {tool, args} <- [{glslang, args}, {validator, ["--target-env", "vulkan1.1", spirv]}] do
    case System.cmd(tool, args, stderr_to_stdout: true) do
      {_, 0} -> :ok
      {log, status} -> raise("#{tool} exited #{status}: #{log}")
    end
  end

  binary =
    NxIREE.Vulkan.export(File.read!(spirv), inputs, result,
      workgroups: groups,
      target_features: capabilities
    )

  IO.puts("EXPORTED #{name} #{byte_size(binary)} bytes")
  {name, binary}
end

try do
  passes =
    directions
    |> Enum.with_index()
    |> Enum.map(fn {{dx, dy}, index} ->
      lines =
        cond do
          dy == 0 -> height
          dx == 0 -> width
          true -> width + height - 1
        end

      {"path#{index}", if(packed, do: "aggregate_packed.comp", else: "aggregate.comp"),
       [
         "-DDX=#{dx}",
         "-DDY=#{dy}",
         "-DP1=72",
         "-DP2=288",
         "-DLINES_PER_GROUP=#{lines_per_group}",
         "-DPASS_COUNT=#{index + 1}",
         "-DFORCE_SHARED=#{if selection == "coalesced_shared", do: 1, else: 0}",
         "-DFIRST=#{if index == 0, do: 1, else: 0}",
         "-DFUSE_SELECT=#{if fuse_select and index == 7, do: 1, else: 0}"
       ], [volume, volume], if(fuse_select and index == 7, do: disparity, else: volume),
       {div(lines + lines_per_group - 1, lines_per_group), 1, 1}, subgroup}
    end)

  passes =
    if fuse_select do
      {_, source, defines, inputs, _, groups, capabilities} = List.last(passes)

      diagnostic =
        {"path7_sum", source, List.replace_at(defines, -1, "-DFUSE_SELECT=0"), inputs, volume,
         groups, capabilities}

      passes ++ [diagnostic]
    else
      passes
    end

  selector =
    if selection == "serial" do
      {"select", "select.comp", [], [volume], disparity, pixel_groups, features}
    else
      {"select", "select_coalesced.comp",
       [
         "-DFORCE_SHARED=#{if selection == "coalesced_shared", do: 1, else: 0}",
         "-DPACKED=#{if packed, do: 1, else: 0}",
         "-DPIXELS_PER_GROUP=#{pixels_per_group}"
       ], [volume], disparity,
       {div(width * height + pixels_per_group - 1, pixels_per_group), 1, 1}, subgroup}
    end

  cost_source = if packed, do: "cost_packed.comp", else: "cost.comp"

  kernels =
    [
      {"cost_left", cost_source, ["-DSIDE=-1"], [image, image], volume, volume_groups, features},
      {"cost_right", cost_source, ["-DSIDE=1"], [image, image], volume, volume_groups, features},
      selector,
      {"consistent", "consistent.comp", [], [disparity, disparity], disparity, pixel_groups,
       features}
    ] ++ passes

  models = Map.new(kernels, compile)

  File.write!(
    output,
    :erlang.term_to_binary(%{
      models: models,
      shape: {height, width},
      disparities: disparities,
      storage: storage
    })
  )

  IO.puts("WROTE #{output}")
after
  File.rm_rf!(directory)
end
