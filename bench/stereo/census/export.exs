Logger.configure(level: :warning)

width = System.get_env("VIDEO_WIDTH", "320") |> String.to_integer()
height = System.get_env("VIDEO_HEIGHT", "136") |> String.to_integer()
disparities = System.get_env("VIDEO_DISPARITIES", "64") |> String.to_integer()
paths = System.get_env("VIDEO_PATHS", "8") |> String.to_integer()
p1 = System.get_env("VIDEO_P1", "10") |> String.to_integer()
p2 = System.get_env("VIDEO_P2", "120") |> String.to_integer()
uniqueness = System.get_env("VIDEO_UNIQUENESS", "1.1") |> String.to_float()
block = System.get_env("VIDEO_DP_BLOCK", "8") |> String.to_integer()
selector = System.get_env("VIDEO_SELECT", "rolling")
select_lanes = System.get_env("VIDEO_SELECT_LANES", "32") |> String.to_integer()
schedule = System.get_env("VIDEO_PATH_SCHEDULE", "cached")
census = System.get_env("VIDEO_CENSUS", "symmetric")
execution = System.get_env("VIDEO_EXECUTION", "stages")
projection = System.get_env("VIDEO_PROJECTION", "gather")
fusion = System.get_env("VIDEO_FUSION", "none")
mappable = System.get_env("VIDEO_MAPPABLE", "false")
vertical? = fusion in ["vertical", "cost"]
cost? = fusion == "cost"
cost_bound = if census == "classic", do: 48, else: 31
projected? = selector == "projected"
fused? = selector in ["fused", "projected"]

skip? = fn name ->
  (name == "sum" and fused?) or (name == "consistency_projected" and not projected?) or
    (name == "first" and not cost?)
end

unless width > 8 and height > 6 and disparities in [64, 128] and paths in [3, 4, 8] and
         p1 >= 0 and p1 < p2 and p2 + cost_bound <= 255 and uniqueness >= 1.0 and
         block in [4, 8, 16] and
         select_lanes in [4, 8, 16, 32] and
         selector in ["rolling", "gather", "fused", "projected"] and
         schedule in ["generic", "specialized", "cached", "tiled"] and
         census in ["symmetric", "classic"],
       do: raise("invalid census matcher parameters")

unless execution in ["stages", "graph"], do: raise("invalid execution mode")
unless projection in ["gather", "shared"], do: raise("invalid projection mode")
unless fusion in ["none", "vertical", "cost"], do: raise("invalid fusion mode")
unless mappable in ["false", "true"], do: raise("invalid external mapping policy")

if vertical? and
     not (paths == 3 and projected? and schedule == "tiled" and
            select_lanes == div(disparities, block)),
   do: raise("vertical fusion requires 3 tiled projected paths and matching lane groups")

# Cost fusion sums both horizontal path bytes in packed form, so each must stay below 128.
if cost? and 2 * (p2 + cost_bound) > 255,
  do: raise("cost fusion requires P2 + cost bound <= 127")

if projection == "shared" and width > 4096, do: raise("row scatter requires width <= 4096")

image = Nx.template({height, width}, :s32)

# Cost fusion stores features and byte volumes in tiles of `tile` rows per column.
tile = div(128 * block, disparities)
tiled_rows = div(height + tile - 1, tile) * tile

feature =
  if cost?,
    do: Nx.template({tiled_rows, width, if(census == "classic", do: 4, else: 2)}, :u32),
    else: Nx.template(if(census == "classic", do: {height, width, 2}, else: {height, width}), :u32)

cost = Nx.template({tiled_rows, width, div(disparities, 4)}, :u32)

directional =
  if cost?,
    do: Nx.template({tiled_rows, width, div(disparities, 4)}, :u32),
    else: Nx.template({if(vertical?, do: 2, else: paths), height, width, div(disparities, 4)}, :u32)

sum = Nx.template({height, width, div(disparities, 2)}, :u32)
disparity = Nx.template({height, width}, :f32)
winners = Nx.template({height, width}, :u32)
pixels = {div(width * height + 127, 128), 1, 1}
select_pixels = div(128, if(projected?, do: select_lanes, else: 32))
features = "spirv:v1.3,cap:Shader,cap:GroupNonUniform,cap:GroupNonUniformShuffle"

lines = div(128 * block, disparities)

kernels = [
  if(cost?,
    do:
      {"census", [image, image], feature,
       {div(width + div(128, tile) - 1, div(128, tile)), div(height + 4 * tile - 1, 4 * tile), 1}},
    else:
      {"census", [image, image], {feature, feature},
       {div(width + 119, 120), div(height + 15, 16), 1}}
  ),
  {"first", [feature], {directional, cost}, {div(height + lines - 1, lines), 1, 1}},
  if(cost?,
    do: {"paths", [cost], directional, {div(height + lines - 1, lines), 1, 1}},
    else:
      {"paths", [feature, feature], directional,
       {div(width + height - 2 + lines, lines), if(vertical?, do: 2, else: paths), 1}}
  ),
  {"sum", [directional], sum, {div(Nx.size(sum) + 127, 128), 1, 1}},
  {"select",
   cond do
     cost? -> [cost, directional, directional]
     vertical? -> [feature, feature, directional]
     true -> [if(fused?, do: directional, else: sum)]
   end, {disparity, if(projected?, do: winners, else: disparity)},
   {if(vertical?,
      do: div(width + lines - 1, lines),
      else:
        if(selector in ["rolling", "fused"],
          do: div(height + 3, 4),
          else: div(width * height + select_pixels - 1, select_pixels)
        )
    ), 1, 1}},
  {"consistency_projected", [disparity, winners], disparity,
   if(projection == "shared", do: {height, 1, 1}, else: pixels)},
  {"filter", if(projected?, do: [disparity], else: [disparity, disparity]), disparity, pixels}
]

pipeline =
  [
    if(cost?,
      do: {"census", ["left", "right"], ["features"]},
      else: {"census", ["left", "right"], ["left_features", "right_features"]}
    ),
    {"first", ["features"], ["paths_forward", "cost"]},
    if(cost?,
      do: {"paths", ["cost"], ["paths_reverse"]},
      else: {"paths", ["left_features", "right_features"], ["paths"]}
    ),
    {"sum", ["paths"], ["sum"]},
    {"select",
     cond do
       cost? -> ["cost", "paths_forward", "paths_reverse"]
       vertical? -> ["left_features", "right_features", "paths"]
       true -> [if(fused?, do: "paths", else: "sum")]
     end, ["left_disparity", if(projected?, do: "winners", else: "right_disparity")]},
    {"consistency_projected", ["left_disparity", "winners"], ["consistent"]},
    {"filter", if(projected?, do: ["consistent"], else: ["left_disparity", "right_disparity"]),
     ["output"]}
  ]
  |> Enum.reject(fn {name, _, _} -> skip?.(name) end)

directory =
  Path.join(System.tmp_dir!(), "stereo-census-" <> Base.encode16(:crypto.strong_rand_bytes(12)))

File.mkdir!(directory)

try do
  models =
    kernels
    |> Enum.reject(fn {name, _, _, _} -> skip?.(name) end)
    |> Map.new(fn {name, inputs, output, groups} ->
      spirv = Path.join(directory, name <> ".spv")

      source =
        case {name, selector} do
          {"census", _} when cost? -> "census_pair"
          {"first", _} -> "cost_first"
          {"paths", _} when cost? -> "cost_scan"
          {"select", _} when cost? -> "cost_scan"
          {"select", _} when vertical? -> "paths"
          {"select", "projected"} -> "select_projected"
          {"select", mode} when mode in ["rolling", "fused"] -> "select_rolling"
          _ -> name
        end

      args = [
        "-V",
        "--target-env",
        "vulkan1.1",
        "-S",
        "comp",
        "-DWIDTH=#{width}",
        "-DHEIGHT=#{height}",
        "-DDISPARITIES=#{disparities}",
        "-DROW_SCATTER=#{if(projection == "shared", do: 1, else: 0)}",
        "-DPATHS=#{if(vertical? and name == "paths", do: 2, else: paths)}",
        "-DFUSE_VERTICAL=#{if(vertical? and name == "select", do: 1, else: 0)}",
        "-DP1=#{p1}",
        "-DP2=#{p2}",
        "-DBLOCK=#{block}",
        "-DSPECIALIZE_DIRECTIONS=#{if schedule == "generic", do: 0, else: 1}",
        "-DCACHE_HORIZONTAL=#{if schedule in ["cached", "tiled"], do: 1, else: 0}",
        "-DTILE_ROWS=#{if schedule == "tiled", do: 1, else: 0}",
        "-DCLASSIC_CENSUS=#{if census == "classic", do: 1, else: 0}",
        "-DFEATURE_TILE=#{tile}",
        "-DFUSE_SUM=#{if selector == "fused", do: 1, else: 0}",
        "-DMEDIAN_ONLY=#{if projected?, do: 1, else: 0}",
        "-DSELECT_LANES=#{select_lanes}",
        "-DUNIQUENESS=#{uniqueness}",
        Path.join(__DIR__, source <> ".comp"),
        "-o",
        spirv
      ]

      for {tool, tool_args} <- [
            {System.fetch_env!("GLSLANG"), args},
            {System.fetch_env!("SPIRV_VAL"), ["--target-env", "vulkan1.1", spirv]}
          ] do
        case System.cmd(tool, tool_args, stderr_to_stdout: true) do
          {_, 0} -> :ok
          {log, status} -> raise("#{tool} exited #{status}: #{log}")
        end
      end

      binary =
        NxIREE.Vulkan.export(File.read!(spirv), inputs, output,
          workgroups: groups,
          target_features: features,
          external_resources_mappable: mappable == "true"
        )

      IO.puts("EXPORTED #{name} #{byte_size(binary)} bytes")
      {name, binary}
    end)

  models =
    if execution == "graph" do
      stages =
        Enum.map(pipeline, fn {name, inputs, outputs} ->
          {_, _, templates, groups} = Enum.find(kernels, &(elem(&1, 0) == name))
          templates = if is_tuple(templates), do: Tuple.to_list(templates), else: [templates]

          [
            spirv: File.read!(Path.join(directory, name <> ".spv")),
            inputs: Enum.map(inputs, &String.to_atom/1),
            outputs: Enum.zip(Enum.map(outputs, &String.to_atom/1), templates),
            workgroups: groups
          ]
        end)

      binary =
        NxIREE.Vulkan.export_pipeline([left: image, right: image], stages, :output,
          target_features: features,
          external_resources_mappable: mappable == "true"
        )

      IO.puts("EXPORTED pipeline #{byte_size(binary)} bytes")
      Map.put(models, "pipeline", binary)
    else
      models
    end

  bundle = %{
    models: models,
    shape: {height, width},
    disparities: disparities,
    paths: paths,
    p1: p1,
    p2: p2,
    uniqueness: uniqueness,
    block: block,
    selector: selector,
    select_lanes: select_lanes,
    schedule: schedule,
    census: census,
    execution: execution,
    projection: projection,
    fusion: fusion,
    volume_tile: if(cost?, do: tile),
    external_resources_mappable: mappable == "true",
    pipeline: pipeline
  }

  output = System.fetch_env!("STEREO_OUT")
  File.write!(output, :erlang.term_to_binary(bundle))
  IO.puts("WROTE #{output}")
after
  File.rm_rf!(directory)
end
