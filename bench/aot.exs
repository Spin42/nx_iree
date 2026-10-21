# Exports the Jetson test functions with NxIREE.export/3 for a Vulkan target:
#
#   * nvk (default) - a capability list NVK provides on the Nano; compiles
#     every model in real_models.exs
#   * generic - IREE's baseline profile plus 8-bit storage
#   * any named IREE target, e.g. pascal (its 32-wide subgroups match the
#     Maxwell GPU, but it fails to compile BERT- and GPT-style models)
#
#   NXIREE_VULKAN_TARGET=nvk NXIREE_FLAGS="..." AOT_OUT=jetson-modules.bin mix run aot.exs
Logger.configure(level: :warning)

target = System.get_env("NXIREE_VULKAN_TARGET", "nvk")
out = System.get_env("AOT_OUT", "jetson-modules.bin")

nvk_features =
  Enum.join(
    ~w(spirv:v1.3 cap:Shader cap:StorageBuffer8BitAccess cap:Int8 ext:SPV_KHR_8bit_storage
       cap:GroupNonUniform cap:GroupNonUniformArithmetic cap:GroupNonUniformShuffle
       cap:GroupNonUniformBallot cap:Int64 cap:Int16 cap:Float16 cap:StorageBuffer16BitAccess
       ext:SPV_KHR_16bit_storage),
    ","
  )

target_flags =
  case target do
    "nvk" -> ["--iree-vulkan-target-features=#{nvk_features}"]
    "generic" -> []
    named -> ["--iree-vulkan-target=#{named}"]
  end

flags =
  ["--iree-hal-target-backends=vulkan-spirv" | target_flags] ++
    String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)

export = fn fun, templates -> NxIREE.export(fun, templates, iree_compiler_flags: flags) end

chain = fn t ->
  t |> Nx.multiply(1.5) |> Nx.add(0.25) |> Nx.exp() |> Nx.multiply(t) |> Nx.sum()
end

models =
  Map.new(
    [
      {{:add, 256}, export.(fn t -> Nx.add(t, t) end, [Nx.template({256}, :f32)])},
      {{:chain, 4_000_000}, export.(chain, [Nx.template({4_000_000}, :f32)])}
    ] ++
      for n <- [256, 512, 1024, 2048] do
        t = Nx.template({n, n}, :f32)
        {{:matmul, n}, export.(&Nx.dot/2, [t, t])}
      end
  )

File.write!(out, :erlang.term_to_binary(%{target: target, flags: flags, models: models}))

for {k, bin} <- Enum.sort(models), do: IO.puts("#{inspect(k)}: #{byte_size(bin)} bytes")
IO.puts("wrote #{out}")
