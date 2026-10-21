# Exports the Jetson test functions with NxIREE.export/3 for a Vulkan target
# (default pascal: the closest IREE preset to the Nano's Maxwell GPU, and its
# 32-wide subgroups match the hardware).
#
#   NXIREE_VULKAN_TARGET=pascal NXIREE_FLAGS="..." AOT_OUT=jetson-modules.bin mix run aot.exs
Logger.configure(level: :warning)

target = System.get_env("NXIREE_VULKAN_TARGET", "pascal")
out = System.get_env("AOT_OUT", "jetson-modules.bin")

flags =
  ["--iree-hal-target-backends=vulkan-spirv", "--iree-vulkan-target=#{target}"] ++
    String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)

export = fn fun, templates -> NxIREE.export(fun, templates, iree_compiler_flags: flags) end

chain = fn t -> t |> Nx.multiply(1.5) |> Nx.add(0.25) |> Nx.exp() |> Nx.multiply(t) |> Nx.sum() end

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
