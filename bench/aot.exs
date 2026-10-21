# Compiles test functions ahead of time for the Jetson (vulkan-spirv, pascal
# target) and writes them with plain output templates to a term file.
Logger.configure(level: :warning)

target = System.get_env("NXIREE_VULKAN_TARGET", "pascal")
out = System.get_env("AOT_OUT", "jetson-modules.bin")

flags = [
  "--iree-hal-target-backends=vulkan-spirv",
  "--iree-input-type=stablehlo_xla",
  "--iree-execution-model=async-internal",
  "--iree-vulkan-target=#{target}"
] ++ String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)

opts = [iree_compiler_flags: flags]

to_module = fn fun, templates ->
  {:ok, module} = NxIREE.Compiler.to_bytecode(fun, templates, opts)
  container = Nx.Defn.Composite.traverse(module.output_container, &Nx.to_template/1)
  %{module | output_container: container, mlir_module: nil}
end

matmuls =
  for n <- [256, 512, 1024, 2048] do
    t = Nx.template({n, n}, :f32)
    {{:matmul, n}, to_module.(&Nx.dot/2, [t, t])}
  end

chain = fn t -> t |> Nx.multiply(1.5) |> Nx.add(0.25) |> Nx.exp() |> Nx.multiply(t) |> Nx.sum() end
chain_mod = {{:chain, 4_000_000}, to_module.(chain, [Nx.template({4_000_000}, :f32)])}

add_mod = {{:add, 256}, to_module.(fn t -> Nx.add(t, t) end, [Nx.template({256}, :f32)])}

modules = Map.new([add_mod, chain_mod | matmuls])

pid_size =
  if function_exported?(EXLA.Executable, :callback_server_pid_size, 0),
    do: apply(EXLA.Executable, :callback_server_pid_size, []),
    else: 0

File.write!(out, :erlang.term_to_binary(%{modules: modules, callback_pid_size: pid_size, target: target}))

for {k, m} <- modules, do: IO.puts("#{inspect(k)}: #{byte_size(m.bytecode)} bytes")
IO.puts("callback pid size #{pid_size}, wrote #{out}")
