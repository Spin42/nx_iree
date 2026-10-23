glslang = System.fetch_env!("GLSLANG")
validator = System.fetch_env!("SPIRV_VAL")

directory =
  Path.join(System.tmp_dir!(), "vulkan-stream-" <> Base.encode16(:crypto.strong_rand_bytes(12)))

File.mkdir!(directory)

try do
  spirv = Path.join(directory, "stream.spv")

  for {tool, args} <- [
        {glslang,
         ["-V", "--target-env", "vulkan1.1", Path.join(__DIR__, "stream.comp"), "-o", spirv]},
        {validator, ["--target-env", "vulkan1.1", spirv]}
      ] do
    case System.cmd(tool, args, stderr_to_stdout: true) do
      {_, 0} -> :ok
      {log, status} -> raise("#{tool} exited #{status}: #{log}")
    end
  end

  flags = ["--iree-hal-target-backends=vulkan-spirv"]

  models =
    Map.new([1_048_576, 4_194_304], fn words ->
      tensor = Nx.template({words}, :u32)
      kernel = %NxIREE.Vulkan.Kernel{spirv: File.read!(spirv), workgroups: {div(words, 128), 1, 1}}

      stream = fn x ->
        Nx.block(kernel, [x], tensor, fn _kernel, x -> Nx.bitwise_xor(x, 0x5A5A5A5A) end)
      end

      {words, NxIREE.export(stream, [tensor], iree_compiler_flags: flags)}
    end)

  File.write!(System.fetch_env!("VULKAN_BUNDLE"), :erlang.term_to_binary(models))
after
  File.rm_rf!(directory)
end
