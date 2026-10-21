defmodule NxIREE do
  @moduledoc """
  IREE compiler and runtime bindings for Nx.

  NxIREE provides an `Nx.Defn` compiler which runs on [IREE](https://iree.dev).
  The following example shows how we can compile a function to run on the CPU:

      f = &Nx.add/2
      compiler_flags = ["--iree-hal-target-backends=llvm-cpu", "--iree-input-type=stablehlo_xla", "--iree-execution-model=async-internal"]
      opts = [compiler: NxIREE.Compiler, iree_compiler_flags: compiler_flags, iree_runtime_options: [device: "local-sync://"])
      Nx.Defn.compile(f, [Nx.template({1}, :f32), Nx.template({3}, :f32)], opts)

  And this next example compiles the same function for running on the Apple GPU through Metal:

      f = &Nx.add/2
      compiler_flags = ["--iree-hal-target-backends=metal-spirv", "--iree-input-type=stablehlo_xla", "--iree-execution-model=async-internal"]
      opts = [compiler: NxIREE.Compiler, iree_compiler_flags: compiler_flags, iree_runtime_options: [device: "metal://"])
      Nx.Defn.compile(f, [Nx.template({1}, :f32), Nx.template({3}, :f32)], opts)

  `NxIREE.Compiler` also provides a `to_bytecode` function which outputs the bytecode for usage with embedded devices,
  such as the iOS devices usable through [`LiveNxIREE`](https://github.com/elixir-nx/nx_iree/tree/main/embedded_devices/live_nx_iree).
  """

  require Logger

  @doc """
  Compiles the given MLIR module with the given list of flags.

  Returns the bytecode for the compiled module.

  ## Examples

      iex> mlir_module = \"""
      ...> func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
      ...>   %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      ...>   return %0 : tensor<4xf32>
      ...> }
      ...>\"""
      iex> flags = ["--iree-hal-target-backends=llvm-cpu", "--iree-input-type=stablehlo_xla", "--iree-execution-model=async-internal"]
      iex> NxIREE.compile(mlir_module, flags)
  """
  def compile(mlir_module, flags, opts \\ []) do
    output_container = opts[:output_container]
    {:ok, tmpfile} = create_temp_file(mlir_module)

    compiler_path = opts[:compiler_path] || Path.join(:code.priv_dir(:nx_iree), "iree-compile")
    outfile = tmpfile <> ".vmfb"

    unless File.exists?(compiler_path) do
      raise NxIREE.CompileError,
        message:
          "iree-compile not found at #{compiler_path}. Devices built without the " <>
            "compiler run modules compiled on the host with NxIREE.export/3"
    end

    phases_dir = tmpfile <> ".phases"

    phase_flags =
      if Enum.any?(flags, &String.starts_with?(&1, "--dump-compilation-phases-to")),
        do: [],
        else: ["--dump-compilation-phases-to=#{phases_dir}"]

    try do
      case System.cmd(compiler_path, flags ++ phase_flags ++ ["-o", outfile, tmpfile],
             stderr_to_stdout: true
           ) do
        {log, 0} ->
          if log != "", do: Logger.debug("iree-compile: " <> log)

          %NxIREE.Module{
            bytecode: File.read!(outfile),
            compilation_flags: flags,
            mlir_module: mlir_module,
            output_container: output_container,
            written_inputs: written_inputs(phases_dir)
          }

        {log, status} ->
          raise NxIREE.CompileError,
            message: "iree-compile exited with status #{status}:\n" <> compile_errors(log)
      end
    after
      File.rm(tmpfile)
      File.rm(outfile)
      File.rm_rf(phases_dir)
    end
  end

  # Arguments IREE may write in place, from the stream phase of the compilation
  defp written_inputs(phases_dir) do
    with [path] <- Path.wildcard(Path.join(phases_dir, "*.stream.mlir")),
         {:ok, indices} <- path |> File.read!() |> NxIREE.WrittenInputs.analyze() do
      indices
    else
      _ -> :all
    end
  end

  # The compiler's diagnostics with the CPU-target advice it prints for every
  # module left out
  defp compile_errors(log) do
    log
    |> String.split("\n")
    |> Enum.filter(&String.contains?(&1, "error"))
    |> case do
      [] -> log
      errors -> Enum.join(errors, "\n")
    end
  end

  defp create_temp_file(content) do
    tmpfile = Path.join(System.tmp_dir!(), "#{System.unique_integer()}-nx-iree-tempfile.mlir")

    case File.write(tmpfile, content) do
      :ok -> {:ok, tmpfile}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Loads the module on a device once, so that `call/3` on that device reuses
  its pipelines instead of recreating them on every call.

  ## Options

    * `:device` - The device to load the module on. Defaults to the same device
      `call/3` would pick.
  """
  def load(%NxIREE.Module{bytecode: bytecode} = module, opts \\ []) do
    opts = Keyword.validate!(opts, device: nil)

    %NxIREE.Device{driver_name: driver_name, ref: device_ref, uri: device_uri} =
      NxIREE.Device.fetch!(opts[:device])

    instance_ref = NxIREE.VM.get_instance()

    case NxIREE.Native.load_module(instance_ref, device_ref, driver_name, bytecode) do
      {:ok, loaded_ref} -> {:ok, %{module | loaded: {device_uri, loaded_ref}}}
      {:error, error} -> {:error, List.to_string(error)}
    end
  end

  @doc """
  Calls a function in the given module with the provided Nx inputs.

  ## Options

    # `:function` - The name of the function to call in the module. If not provided, will default to `"main"`.
    * `:device` - The device to run the module on. If not provided, will default to known GPU devices (CUDA, ROCm, Metal, Vulkan) over others.
      Valid values can be obtained through `list_devices/0` or `list_devices/1`.
  """
  def call(
        %NxIREE.Module{bytecode: bytecode, output_container: output_container} = module,
        inputs,
        opts \\ []
      ) do
    opts = Keyword.validate!(opts, function: "main", device: nil)

    %NxIREE.Device{driver_name: driver_name, ref: device_ref, uri: device_uri} =
      NxIREE.Device.fetch!(opts[:device])

    written = module.written_inputs || :all

    {input_refs, copy_inputs} =
      inputs
      |> Enum.with_index()
      |> Enum.map(fn {input, i} ->
        input = if is_function(input, 0), do: input.(), else: input

        case input do
          # Buffers are only valid on the device that allocated them; tensors
          # on other devices are copied by allocate_buffer below. Existing
          # buffers the module may write are passed as copies.
          %Nx.Tensor{data: %NxIREE.Backend{ref: ref, device: ^device_ref}} ->
            {ref, written == :all or i in written}

          t ->
            {:ok, ref} = NxIREE.VM.allocate_buffer(t, device_ref)
            {ref, false}
        end
      end)
      |> Enum.unzip()

    result =
      case module.loaded do
        {^device_uri, loaded_ref} ->
          NxIREE.Native.call_loaded_io(loaded_ref, driver_name, input_refs, copy_inputs)

        _ ->
          instance_ref = NxIREE.VM.get_instance()

          NxIREE.Native.call_io(
            instance_ref,
            device_ref,
            driver_name,
            bytecode,
            input_refs,
            copy_inputs
          )
      end

    case result do
      {:ok, refs} ->
        {tensors, []} =
          Nx.Defn.Composite.traverse(output_container, refs, fn hole,
                                                                [{ref, _dims, _type_str} | refs] ->
            data = %NxIREE.Backend{
              ref: ref,
              data: nil,
              device_uri: device_uri,
              device: device_ref,
              driver: driver_name
            }

            {%{hole | data: data}, refs}
          end)

        # tensors =
        #   Enum.map(refs, fn {ref, dims, type_str} ->
        #     %Nx.Tensor{
        #       names: Enum.map(dims, fn _ -> nil end),
        #       type: type_str_to_nx(type_str),
        #       shape: List.to_tuple(dims),
        #       data: %NxIREE.Backend{
        #         ref: ref,
        #         data: nil,
        #         device_uri: device,
        #         device: device_ref,
        #         driver: driver_name
        #       }
        #     }
        #   end)

        {:ok, tensors}

      {:error, error} ->
        raise "IREE call failed due to: #{inspect(error)}"
    end
  end

  @doc """
  Compiles `fun` ahead of time for the given argument templates and returns a
  binary to load on a device with `load_exported/2`.

  `args` are the function's arguments, as tensors or `Nx.template/2`
  templates (containers are accepted); only their shapes and types matter.
  This runs EXLA and the IREE compiler, so it belongs on the host.

  ## Options

    * `:iree_compiler_flags` - flags for `iree-compile`. They must name the
      target with `--iree-hal-target-backends`, e.g. `vulkan-spirv` plus
      `--iree-vulkan-target=pascal` for a Jetson Nano.

  ## Examples

      flags = ["--iree-hal-target-backends=llvm-cpu", "--iree-input-type=stablehlo_xla"]
      exported = NxIREE.export(&Nx.dot/2, [Nx.template({4, 4}, :f32), Nx.template({4, 4}, :f32)],
        iree_compiler_flags: flags)

      # on the device
      model = NxIREE.load_exported(exported, device: "local-sync://")
      NxIREE.run(model, [a, b])
  """
  def export(fun, args, opts \\ []) when is_function(fun) and is_list(args) do
    opts = Keyword.validate!(opts, iree_compiler_flags: [])
    flags = opts[:iree_compiler_flags]

    unless Enum.any?(flags, &String.starts_with?(&1, "--iree-hal-target-backends=")) do
      raise ArgumentError,
            "export/3 needs --iree-hal-target-backends in :iree_compiler_flags, " <>
              "since the target device is not known on the host"
    end

    templates = Enum.map(args, &Nx.Defn.Composite.traverse(&1, fn t -> Nx.to_template(t) end))

    {:ok, module} = NxIREE.Compiler.to_bytecode(fun, templates, iree_compiler_flags: flags)

    inputs =
      templates
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.map(&{&1.shape, &1.type})

    :erlang.term_to_binary(%{
      format: NxIREE.Exported.format(),
      bytecode: module.bytecode,
      compilation_flags: flags,
      inputs: inputs,
      used_inputs: module.used_inputs,
      written_inputs: module.written_inputs,
      output: Nx.Defn.Composite.traverse(module.output_container, &Nx.to_template/1),
      callback_pid_size: NxIREE.Compiler.callback_pid_size()
    })
  end

  @doc """
  Loads a binary produced by `export/3` on a device, once, for repeated
  `run/2` calls. Needs neither EXLA nor the IREE compiler.

  ## Options

    * `:device` - the device to load on, as a URI or `NxIREE.Device`.
      Defaults to the same device `call/3` would pick. It must match the
      backend the binary was compiled for.
  """
  def load_exported(binary, opts \\ []) when is_binary(binary) do
    opts = Keyword.validate!(opts, device: nil)
    format = NxIREE.Exported.format()

    data =
      case :erlang.binary_to_term(binary, [:safe]) do
        %{format: ^format} = data ->
          data

        %{format: other} ->
          raise ArgumentError, "unsupported export format #{inspect(other)}, expected #{format}"

        _ ->
          raise ArgumentError, "not a binary produced by NxIREE.export/3"
      end

    device = NxIREE.Device.fetch!(opts[:device])

    module = %NxIREE.Module{
      bytecode: data.bytecode,
      compilation_flags: data.compilation_flags,
      output_container: data.output,
      used_inputs: data.used_inputs,
      written_inputs: data.written_inputs
    }

    case load(module, device: device) do
      {:ok, module} ->
        %NxIREE.Exported{
          module: module,
          device: device,
          inputs: data.inputs,
          used_inputs: data.used_inputs,
          leading_inputs: NxIREE.Compiler.leading_inputs(device, data.callback_pid_size)
        }

      {:error, error} ->
        raise ArgumentError, "could not load the exported module on #{device.uri}: #{error}"
    end
  end

  @doc """
  Runs a function loaded with `load_exported/2`.

  `args` must match the shapes and types given to `export/3`. Results stay on
  the device as `NxIREE.Backend` tensors.
  """
  def run(%NxIREE.Exported{} = exported, args) when is_list(args) do
    tensors =
      args
      |> Enum.map(&Nx.Defn.Composite.traverse(&1, fn t -> Nx.to_tensor(t) end))
      |> Nx.Defn.Composite.flatten_list()

    if length(tensors) != length(exported.inputs) do
      raise ArgumentError,
            "expected #{length(exported.inputs)} input tensors, got #{length(tensors)}"
    end

    Enum.zip(tensors, exported.inputs)
    |> Enum.with_index()
    |> Enum.each(fn {{t, {shape, type}}, i} ->
      if t.shape != shape or t.type != type do
        raise ArgumentError,
              "input #{i} must be #{inspect(type)} #{inspect(shape)}, " <>
                "got #{inspect(t.type)} #{inspect(t.shape)}"
      end
    end)

    used = for {t, i} <- Enum.with_index(tensors), i in exported.used_inputs, do: t

    case call(exported.module, exported.leading_inputs ++ used, device: exported.device) do
      {:ok, result} -> result
    end
  end

  @doc """
  Lists all devices available for running IREE modules.
  """
  @spec list_devices(String.t()) :: {:ok, list(String.t())}
  def list_devices do
    # This function returns a tagged tuple for uniformity with the arity-1 clause.
    NxIREE.Device.list()
  end

  @doc """
  Lists all devices available in a given driver for running IREE modules.

  Valid drivers can be obtained through `list_drivers/0`.
  """
  @spec list_devices(String.t()) :: {:ok, list(String.t())} | {:error, :unknown_driver}
  def list_devices(driver) do
    NxIREE.Device.list(driver)
  end

  @doc """
  Lists all drivers available for running IREE modules.
  """
  @spec list_drivers() :: {:ok, list(String.t())} | {:error, :unknown_driver}
  def list_drivers do
    NxIREE.Device.list_drivers()
  end
end
