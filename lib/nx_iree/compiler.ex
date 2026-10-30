defmodule NxIREE.Compiler do
  @moduledoc """
  Compiler for Nx defn
  """

  alias NxIREE.Compiler.GraphSplitter
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  def to_bytecode(fun, templates, opts \\ []) do
    opts = opts |> Keyword.put(:output_mode, :bytecode) |> Keyword.put(:compiler, __MODULE__)

    Nx.Defn.compile(fun, templates, opts)
  catch
    {:bytecode, result} ->
      {:ok, result}
  end

  @behaviour Nx.Defn.Compiler

  # Compiled functions for `__jit__`, keyed by defn key, input templates and options
  @cache __MODULE__.Cache

  @impl true
  def __compile__(_key, vars, fun, opts) do
    {iree_compiler_flags, opts} = Keyword.pop(opts, :iree_compiler_flags, [])
    {iree_runtime_options, opts} = Keyword.pop(opts, :iree_runtime_options, [])
    {output_mode, opts} = Keyword.pop(opts, :output_mode, nil)

    {module_cache, opts} =
      Keyword.pop_lazy(opts, :iree_module_cache, fn ->
        Application.get_env(:nx_iree, :module_cache)
      end)

    {device_indexing, opts} = Keyword.pop(opts, :iree_device_indexing, false)

    unless is_list(iree_compiler_flags) do
      raise "missing :iree_compiler_flags option"
    end

    has_target_backend_flag? =
      Enum.any?(iree_compiler_flags, &String.starts_with?(&1, "--iree-hal-target-backends"))

    {iree_compiler_flags, backend} =
      cond do
        is_nil(iree_runtime_options[:device]) and not has_target_backend_flag? ->
          %{compiler_target_backend: backend} = NxIREE.Device.fetch!(nil)
          flag = "--iree-hal-target-backends=#{backend}"
          {[flag | iree_compiler_flags], backend}

        not has_target_backend_flag? ->
          %{compiler_target_backend: backend} =
            NxIREE.Device.fetch!(iree_runtime_options[:device])

          {["--iree-hal-target-backends=#{backend}" | iree_compiler_flags], backend}

        true ->
          "--iree-hal-target-backends=" <> backend =
            Enum.find(
              iree_compiler_flags,
              &String.starts_with?(&1, "--iree-hal-target-backends=")
            )

          {iree_compiler_flags, backend}
      end

    iree_compiler_flags =
      iree_compiler_flags
      |> default_flag("--iree-input-type", "stablehlo_xla")
      |> default_flag("--iree-execution-model", "async-internal")
      |> default_vulkan_features(backend)
      |> default_memoization(backend)
      |> default_cpu_target(backend, output_mode)
      |> default_f64_flags(backend)

    ensure_f64_supported!(vars, iree_compiler_flags, backend)

    # Opt-in: removes a host round trip per dynamic slice, which gains nothing
    # where round trips are short, as on desktop GPUs
    fun =
      if device_indexing == true,
        do: fn vars -> fun.(vars) |> NxIREE.Compiler.DeviceIndexing.rewrite() end,
        else: fun

    exla_opts = opts |> Keyword.put(:within_defn_compiler, true) |> Keyword.put(:client, :host)

    if output_mode != :bytecode and backend == "metal-spirv" do
      compile_with_graph_splitter(
        fun,
        vars,
        exla_opts,
        iree_compiler_flags,
        iree_runtime_options
      )
    else
      compile_without_graph_splitter(
        fun,
        vars,
        exla_opts,
        iree_compiler_flags,
        iree_runtime_options,
        output_mode,
        module_cache
      )
    end
  end

  defp compile_without_graph_splitter(
         fun,
         vars,
         exla_opts,
         iree_compiler_flags,
         iree_runtime_options,
         output_mode,
         module_cache
       ) do
    {nx_iree_module, pid_size} =
      cached_or_compiled(module_cache, fun, vars, exla_opts, iree_compiler_flags)

    used_inputs = nx_iree_module.used_inputs

    if output_mode == :bytecode do
      throw({:bytecode, nx_iree_module})
    else
      device = NxIREE.Device.fetch!(iree_runtime_options[:device])
      {:ok, nx_iree_module} = NxIREE.load(nx_iree_module, device: device)
      leading_inputs = leading_inputs(device, pid_size)

      fn [inputs] ->
        filtered_inputs =
          filter_inputs_by_indices(inputs, used_inputs)

        {:ok, result} =
          NxIREE.call(
            nx_iree_module,
            leading_inputs ++ filtered_inputs,
            Keyword.put(iree_runtime_options, :device, device)
          )

        [result]
      end
    end
  end

  defp cached_or_compiled(nil, fun, vars, exla_opts, flags) do
    compile_module(fun, vars, exla_opts, flags, nil)
  end

  defp cached_or_compiled(dir, fun, vars, exla_opts, flags) do
    path = NxIREE.ModuleCache.path(dir, NxIREE.ModuleCache.key(fun.(vars), vars, flags))

    case NxIREE.ModuleCache.read(path) do
      {:ok, cached} ->
        cached

      :error ->
        {module, pid_size} = compile_module(fun, vars, exla_opts, flags, path)
        NxIREE.ModuleCache.write!(path, module, pid_size)
        {module, pid_size}
    end
  end

  defp compile_module(fun, vars, exla_opts, flags, cache_path) do
    unless Code.ensure_loaded?(EXLA) do
      hint =
        if cache_path,
          do:
            ", and the module cache has no entry #{Path.basename(cache_path)}: compile " <>
              "this function on the host with the same :iree_compiler_flags and " <>
              ":iree_module_cache",
          else: "; without it, use NxIREE.export/3 or :iree_module_cache"

      raise NxIREE.CompileError, message: "compiling defn functions needs EXLA" <> hint
    end

    hooks = Keyword.get(exla_opts, :hooks, %{})
    fun = fn vars -> fun.(vars) |> NxIREE.Compiler.HostCalls.strip!(hooks) end

    # EXLA's own cache is keyed without the target, which decides whether
    # NxIREE.Vulkan.Kernel blocks lower to dispatches; NxIREE caches modules itself
    {%{mlir_module: mlir_module, output_container: output_container, used_inputs: used_inputs},
     kernels} =
      NxIREE.Vulkan.Kernel.lowering(target_backend(flags), fn ->
        EXLA.to_mlir_module(fun, vars, Keyword.put(exla_opts, :cache, false))
      end)

    module =
      with_kernel_directory(kernels, fn directory ->
        mlir_module
        |> NxIREE.Vulkan.MLIR.inline_kernels(kernels, directory)
        |> NxIREE.compile(flags, output_container: output_container)
      end)

    {%{module | compilation_flags: flags, used_inputs: Enum.sort(used_inputs)},
     callback_pid_size()}
  end

  defp target_backend(flags) do
    Enum.find_value(flags, fn
      "--iree-hal-target-backends=" <> backend -> backend
      _ -> nil
    end)
  end

  defp with_kernel_directory([], fun), do: fun.(nil)

  defp with_kernel_directory(_kernels, fun) do
    directory =
      Path.join(System.tmp_dir!(), "nx_iree-kernels-#{System.unique_integer([:positive])}")

    File.mkdir_p!(directory)

    try do
      fun.(directory)
    after
      File.rm_rf(directory)
    end
  end

  # EXLA emits StableHLO, and NxIREE.call/3 invokes modules synchronously
  # without fences: other execution models return before the device work is
  # done, so results read back as zeros.
  defp default_flag(flags, name, value) do
    if Enum.any?(flags, &String.starts_with?(&1, name <> "=")),
      do: flags,
      else: flags ++ ["#{name}=#{value}"]
  end

  # IREE's default Vulkan target has no 8-bit storage, and IREE 3.13 then
  # misreads i8 tensors consumed by vectorized dispatches (e.g. an attention
  # mask fused into a batch matmul): each byte is read 4 bytes apart. Unless a
  # target or feature set is chosen, declare 8-bit storage on top of the
  # baseline; Vulkan 1.2+ desktop and Jetson drivers all provide it.
  @vulkan_baseline_with_int8 "spirv:v1.3,cap:Shader,cap:StorageBuffer8BitAccess,cap:Int8,ext:SPV_KHR_8bit_storage"

  defp default_vulkan_features(flags, "vulkan-spirv") do
    if Enum.any?(flags, &String.starts_with?(&1, "--iree-vulkan-target")),
      do: flags,
      else: flags ++ ["--iree-vulkan-target-features=#{@vulkan_baseline_with_int8}"]
  end

  defp default_vulkan_features(flags, _backend), do: flags

  # Functions compiled for a call run on this machine, so they can use its
  # CPU features; bytecode for export/3 runs elsewhere and targets a generic
  # CPU of the triple. Either way IREE is told, rather than warning and
  # falling back to a generic CPU.
  defp default_cpu_target(flags, "llvm-cpu", output_mode) do
    chosen? =
      Enum.any?(flags, fn flag ->
        String.starts_with?(flag, "--iree-llvmcpu-target-cpu") or
          String.starts_with?(flag, "--iree-llvmcpu-target-triple")
      end)

    flags =
      cond do
        chosen? -> flags
        output_mode == :bytecode -> flags ++ ["--iree-llvmcpu-target-cpu=generic"]
        true -> flags ++ ["--iree-llvmcpu-target-cpu=host"]
      end

    # Data tiling packs matmul operands into the CPU's preferred layout, and
    # im2col turns convolutions into such matmuls: ViT-base runs about 6x and
    # ResNet-50 about 4x faster with them
    flags
    |> default_switch(
      ~w(--iree-opt-data-tiling --iree-dispatch-creation-data-tiling --iree-global-opt-data-tiling),
      "--iree-opt-data-tiling"
    )
    |> default_switch(
      ["--iree-global-opt-use-im2col-for-convs"],
      "--iree-global-opt-use-im2col-for-convs"
    )
  end

  defp default_cpu_target(flags, _backend, _output_mode), do: flags

  defp default_switch(flags, switches, flag) do
    if Enum.any?(flags, fn given -> Enum.any?(switches, &String.starts_with?(given, &1)) end),
      do: flags,
      else: flags ++ [flag]
  end

  # IREE 3.13 memoizes command buffers with indirect bindings, and on Vulkan
  # a memoized buffer keeps the slice offsets of the call that recorded it:
  # later calls with other dynamic offsets read the wrong window. Recording
  # per call costs nothing measurable on the models in bench/real_models.exs.
  defp default_memoization(flags, backend) when backend in ["llvm-cpu", "vmvx"], do: flags

  defp default_memoization(flags, _backend),
    do: default_flag(flags, "--iree-hal-memoization", "false")

  @demote_f64 "--iree-input-demote-f64-to-f32"

  # IREE demotes f64 to f32 unless told otherwise, which would leave the
  # module expecting f32 buffers where Nx passes f64. The CPU backend computes
  # in f64, so keep f64 there unless the caller chose.
  defp default_f64_flags(flags, "llvm-cpu") do
    if Enum.any?(flags, &String.starts_with?(&1, @demote_f64)),
      do: flags,
      else: flags ++ ["#{@demote_f64}=false"]
  end

  defp default_f64_flags(flags, _backend), do: flags

  # The SPIR-V backends cannot compile f64 math, so f64 inputs fail with a
  # clear message instead of an element type mismatch at call time.
  defp ensure_f64_supported!(vars, flags, backend) do
    demoting? = "#{@demote_f64}=false" not in flags

    f64_input? =
      vars
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.any?(&(&1.type == {:f, 64}))

    if demoting? and f64_input? do
      raise ArgumentError,
            "f64 tensors are not supported on the #{backend} IREE backend " <>
              "(IREE demotes f64 to f32 there); convert the inputs to f32"
    end
  end

  # EXLA modules take the encoded callback server pid as their first argument.
  # IREE runs no host callbacks, so the argument is a zero buffer allocated
  # once per compiled function.
  defp callback_pid_inputs(%NxIREE.Device{} = device) do
    leading_inputs(device, callback_pid_size())
  end

  @doc false
  # Size of the callback pid argument EXLA prepends to every module (0 when
  # the EXLA in use does not add it).
  def callback_pid_size do
    if Code.ensure_loaded?(EXLA.Executable) and
         function_exported?(EXLA.Executable, :callback_server_pid_size, 0) do
      apply(EXLA.Executable, :callback_server_pid_size, [])
    else
      0
    end
  end

  @doc false
  # The zero buffer passed as the callback pid argument
  def leading_inputs(_device, 0), do: []

  def leading_inputs(%NxIREE.Device{} = device, size) do
    {:ok, ref} =
      NxIREE.VM.allocate_buffer(<<0::size(size)-unit(8)>>, device.ref, {size}, {:u, 8})

    data = %NxIREE.Backend{
      ref: ref,
      device: device.ref,
      device_uri: device.uri,
      driver: device.driver_name
    }

    [%T{data: data, shape: {size}, type: {:u, 8}, names: [nil]}]
  end

  defp compile_with_graph_splitter(
         fun,
         vars,
         exla_opts,
         iree_compiler_flags,
         iree_runtime_options
       ) do
    expr = fun.(vars)

    {stages, _, _} = GraphSplitter.traverse(expr)

    function_chain =
      for {stage_id, tag, expr, arguments, argument_sources} <- stages do
        fun = fn _ -> expr end

        mlir_args =
          arguments
          |> Map.values()
          |> Enum.sort_by(fn
            %T{data: %Expr{op: :parameter, args: [idx]}} -> idx
            _ -> nil
          end)

        %{mlir_module: mlir_module, output_container: output_container, used_inputs: used_inputs} =
          EXLA.to_mlir_module(fun, mlir_args, exla_opts)

        iree_compiler_flags =
          if tag == :force_host do
            Enum.map(iree_compiler_flags, fn
              "--iree-hal-target-backends=metal-spirv" -> "--iree-hal-target-backends=llvm-cpu"
              flag -> flag
            end)
          else
            iree_compiler_flags
          end

        iree_runtime_options =
          if tag == :force_host do
            device = NxIREE.Device.fetch!("local-sync://")
            Keyword.put(iree_runtime_options, :device, device)
          else
            Keyword.put_new(
              iree_runtime_options,
              :device,
              NxIREE.Device.find_default_device("metal")
            )
          end

        nx_iree_module =
          NxIREE.compile(mlir_module, iree_compiler_flags, output_container: output_container)

        stage_device = NxIREE.Device.fetch!(iree_runtime_options[:device])
        {:ok, nx_iree_module} = NxIREE.load(nx_iree_module, device: stage_device)
        leading_inputs = callback_pid_inputs(stage_device)

        runtime_fun = fn [inputs] ->
          filtered_inputs =
            filter_inputs_by_indices(inputs, used_inputs)

          {:ok, result} =
            NxIREE.call(
              nx_iree_module,
              leading_inputs ++ filtered_inputs,
              iree_runtime_options
            )

          [result]
        end

        {stage_id, tag, iree_runtime_options[:device], runtime_fun, arguments, argument_sources}
      end

    fn [args] ->
      input_sources =
        args
        |> Enum.with_index()
        |> Map.new(fn {arg, idx} ->
          {{nil, idx}, arg}
        end)

      {_input_sources, result} =
        for {stage_id, _tag, device, runtime_fun, arguments, argument_sources} <- function_chain,
            reduce: {input_sources, nil} do
          {input_sources, _prev_result} ->
            {args, input_sources} =
              Enum.map_reduce(arguments, input_sources, fn {id, param}, input_sources ->
                %Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [idx]}} = param
                source_key = argument_sources[id]

                case input_sources[source_key] do
                  input when is_function(input, 0) ->
                    val = input.()

                    val =
                      Nx.Defn.Composite.traverse(val, fn
                        %Nx.Tensor{data: %NxIREE.Backend{}} = t -> t
                        t -> Nx.backend_transfer(t, {NxIREE.Backend, device: device})
                      end)

                    {{idx, val}, Map.put(input_sources, source_key, val)}

                  val ->
                    {{idx, val}, input_sources}
                end
              end)

            args =
              args
              |> Enum.sort_by(&elem(&1, 0))
              |> Enum.map(fn {_idx, arg} ->
                {:ok, buffer_ref} = NxIREE.VM.allocate_buffer(arg, device.ref)
                arg = put_in(arg.data.ref, buffer_ref)
                put_in(arg.data.device, device.ref)
                put_in(arg.data.device_uri, device.uri)
                put_in(arg.data.driver, device.driver_name)
              end)

            [results] = runtime_fun.([args])

            input_sources =
              [results]
              |> Nx.Defn.Composite.flatten_list()
              |> Enum.with_index()
              |> Enum.reduce(input_sources, fn {result, idx}, acc ->
                Map.put(acc, {stage_id, idx}, result)
              end)

            {input_sources, [results]}
        end

      result
    end
  end

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    templates =
      vars
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.map(&{&1.shape, &1.type, &1.names})

    cache_key = {key, templates, opts}

    compiled =
      case :ets.lookup(@cache, cache_key) do
        [{_, compiled}] ->
          compiled

        [] ->
          compiled = __compile__(key, vars, fun, opts)
          :ets.insert(@cache, {cache_key, compiled})
          compiled
      end

    compiled.(args_list)
  end

  @doc false
  def create_cache do
    :ets.new(@cache, [:named_table, :public, :set, read_concurrency: true])
  end

  # Each computation runs on a single device
  @impl true
  def __partitions_options__(opts), do: [opts]

  @impl true
  def __to_backend__(opts) do
    device = opts |> Keyword.get(:iree_runtime_options, []) |> Keyword.get(:device)
    {NxIREE.Backend, device: NxIREE.Device.fetch!(device)}
  end

  defp filter_inputs_by_indices(args, inputs) do
    filter_by_indices_list(args, 0, Enum.sort(inputs), fn x, _ -> x end)
  end

  defp filter_by_indices_list([var | vars], i, [i | inputs], callback),
    do: [callback.(var, i) | filter_by_indices_list(vars, i + 1, inputs, callback)]

  defp filter_by_indices_list([_var | vars], i, inputs, callback),
    do: filter_by_indices_list(vars, i + 1, inputs, callback)

  # Nx counts a block's parameters as inputs of the enclosing function, so
  # used inputs may index past its arguments; EXLA skips those too.
  defp filter_by_indices_list([], _i, _inputs, _callback),
    do: []
end
