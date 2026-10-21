defmodule NxIREE.ModuleCache do
  @moduledoc """
  A directory of compiled modules shared between machines.

  With `:iree_module_cache` in the `NxIREE.Compiler` options (or
  `config :nx_iree, module_cache: dir`), each compiled function is stored
  under a key derived from its defn expression, input shapes and compiler
  flags. A node that builds the same function finds the module there and
  loads it without EXLA or the IREE compiler, so devices can run unchanged
  code, Bumblebee servings included, from modules compiled on a host:

      # on the host, for the device's target
      opts = [
        compiler: NxIREE.Compiler,
        iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv", "--iree-vulkan-target=pascal"],
        iree_module_cache: "priv/nx_iree_cache"
      ]

      Nx.Defn.compile(&MyModel.predict/2, templates, opts)

  Ship the directory with the device application and give the device the
  same options. The flags must match: they are part of the key.
  """

  @doc false
  def key(expr, vars, flags) do
    templates =
      vars
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.map(&{&1.shape, &1.type, &1.names, &1.vectorized_axes})

    NxIREE.Fingerprint.of({expr, templates, flags |> Enum.uniq() |> Enum.sort(), format()})
  end

  @doc false
  def path(dir, key), do: Path.join(dir, key <> ".nxiree")

  @doc false
  def format, do: NxIREE.Exported.format()

  @doc false
  # Returns the cached module with its metadata, or :error
  def read(path) do
    with {:ok, binary} <- File.read(path),
         expected = format(),
         %{format: ^expected} = entry <-
           decode(binary) do
      {:ok,
       {%NxIREE.Module{
          bytecode: entry.bytecode,
          compilation_flags: entry.compilation_flags,
          output_container: entry.output,
          used_inputs: entry.used_inputs,
          written_inputs: entry.written_inputs
        }, entry.callback_pid_size}}
    else
      _ -> :error
    end
  end

  defp decode(binary) do
    :erlang.binary_to_term(binary, [:safe])
  rescue
    ArgumentError -> :error
  end

  @doc false
  def write!(path, %NxIREE.Module{} = module, callback_pid_size) do
    entry = %{
      format: format(),
      bytecode: module.bytecode,
      compilation_flags: module.compilation_flags,
      output: Nx.Defn.Composite.traverse(module.output_container, &Nx.to_template/1),
      used_inputs: module.used_inputs,
      written_inputs: module.written_inputs,
      callback_pid_size: callback_pid_size
    }

    File.mkdir_p!(Path.dirname(path))
    # written aside and renamed, so concurrent readers never see a partial file
    tmp = path <> ".#{System.unique_integer([:positive])}.tmp"
    File.write!(tmp, :erlang.term_to_binary(entry))
    File.rename!(tmp, path)
  end
end
