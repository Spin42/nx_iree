defmodule NxIREE.Tuning do
  @moduledoc false

  # Tuning specs selected by name with the compiler's :iree_tuning option.
  # The flags carry a marker naming the spec, so module cache keys do not
  # depend on where the spec file lives; the marker becomes IREE's
  # transform-dialect library flag only for the compiler invocation.
  #
  #   * :maxwell_matmul - 64x64x8 workgroup tiles, 16x8 threads, no
  #     multi-buffering for M x N x K matmuls on SPIR-V. On the Jetson Nano's
  #     GM20B it takes a 1024 matmul from 34-42 to 99 GFLOPS and DistilBERT
  #     from 346 to 145 ms per batch of 4.

  @specs %{maxwell_matmul: "maxwell_matmul.mlir"}

  for {_name, file} <- @specs, do: @external_resource(Path.join([__DIR__, "tuning", file]))

  @texts Map.new(@specs, fn {name, file} ->
           {name, File.read!(Path.join([__DIR__, "tuning", file]))}
         end)

  @marker "--nx-iree-tuning="

  @doc """
  The flag that selects `name`, raising for unknown names or backends.
  """
  def flag(name, backend) do
    unless Map.has_key?(@texts, name) do
      raise ArgumentError,
            "unknown :iree_tuning #{inspect(name)}, expected one of #{inspect(Map.keys(@texts))}"
    end

    unless backend == "vulkan-spirv" do
      raise ArgumentError, ":iree_tuning applies to the vulkan-spirv backend, got #{backend}"
    end

    @marker <> Atom.to_string(name)
  end

  @doc """
  Replaces tuning markers in `flags` with IREE's flag for the spec's file.
  """
  def resolve(flags) do
    Enum.map(flags, fn
      @marker <> name ->
        "--iree-codegen-transform-dialect-library=" <> spec_file(String.to_existing_atom(name))

      flag ->
        flag
    end)
  end

  defp spec_file(name) do
    text = Map.fetch!(@texts, name)
    digest = :crypto.hash(:sha256, text) |> Base.encode16(case: :lower) |> binary_part(0, 16)
    path = Path.join(System.tmp_dir!(), "nx_iree-#{name}-#{digest}.mlir")

    unless File.exists?(path) do
      tmp = path <> ".#{System.unique_integer([:positive])}"
      File.write!(tmp, text)
      File.rename!(tmp, path)
    end

    path
  end
end
