defmodule NxIREE.Tuning do
  @moduledoc """
  Behaviour for libraries that supply IREE transform-dialect tuning specs.

  Pass the provider and spec name to NxIREE.Compiler or NxIREE.export/3:

      iree_tuning: {ProviderModule, :spec_name}

  The provider returns the spec text and supported compiler backends.
  Cache keys include the text's SHA-256 digest, independently of its location
  on the host.
  """

  @callback fetch(atom()) :: {:ok, String.t(), [String.t()]} | :error

  @marker "--nx-iree-tuning="

  @doc false
  def flag({provider, name}, backend) when is_atom(provider) and is_atom(name) do
    {text, backends} = fetch!(provider, name)

    unless backend in backends do
      raise ArgumentError,
            ":iree_tuning #{inspect({provider, name})} supports #{inspect(backends)}, got #{backend}"
    end

    @marker <>
      Enum.join(
        [
          URI.encode_www_form(Atom.to_string(provider)),
          URI.encode_www_form(Atom.to_string(name)),
          digest(text)
        ],
        ":"
      )
  end

  def flag(value, _backend) do
    raise ArgumentError,
          "expected :iree_tuning to be {ProviderModule, :spec_name}, got #{inspect(value)}"
  end

  @doc false
  def resolve(flags) do
    Enum.map(flags, fn
      @marker <> marker ->
        [provider, name, expected_digest] = String.split(marker, ":", parts: 3)
        provider = provider |> URI.decode_www_form() |> String.to_existing_atom()
        name = name |> URI.decode_www_form() |> String.to_existing_atom()
        {text, _backends} = fetch!(provider, name)

        unless digest(text) == expected_digest do
          raise ArgumentError, "tuning spec #{inspect({provider, name})} changed after selection"
        end

        "--iree-codegen-transform-dialect-library=" <> spec_file(text, expected_digest)

      flag ->
        flag
    end)
  end

  defp fetch!(provider, name) do
    unless Code.ensure_loaded?(provider) and function_exported?(provider, :fetch, 1) do
      raise ArgumentError, "#{inspect(provider)} must implement NxIREE.Tuning.fetch/1"
    end

    case provider.fetch(name) do
      {:ok, text, backends} when is_binary(text) and is_list(backends) ->
        {text, backends}

      :error ->
        raise ArgumentError, "unknown :iree_tuning #{inspect({provider, name})}"
    end
  end

  defp digest(text), do: :crypto.hash(:sha256, text) |> Base.encode16(case: :lower)

  defp spec_file(text, digest) do
    path = Path.join(System.tmp_dir!(), "nx_iree-tuning-#{digest}.mlir")

    unless File.exists?(path) do
      tmp = path <> ".#{System.unique_integer([:positive])}"
      File.write!(tmp, text)
      File.rename!(tmp, path)
    end

    path
  end
end
