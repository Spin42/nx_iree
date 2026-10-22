defmodule NxIREE.MixHelpers do
  @moduledoc false

  def link_compiler!(dir, priv_path) do
    dir = Path.expand(dir)

    compiler =
      Enum.find(
        [
          Path.join(dir, "iree-compile"),
          Path.join([dir, "compiler", "_mlir_libs", "iree-compile"])
        ],
        &File.regular?/1
      ) || raise "iree-compile not found in NX_IREE_COMPILER_DIR=#{dir}"

    File.mkdir_p!(priv_path)
    link = Path.join(priv_path, "iree-compile")

    unless File.read_link(link) == {:ok, compiler} do
      case File.rm(link) do
        :ok -> :ok
        {:error, :enoent} -> :ok
        {:error, reason} -> raise File.Error, reason: reason, action: "remove", path: link
      end

      File.ln_s!(compiler, link)
    end

    :ok
  end

  def github_release_path(file, version \\ nil) do
    version = version || File.read!(Path.join(:code.priv_dir(:nx_iree), "VERSION"))

    Path.join(
      "https://github.com/elixir-nx/nx_iree/releases/download/v#{version}/",
      file
    )
  end

  def download!(name, url, dest) do
    case download(name, url, dest) do
      :ok ->
        :ok

      {:error, :not_found} ->
        {:error, :not_found}
    end
  end

  defp download(name, url, dest) do
    {:ok, _} = Application.ensure_all_started(:req)

    Req.new(url: url, into: File.stream!(dest, [:write, :binary]))
    |> Req.Request.prepend_response_steps(
      pre_redirect: &Req.Steps.redirect/1,
      handle_non_200_responses: fn
        {request, %{status: 200} = response} ->
          {request, response}

        {request, response} ->
          {%{request | into: nil}, response}
      end
    )
    |> Req.get()
    |> case do
      {:ok, %{status: 200}} ->
        :ok

      {:ok, %{status: 404}} ->
        {:error, :not_found}

      {:ok, %{status: status}} ->
        raise "unable to download #{name} from #{url}: status #{status}"

      {:error, reason} ->
        raise "unable to download #{name} from #{url}: #{inspect(reason)}"
    end
  end
end
