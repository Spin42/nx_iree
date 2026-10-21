# Force NxIREE.MixHelpers to be available before the app is compiled
Code.eval_file("./lib/nx_iree/mix_helpers.exs")

defmodule NxIREE.MixProject do
  use Mix.Project

  @version File.read!(Path.join([__DIR__, "priv", "VERSION"]))
  @source_url "https://github.com/elixir-nx/nx_iree"

  import NxIREE.MixHelpers, only: [download!: 3, github_release_path: 2]

  def cli do
    [preferred_envs: [docs: :docs, "hex.publish": :docs]]
  end

  def project do
    n_jobs = to_string(max(System.schedulers_online() - 2, 1))

    [
      app: :nx_iree,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      name: "NxIREE",
      description: "IREE compiler and runtime facilities for Nx",
      package: package(),
      compilers: [:nx_iree, :elixir_make] ++ Mix.compilers(),
      aliases: [
        "compile.nx_iree": &compile/1,
        "iree.version": &version/1
      ],
      make_env: fn ->
        priv_path = Path.join(Mix.Project.app_path(), "priv")
        cwd_relative_to_priv = relative_to(__DIR__, priv_path)

        %{
          "MIX_BUILD_EMBEDDED" => "#{Mix.Project.config()[:build_embedded]}",
          "CWD_RELATIVE_TO_PRIV_PATH" => cwd_relative_to_priv,
          "MAKE_NUM_JOBS" => n_jobs,
          "IREE_GIT_REV" => nx_iree_config().tag,
          "NX_IREE_SOURCE_DIR" => nx_iree_config().source_dir,
          "NX_IREE_CACHE_SO" => nx_iree_config().nx_iree_so_path,
          "NX_IREE_PREFER_PRECOMPILED" => to_string(nx_iree_config().use_precompiled)
        }
      end
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {NxIREE.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:elixir_make, "~> 0.6", runtime: false},
      {:exla, "~> 1.0", optional: true},
      {:nx, "~> 1.0"},
      {:ex_doc, "~> 0.34", only: :docs, runtime: false},
      {:req, "~> 0.5", runtime: false}
    ]
  end

  defp package do
    [
      maintainers: ["Paulo Valente"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url},
      files: [
        "lib",
        "priv",
        "mix.exs",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "cmake",
        "c_src",
        "Makefile"
      ]
    ]
  end

  defp docs do
    [
      main: "NxIREE",
      source_url_pattern: "#{@source_url}/blob/v#{@version}/nx_iree/%{path}#L%{line}",
      extras: [
        "CHANGELOG.md"
      ],
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"]
    ]
  end

  defp version(_args) do
    IO.puts(nx_iree_config().tag)
  end

  defp compile(args) do
    ensure_build_priv!()

    # Devices only run precompiled modules; iree-compile runs on the host
    unless cross_compiling?() do
      :ok = download_and_unzip_iree_release(args)
    end

    if nx_iree_config().use_precompiled and not File.exists?(nx_iree_config().nx_iree_so_path) do
      case download_precompiled_nx_iree_lib() do
        :ok ->
          :ok

        {:error, :not_found} ->
          System.put_env("NX_IREE_PREFER_PRECOMPILED", "false")
          :ok

        {:error, reason} ->
          raise "unable to download nx_iree: #{inspect(reason)}"
      end
    else
      :ok
    end
  end

  defp nx_iree_config() do
    version = System.get_env("NX_IREE_VERSION", "3.13.0")
    tag = System.get_env("NX_IREE_GIT_REV", "v3.13.0")

    env_dir = System.get_env("NX_IREE_COMPILER_DIR")

    home_cache = Path.join(System.fetch_env!("HOME"), ".cache")

    source_env_dir =
      System.get_env("NX_IREE_SOURCE_DIR", Path.join([home_cache, "nx_iree", "iree-#{tag}"]))

    dir = env_dir || Path.join(__DIR__, "cache/iree")
    source_dir = source_env_dir || Path.join(__DIR__, "cache/iree-source")

    use_precompiled = System.get_env("NX_IREE_PREFER_PRECOMPILED", "false") in ["1", "true"]

    # Matches NX_IREE_CACHE_DIR in the Makefile
    cache =
      if cross_compiling?(),
        do: Path.join([__DIR__, "cache", Path.basename(System.get_env("CROSSCOMPILE"))]),
        else: Path.join(__DIR__, "cache")

    %{
      version: version,
      tag: tag,
      base: "iree",
      env_dir: env_dir,
      dir: dir,
      source_dir: source_dir,
      use_precompiled: use_precompiled,
      nx_iree_so_path: Path.join(cache, "libnx_iree.so"),
      nx_iree_tar_gz_path: Path.join(cache, "libnx_iree.tar.gz")
    }
  end

  # Nerves sets CROSSCOMPILE when building for a device
  defp cross_compiling?, do: System.get_env("CROSSCOMPILE", "") != ""

  # Mix links the build's priv to the source priv (for this project and for
  # path dependencies). Build products are written into the build's priv, so
  # replace that link with a directory of its own: otherwise host and device
  # builds overwrite each other's files through the shared source directory.
  # The Makefile does the same before installing the NIF.
  defp ensure_build_priv! do
    priv = Path.join(Mix.Project.app_path(), "priv")

    case File.read_link(priv) do
      {:ok, _} ->
        source = Path.expand("priv", __DIR__)
        File.rm!(priv)
        File.mkdir_p!(priv)
        File.cp!(Path.join(source, "VERSION"), Path.join(priv, "VERSION"))

      {:error, _} ->
        File.mkdir_p!(priv)
    end
  end

  defp download_and_unzip_iree_release(args) do
    nx_iree_config = nx_iree_config()

    cache_dir =
      if dir = System.get_env("NX_IREE_CACHE") do
        Path.expand(dir)
      else
        :filename.basedir(:user_cache, "iree")
      end

    if "--force" in args do
      File.rm_rf(nx_iree_config.dir)
      File.rm_rf(nx_iree_config.source_dir)
      File.rm_rf(cache_dir)
    end

    priv_path = Path.join(Mix.Project.app_path(), "priv")

    if File.dir?(nx_iree_config.dir) and File.exists?(Path.join(priv_path, "iree-compile")) do
      :ok
    else
      download_and_unzip_iree_release(cache_dir, nx_iree_config)
    end
  end

  defp download_and_unzip_iree_release(cache_dir, nx_iree_config) do
    File.mkdir_p!(cache_dir)

    nx_iree_zip =
      Path.join(cache_dir, "iree-compiler-#{nx_iree_config.version}.zip")

    unless File.exists?(nx_iree_zip) do
      # Download iree release for the compiler
      os = :os.type()

      arch =
        case List.to_string(:erlang.system_info(:system_architecture)) do
          "x86_64" <> _ -> "x86_64"
          "aarch64" <> _ -> "aarch64"
          "amd64" <> _ -> "amd64"
        end

      url =
        case {os, arch} do
          {{:unix, :linux}, arch} ->
            "https://github.com/iree-org/iree/releases/download/#{nx_iree_config.tag}/iree_base_compiler-#{nx_iree_config.version}-cp312-abi3-manylinux_2_27_#{arch}.manylinux_2_28_#{arch}.whl"

          {{:unix, :darwin}, _} ->
            # MacOS
            "https://github.com/iree-org/iree/releases/download/#{nx_iree_config.tag}/iree_base_compiler-#{nx_iree_config.version}-cp312-abi3-macosx_13_0_universal2.whl"

          os ->
            Mix.raise("OS #{inspect(os)} is not supported")
        end

      download!("IREE", url, nx_iree_zip)
    end

    # Unpack iree and move to the target cache dir
    parent_iree_dir = Path.dirname(nx_iree_config.dir)
    File.mkdir_p!(parent_iree_dir)

    # Extract to the parent directory (it will be inside the iree directory)
    {:ok, _} =
      nx_iree_zip
      |> String.to_charlist()
      |> :zip.unzip(cwd: String.to_charlist(parent_iree_dir))

    # Remove stray files

    File.rm_rf(Path.join(parent_iree_dir, "iree_base_compiler.egg-info"))

    File.rm_rf(
      Path.join(parent_iree_dir, "iree_base_compiler-#{nx_iree_config.version}.dist-info")
    )

    iree_compile_path =
      Path.join([parent_iree_dir, "iree", "compiler", "_mlir_libs", "iree-compile"])

    iree_lld_path =
      Path.join([parent_iree_dir, "iree", "compiler", "_mlir_libs", "iree-lld"])

    File.chmod!(iree_compile_path, 0o755)
    File.chmod!(iree_lld_path, 0o755)

    priv_path = Path.join(Mix.Project.app_path(), "priv")
    File.mkdir_p!(priv_path)

    link_name = Path.join(priv_path, "iree-compile")

    File.rm(link_name)
    File.ln_s!(iree_compile_path, link_name)

    :ok
  end

  defp download_precompiled_nx_iree_lib() do
    nx_iree_config = nx_iree_config()

    arch =
      case to_string(:erlang.system_info(:system_architecture)) do
        "x86_64" <> _ -> "x86_64"
        "aarch64" <> _ -> "aarch64"
        _ -> Mix.raise("Unsupported architecture")
      end

    nif_version = :erlang.system_info(:nif_version)

    os_name =
      case :os.type() do
        {:unix, :darwin} ->
          "macos"

        {:unix, :linux} ->
          "linux"

        os ->
          Mix.raise("OS #{inspect(os)} is not supported")
      end

    # This is the precompiled path, which should match what's included in releases
    # by the github actions workflows
    version_path = "libnx_iree-#{os_name}-#{arch}-nif-#{nif_version}"
    source_tar_path = "#{version_path}.tar.gz"
    zip_name = nx_iree_config.nx_iree_tar_gz_path

    result =
      download!(
        "NxIREE NIFs",
        github_release_path(source_tar_path, @version),
        zip_name
      )

    case result do
      {:error, reason} ->
        {:error, reason}

      :ok ->
        parent_dir = Path.dirname(zip_name)

        :ok =
          zip_name
          |> String.to_charlist()
          |> :erl_tar.extract([:compressed, cwd: String.to_charlist(parent_dir)])

        File.rename(
          Path.join([parent_dir, version_path, "libnx_iree.so"]),
          Path.join(parent_dir, "libnx_iree.so")
        )

        File.rename(
          Path.join([parent_dir, version_path, "iree-runtime"]),
          Path.join(parent_dir, "iree-runtime")
        )

        File.rmdir(Path.join(parent_dir, version_path))

        :ok
    end
  end

  # Returns `path` relative to the `from` directory.
  defp relative_to(path, from) do
    path_parts = path |> Path.expand() |> Path.split()
    from_parts = from |> Path.expand() |> Path.split()
    {path_parts, from_parts} = drop_common_prefix(path_parts, from_parts)
    root_relative = for _ <- from_parts, do: ".."
    Path.join(root_relative ++ path_parts)
  end

  defp drop_common_prefix([h | left], [h | right]), do: drop_common_prefix(left, right)
  defp drop_common_prefix(left, right), do: {left, right}
end
