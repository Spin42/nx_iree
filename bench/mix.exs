defmodule NxIREEBench.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_iree_bench,
      version: "0.1.0",
      elixir: "~> 1.15",
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx_iree, path: "..", override: true},
      {:c100_gpu, github: "mlainez/c100_gpu"},
      {:exla, "~> 1.0"},
      {:nx, "~> 1.0"},
      {:axon, "~> 0.9"},
      {:bumblebee, "~> 0.8"}
    ]
  end
end
