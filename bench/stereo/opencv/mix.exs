defmodule OpenCVStereoBench.MixProject do
  use Mix.Project

  def project do
    [app: :opencv_stereo_bench, version: "0.1.0", deps: [{:evision, "== 1.0.0"}]]
  end

  def application, do: [extra_applications: [:logger]]
end
