if root = System.get_env("EVISION_RUNTIME_ROOT") do
  for app <- ~w(complex telemetry nx castore evision),
      do: Code.append_path(Path.join([root, "lib", app, "ebin"]))
end

{:ok, _} = Application.ensure_all_started(:evision)

IO.inspect(Evision.OCL.haveOpenCL(), label: "EVISION_HAVE_OPENCL")
IO.inspect(Evision.OCL.useOpenCL(), label: "EVISION_USE_OPENCL")
mat = Evision.Mat.from_binary(:binary.copy(<<10, 20, 30>>, 16), :u8, 4, 4, 3)
umat = Evision.UMat.uMat(mat)
IO.inspect(umat, label: "EVISION_UMAT_CONSTRUCTION")
IO.inspect(Evision.cvtColor(mat, 6), label: "EVISION_MAT_CVTCOLOR")

try do
  IO.inspect(Evision.cvtColor(umat, 6), label: "EVISION_UMAT_POSITIONAL_CVTCOLOR")
rescue
  e in FunctionClauseError ->
    IO.puts("EVISION_UMAT_POSITIONAL_CVTCOLOR=" <> Exception.message(e))
end

IO.inspect(Evision.cvtColor(src: umat, code: 6), label: "EVISION_UMAT_KEYWORD_CVTCOLOR")
IO.puts("EVISION_ACCELERATION_PROBE_PASS")
