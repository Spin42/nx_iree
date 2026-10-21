# Writes the StableHLO module EXLA produces for an NxN f32 matmul.
#   N=256 OUT=mm256.mlir mix run dump_mlir.exs
n = String.to_integer(System.get_env("N", "256"))
t = Nx.template({n, n}, :f32)
%{mlir_module: mlir} = EXLA.to_mlir_module(&Nx.dot/2, [t, t])
File.write!(System.get_env("OUT", "mm#{n}.mlir"), mlir)
