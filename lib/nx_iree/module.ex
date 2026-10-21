defmodule NxIREE.Module do
  @doc """
  Holds the bytecode and other metadata for a compiled MLIR module.
  """

  defstruct [
    :bytecode,
    :compilation_flags,
    :mlir_module,
    :output_container,
    :used_inputs,
    :written_inputs,
    :loaded
  ]

  @type t :: %__MODULE__{
          bytecode: String.t(),
          compilation_flags: list(String.t()),
          mlir_module: String.t(),
          output_container: term(),
          used_inputs: nil | [non_neg_integer()],
          written_inputs: :all | [non_neg_integer()],
          loaded: nil | {device_uri :: String.t(), reference()}
        }
end
