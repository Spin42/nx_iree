defmodule NxIREE.CompileError do
  @moduledoc """
  Raised when `iree-compile` rejects a module or is not available.
  """
  defexception [:message]
end
