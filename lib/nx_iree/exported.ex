defmodule NxIREE.Exported do
  @moduledoc """
  A function compiled ahead of time with `NxIREE.export/3` and loaded on a
  device with `NxIREE.load_exported/2`.

  Exported functions run without EXLA or the IREE compiler, which suits
  devices such as Nerves targets: compile on the host, ship the binary,
  load it once on the device and call it with `NxIREE.run/2`.
  """

  defstruct [:module, :device, :inputs, :used_inputs, :leading_inputs]

  @format 1

  @doc false
  def format, do: @format

  @type t :: %__MODULE__{
          module: NxIREE.Module.t(),
          device: NxIREE.Device.t(),
          inputs: [{tuple(), Nx.Type.t()}],
          used_inputs: [non_neg_integer()],
          leading_inputs: [Nx.Tensor.t()]
        }
end
