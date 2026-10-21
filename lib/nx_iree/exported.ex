defmodule NxIREE.Exported do
  @moduledoc """
  A function compiled ahead of time with `NxIREE.export/3` and loaded on a
  device with `NxIREE.load_exported/2`.

  Exported functions run without EXLA or the IREE compiler, which suits
  devices such as Nerves targets: compile on the host, ship the binary,
  load it once on the device and call it with `NxIREE.run/2`.
  """

  defstruct [:module, :device, :inputs, :used_inputs, :leading_inputs]

  @format 2

  @doc false
  def format, do: @format

  @doc false
  # Decodes an export or module cache entry. :safe refuses atoms the VM does
  # not have yet, and on a device that has not loaded Nx's tensor modules the
  # output template's struct atoms are such atoms.
  def decode(binary) do
    Code.ensure_loaded(Nx.Tensor)
    Code.ensure_loaded(Nx.TemplateBackend)
    :erlang.binary_to_term(binary, [:safe])
  end

  @type t :: %__MODULE__{
          module: NxIREE.Module.t(),
          device: NxIREE.Device.t(),
          inputs: [{tuple(), Nx.Type.t()}],
          used_inputs: [non_neg_integer()],
          leading_inputs: [Nx.Tensor.t()]
        }
end
