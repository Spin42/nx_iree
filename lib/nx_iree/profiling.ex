defmodule NxIREE.Profiling do
  @moduledoc """
  Captures existing IREE dispatch and queue events to an `.ireeprof` file.

  Read the file with IREE's `iree-profile` tool. Capturing adds overhead;
  use an unprofiled run for reported application performance.
  """

  @doc """
  Captures device work issued by `fun`, returning its result.

  The device must be idle before and after the function, and no other caller
  may submit work or change profiling on it during setup or teardown. The
  function must await all work it starts, including tasks, before returning.
  Nested capture is unsupported. Unsupported device capabilities raise.
  The output file is created or overwritten.
  """
  def capture(%NxIREE.Device{} = device, path, fun)
      when is_binary(path) and is_function(fun, 0) do
    if path == "" or String.contains?(path, <<0>>),
      do: raise(ArgumentError, "profiling path must be nonempty without NUL bytes")

    check!(NxIREE.Native.begin_profiling(device.ref, path))

    try do
      fun.()
    after
      check!(NxIREE.Native.end_profiling(device.ref))
    end
  end

  defp check!(:ok), do: :ok
  defp check!({:error, reason}), do: raise("IREE profiling failed: #{reason}")
end
