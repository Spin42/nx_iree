defmodule NxIREE.ProfilingTest do
  use ExUnit.Case

  test "rejects invalid paths before touching the device" do
    for path <- ["", "bad\0path"] do
      assert_raise ArgumentError, ~r/profiling path/, fn ->
        NxIREE.Profiling.capture(%NxIREE.Device{}, path, fn -> flunk("must not run") end)
      end
    end
  end

  @tag :vulkan
  test "capture returns the result and closes the session after an exception" do
    {:ok, devices} = NxIREE.list_devices("vulkan")
    match = System.get_env("NX_IREE_DEFAULT_DEVICE", "")
    device = Enum.find(devices, &String.contains?(&1.uri, match))
    assert device

    path =
      Path.join(
        System.tmp_dir!(),
        "nx-iree-profile-#{System.unique_integer([:positive])}.ireeprof"
      )

    on_exit(fn -> File.rm(path) end)

    assert_raise RuntimeError, "capture body", fn ->
      NxIREE.Profiling.capture(device, path, fn -> raise "capture body" end)
    end

    assert NxIREE.Profiling.capture(device, path, fn -> :result end) == :result
    assert <<"IRPF", _::binary>> = File.read!(path)
  end
end
