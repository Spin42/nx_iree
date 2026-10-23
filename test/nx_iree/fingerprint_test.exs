defmodule NxIREE.FingerprintTest do
  use ExUnit.Case, async: true

  test "templates enter fingerprints by shape and type" do
    of = fn template -> NxIREE.Fingerprint.of(%{output: template}) end

    assert of.(Nx.template({4}, :f32)) == of.(Nx.template({4}, :f32))
    refute of.(Nx.template({4}, :f32)) == of.(Nx.template({5}, :f32))
    refute of.(Nx.template({4}, :f32)) == of.(Nx.template({4}, :u32))
  end
end
