defmodule NxIREE.TuningTest do
  use ExUnit.Case, async: true

  defmodule Provider do
    @behaviour NxIREE.Tuning

    @text """
    module attributes {transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint} {
      transform.named_sequence @match(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
        transform.yield %op : !transform.any_op
      }
      transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
        transform.yield
      }
      transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
          attributes {iree_codegen.tuning_spec_entrypoint} {
        %res = transform.foreach_match in %variant_op
          @match -> @apply_op_config
          : (!transform.any_op) -> !transform.any_op
        transform.yield %res : !transform.any_op
      }
    }
    """

    @impl true
    def fetch(:example), do: {:ok, Process.get(:tuning_text, @text), ["vulkan-spirv"]}
    def fetch(_), do: :error
  end

  test "exports retain a content-addressed provider marker instead of a host path" do
    template = Nx.template({4}, :f32)

    exported =
      NxIREE.export(&Nx.add(&1, &1), [template],
        iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv"],
        iree_tuning: {Provider, :example}
      )
      |> :erlang.binary_to_term()

    marker = NxIREE.Tuning.flag({Provider, :example}, "vulkan-spirv")
    assert is_binary(exported.bytecode) and byte_size(exported.bytecode) > 0
    assert marker in exported.compilation_flags
    refute Enum.any?(exported.compilation_flags, &(&1 =~ "transform-dialect-library"))

    ["--iree-codegen-transform-dialect-library=" <> path] = NxIREE.Tuning.resolve([marker])
    {:ok, text, _} = Provider.fetch(:example)
    assert File.read!(path) == text
  end

  test "changing a provider's spec content changes the cache marker" do
    before = NxIREE.Tuning.flag({Provider, :example}, "vulkan-spirv")
    {:ok, text, _} = Provider.fetch(:example)
    Process.put(:tuning_text, text <> "\n")
    after_change = NxIREE.Tuning.flag({Provider, :example}, "vulkan-spirv")
    refute before == after_change

    assert_raise ArgumentError, ~r/changed after selection/, fn ->
      NxIREE.Tuning.resolve([before])
    end
  end

  test "unknown specs, invalid providers, and unsupported backends raise" do
    assert_raise ArgumentError, ~r/unknown :iree_tuning/, fn ->
      NxIREE.Tuning.flag({Provider, :nope}, "vulkan-spirv")
    end

    assert_raise ArgumentError, ~r/must implement NxIREE.Tuning.fetch/, fn ->
      NxIREE.Tuning.flag({String, :example}, "vulkan-spirv")
    end

    assert_raise ArgumentError, ~r/supports .*vulkan-spirv.*got llvm-cpu/, fn ->
      NxIREE.Tuning.flag({Provider, :example}, "llvm-cpu")
    end

    assert_raise ArgumentError, ~r/expected :iree_tuning to be/, fn ->
      NxIREE.Tuning.flag(:example, "vulkan-spirv")
    end
  end
end
