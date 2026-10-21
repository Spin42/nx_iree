# Bisects BERT's attention-weights graph on Vulkan vs EXLA: the mask chain
# Bumblebee builds (key_mask and causal_mask -> select -> add -> softmax),
# softmax as a nested defn call, and Bumblebee's own attention layer.
Logger.configure(level: :warning)

defmodule Bisect do
  import Nx.Defn

  defn softmax_nested(x), do: Axon.Activations.softmax(x, axis: -1)

  defn bias_and(m) do
    mask = Nx.logical_and(m |> Nx.new_axis(1) |> Nx.new_axis(1), Nx.broadcast(1, {1, 1}) |> Nx.new_axis(0) |> Nx.new_axis(0))
    Nx.select(mask, Nx.tensor(0.0, type: :f32), Nx.Constants.min_finite(:f32))
  end

  defn bias_plain(m) do
    Nx.select(m |> Nx.new_axis(1) |> Nx.new_axis(1), Nx.tensor(0.0, type: :f32), Nx.Constants.min_finite(:f32))
  end
end

{:ok, vk} = NxIREE.list_devices("vulkan")
gpu = Enum.find(vk, &String.contains?(&1.uri, System.get_env("NXIREE_DEVICE_MATCH", "6d657361")))
flags = ["--iree-hal-target-backends=vulkan-spirv", "--iree-input-type=stablehlo_xla"] ++ String.split(System.get_env("NXIREE_FLAGS", ""), " ", trim: true)
opts = [compiler: NxIREE.Compiler, iree_compiler_flags: flags, iree_runtime_options: [device: gpu]]

key = Nx.Random.key(5)
{s, _} = Nx.Random.normal(key, 0.0, 0.05, shape: {1, 4, 8, 8})
m = Nx.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])

check = fn name, fun, args ->
  want = Nx.Defn.jit(fun, compiler: EXLA).(args) |> Nx.backend_transfer(Nx.BinaryBackend)
  got = Nx.Defn.jit(fun, opts).(args) |> Nx.backend_transfer(Nx.BinaryBackend)
  want_f = Nx.as_type(want, :f32)
  got_f = Nx.as_type(got, :f32)
  finite = Nx.abs(want_f) |> Nx.less(1.0e30)
  err = Nx.select(finite, Nx.abs(Nx.subtract(got_f, want_f)), 0.0) |> Nx.reduce_max() |> Nx.to_number()
  bad_inf = Nx.select(finite, 0, Nx.not_equal(got_f, want_f)) |> Nx.reduce_max() |> Nx.to_number()
  status = if err < 1.0e-5 and bad_inf == 0, do: "ok  ", else: "BAD "
  row = fn t -> t |> Nx.flatten() |> Nx.slice([0], [8]) |> Nx.to_flat_list() |> Enum.map(&(if is_float(&1), do: Float.round(&1, 4), else: &1)) end
  IO.puts("#{status} #{String.pad_trailing(name, 34)} max err #{:erlang.float_to_binary(err * 1.0, decimals: 6)}  got #{inspect(row.(got))}  want #{inspect(row.(want))}")
end

check.("and chain (u8 output)", fn {m} -> Nx.logical_and(m, Nx.broadcast(1, {1, 1})) end, {m})
check.("bias plain select", fn {m} -> Bisect.bias_plain(m) end, {m})
check.("bias via and", fn {m} -> Bisect.bias_and(m) end, {m})
check.("plain bias + add", fn {s, m} -> Nx.add(s, Bisect.bias_plain(m)) end, {s, m})
check.("and bias + add", fn {s, m} -> Nx.add(s, Bisect.bias_and(m)) end, {s, m})
check.("softmax(s) inline", fn {s} -> Axon.Activations.softmax(s, axis: -1) end, {s})
check.("softmax(s) nested", fn {s} -> Bisect.softmax_nested(s) end, {s})
check.("plain bias, softmax inline", fn {s, m} -> Axon.Activations.softmax(Nx.add(s, Bisect.bias_plain(m)), axis: -1) end, {s, m})
check.("plain bias, softmax nested", fn {s, m} -> Bisect.softmax_nested(Nx.add(s, Bisect.bias_plain(m))) end, {s, m})
check.("and bias, softmax inline", fn {s, m} -> Axon.Activations.softmax(Nx.add(s, Bisect.bias_and(m)), axis: -1) end, {s, m})
check.("and bias, softmax nested", fn {s, m} -> Bisect.softmax_nested(Nx.add(s, Bisect.bias_and(m))) end, {s, m})

# The form IREE gives BERT's attention weights: the mask reaches the scores
# dispatch as a separate u8 tensor in memory and is fused into a batch matmul.
{q, key} = Nx.Random.normal(key, shape: {1, 4, 8, 8})
{k, key} = Nx.Random.normal(key, shape: {1, 4, 8, 8})
m8 = Nx.as_type(m, :u8)
bmm_bias = fn q, k, m8 ->
  scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(1 / :math.sqrt(8))
  Nx.add(scores, Nx.select(Nx.not_equal(m8, 0), 0.0, Nx.Constants.min_finite(:f32)) |> Nx.reshape({1, 1, 1, 8}))
end
check.("u8 mask input + batch matmul", fn {q, k, m8} -> bmm_bias.(q, k, m8) end, {q, k, m8})
check.("s32 mask input + batch matmul", fn {q, k, m} -> bmm_bias.(q, k, m) end, {q, k, m})

# Bumblebee's attention layer itself, with q/k/v as plain inputs.
q = Axon.input("q", shape: {1, 8, 4, 8})
k = Axon.input("k", shape: {1, 8, 4, 8})
v = Axon.input("v", shape: {1, 8, 4, 8})
km = Axon.input("mask", shape: {1, 8})
none = Bumblebee.Layers.none()
{_out, weights} = Bumblebee.Layers.attention(q, k, v, km, none, none, none)
{_, predict} = Axon.build(weights)
{qt, key} = Nx.Random.normal(key, shape: {1, 8, 4, 8})
{kt, key} = Nx.Random.normal(key, shape: {1, 8, 4, 8})
{vt, _} = Nx.Random.normal(key, shape: {1, 8, 4, 8})
inputs = %{"q" => qt, "k" => kt, "v" => vt, "mask" => m}
check.("Bumblebee.Layers.attention", fn {inputs} -> predict.(%{}, inputs) end, {inputs})
