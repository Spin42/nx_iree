defmodule NxIREE.Compiler.DeviceIndexing do
  @moduledoc false

  # IREE lowers slices and slice updates whose start indices are tensors to
  # host-side offsets: the host waits for the device and reads the indices
  # back before it can go on. In loops this costs a round trip per iteration
  # (text generation writes each token into the sequence and the KV cache at
  # a computed position). For GPU targets those ops are rewritten so the
  # indices stay on the device: a dynamic slice becomes a gather of the
  # clamped window, a dynamic slice update a select between the target and
  # the update gathered into place. Start indices are clamped as Nx does, to
  # [0, dim - window].

  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  @doc """
  Returns `container` with dynamic slices and slice updates rewritten.
  """
  def rewrite(container) do
    # New nodes are built with Nx functions; constants such as clip bounds
    # must become expressions rather than tensors of the global backend
    Nx.with_default_backend(Expr, fn ->
      {container, _cache} = Composite.traverse(container, %{}, &rewrite/2)
      container
    end)
  end

  defp rewrite(%T{data: %Expr{id: id}} = t, cache) do
    case cache do
      %{^id => result} ->
        {result, cache}

      _ ->
        {args, cache} = Tree.apply_args(t, :all, cache, &rewrite/2)
        t = %{t | data: %{t.data | args: args}}
        result = if dynamic?(t), do: device_side(t), else: t
        {result, Map.put(cache, id, result)}
    end
  end

  defp rewrite(other, cache), do: {other, cache}

  defp dynamic?(%T{data: %Expr{op: :slice, args: [_, starts, _, _]}}),
    do: tensor_start?(starts)

  defp dynamic?(%T{data: %Expr{op: :put_slice, args: [_, starts, _]}}),
    do: tensor_start?(starts)

  defp dynamic?(_), do: false

  defp tensor_start?(starts), do: Enum.any?(starts, &(not is_integer(&1)))

  defp device_side(%T{data: %Expr{op: :slice, args: [source, starts, lengths, strides]}} = out) do
    axes = Enum.zip([starts, lengths, strides, Tuple.to_list(source.shape)])

    # Dynamic axes are gathered to their window, then the static ones sliced
    gathered =
      axes
      |> Enum.with_index()
      |> Enum.reduce(source, fn
        {{start, _length, _stride, _dim}, _axis}, tensor when is_integer(start) ->
          tensor

        {{start, length, stride, dim}, axis}, tensor ->
          count = div(length - 1, stride) + 1
          indices = Nx.add(clamp(start, dim - length), Nx.multiply(iota(count), stride))
          Nx.take(tensor, indices, axis: axis)
      end)

    {static_starts, static_lengths, static_strides} =
      axes
      |> Enum.zip(Tuple.to_list(gathered.shape))
      |> Enum.map(fn
        {{start, length, stride, _dim}, _size} when is_integer(start) -> {start, length, stride}
        {_axis, size} -> {0, size, 1}
      end)
      |> unzip3()

    gathered
    |> Nx.slice(static_starts, static_lengths, strides: static_strides)
    |> Nx.as_type(out.type)
    |> Nx.rename(out.names)
  end

  defp device_side(%T{data: %Expr{op: :put_slice, args: [tensor, starts, update]}} = out) do
    partial =
      for {{dim, size}, axis} <-
            Enum.with_index(Enum.zip(Tuple.to_list(tensor.shape), Tuple.to_list(update.shape))),
          dim != size,
          do: axis

    case partial do
      [axis] -> scatter_rows(out, tensor, Enum.at(starts, axis), update, axis)
      _ -> select_window(out, tensor, starts, update)
    end
  end

  # The update spans every axis but one: scatter its rows at positions
  # computed on the device
  defp scatter_rows(out, tensor, start, update, axis) do
    dim = elem(tensor.shape, axis)
    size = elem(update.shape, axis)
    rank = tuple_size(tensor.shape)

    positions = Nx.add(clamp(start, dim - size), iota(size)) |> Nx.reshape({size, 1})
    rows = Nx.transpose(update, axes: [axis | Enum.to_list(0..(rank - 1)) -- [axis]])

    tensor
    |> Nx.as_type(out.type)
    |> Nx.indexed_put(positions, Nx.as_type(rows, out.type), axes: [axis])
    |> Nx.rename(out.names)
  end

  defp select_window(out, tensor, starts, update) do
    rank = tuple_size(tensor.shape)

    {update, mask} =
      [starts, Tuple.to_list(tensor.shape), Tuple.to_list(update.shape)]
      |> Enum.zip()
      |> Enum.with_index()
      |> Enum.reduce({update, nil}, fn
        {{_start, dim, dim}, _axis}, acc ->
          acc

        {{start, dim, size}, axis}, {update, mask} ->
          position = Nx.subtract(iota(dim), clamp(start, dim - size))
          inside = Nx.logical_and(Nx.greater_equal(position, 0), Nx.less(position, size))
          update = Nx.take(update, Nx.clip(position, 0, size - 1), axis: axis)

          shape = Tuple.duplicate(1, rank) |> put_elem(axis, dim)
          inside = Nx.reshape(inside, shape)
          mask = if mask, do: Nx.logical_and(mask, inside), else: inside
          {update, mask}
      end)

    tensor = Nx.as_type(tensor, out.type)
    update = Nx.as_type(update, out.type)

    result =
      if mask, do: Nx.select(Nx.broadcast(mask, tensor.shape), update, tensor), else: update

    Nx.rename(result, out.names)
  end

  defp clamp(start, max_start) when is_integer(start), do: start |> max(0) |> min(max_start)
  defp clamp(start, max_start), do: Nx.clip(start, 0, max_start)

  defp iota(n), do: Nx.iota({n})

  defp unzip3(triples) do
    {Enum.map(triples, &elem(&1, 0)), Enum.map(triples, &elem(&1, 1)),
     Enum.map(triples, &elem(&1, 2))}
  end
end
