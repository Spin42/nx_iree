defmodule NxIREE.Compiler.HostCalls do
  @moduledoc false

  # IREE modules cannot call back into the BEAM, so expressions are checked
  # for host calls before lowering. A named io_call with no callback and no
  # hook registered for it does nothing and is removed (Bumblebee's
  # generation loop reports each token through one, for streaming); io_calls
  # and runtime_calls that would run Elixir code raise.

  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  @doc """
  Returns `container` without its no-op io_calls, raising on host calls that
  would have an effect. `hooks` are the jit's `:hooks`.
  """
  def strip!(container, hooks) do
    {container, _cache} = Composite.traverse(container, %{}, &rewrite(&1, &2, hooks))
    container
  end

  defp rewrite(%T{data: %Expr{id: id, op: op}} = t, cache, hooks) do
    case cache do
      %{^id => result} ->
        {result, cache}

      _ ->
        {result, cache} = rewrite(op, t, cache, hooks)
        {result, Map.put(cache, id, result)}
    end
  end

  defp rewrite(other, cache, _hooks), do: {other, cache}

  defp rewrite(:io_call, %T{data: %Expr{args: [data, spec, template, _ref]}}, cache, hooks) do
    ensure_no_op!(spec, hooks)
    {data, cache} = Composite.traverse(data, cache, &rewrite(&1, &2, hooks))

    case template do
      %T{} -> {data, cache}
      # a container io_call is read through :elem nodes, which pick from these
      _ -> {{:removed_io_call, Composite.flatten_list([data])}, cache}
    end
  end

  defp rewrite(:elem, %T{data: %Expr{args: [root, index]} = expr} = t, cache, hooks) do
    case rewrite(root, cache, hooks) do
      {{:removed_io_call, leaves}, cache} -> {Enum.at(leaves, index), cache}
      {root, cache} -> {%{t | data: %{expr | args: [root, index]}}, cache}
    end
  end

  defp rewrite(:runtime_call, %T{data: %Expr{args: [_, fun | _]}}, _cache, _hooks) do
    raise ArgumentError,
          "Nx.runtime_call/4 (#{inspect(fun)}) is not supported by NxIREE: " <>
            "IREE modules cannot call back into Elixir"
  end

  defp rewrite(_op, %T{data: expr} = t, cache, hooks) do
    {args, cache} = Tree.apply_args(t, :all, cache, &rewrite(&1, &2, hooks))
    {%{t | data: %{expr | args: args}}, cache}
  end

  defp ensure_no_op!({:named, name, callback}, hooks) do
    if callback || Map.get(hooks, name) do
      raise ArgumentError,
            "the #{inspect(name)} hook is not supported by NxIREE: IREE modules cannot " <>
              "call back into Elixir (for Bumblebee, do not enable streaming)"
    end
  end

  defp ensure_no_op!({:fn, fun}, _hooks) do
    raise ArgumentError,
          "Nx.io_call/2 with #{inspect(fun)} (print_value, print_expr and the like) is not " <>
            "supported by NxIREE: IREE modules cannot call back into Elixir"
  end
end
