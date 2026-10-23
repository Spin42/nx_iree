defmodule NxIREE.Fingerprint do
  @moduledoc false

  # A stable identity for a defn expression, so a module compiled on one node
  # can be found on another that builds the same expression. Expression ids
  # and contexts are references, unique per build: they are numbered in
  # traversal order instead. Shared subexpressions are emitted once and then
  # referred to by number. Literal tensors count by their contents and
  # functions by module, name and arity.

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @doc """
  Returns a hex SHA-256 of `term` with references, expressions, tensors and
  functions in canonical form.
  """
  def of(term) do
    {canonical, _state} = canon(term, %{refs: %{}, exprs: %{}})

    canonical
    |> :erlang.term_to_binary([:deterministic, minor_version: 2])
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp canon(%T{data: %Expr{id: id, op: op, args: args, context: context}} = t, state) do
    case state.exprs do
      %{^id => n} ->
        {{:expr_ref, n}, state}

      _ ->
        {args, state} = canon(args, state)
        {context, state} = canon(context, state)
        n = map_size(state.exprs)
        state = put_in(state.exprs[id], n)
        {{:expr, n, op, args, context, t.shape, t.type, t.names, t.vectorized_axes}, state}
    end
  end

  # Templates, such as the output templates of NxIREE.Vulkan.Pipeline stages
  defp canon(%T{data: %Nx.TemplateBackend{}} = t, state) do
    {{:template, t.shape, t.type, t.names, t.vectorized_axes}, state}
  end

  defp canon(%T{} = t, state) do
    digest = :crypto.hash(:sha256, Nx.to_binary(t))
    {{:tensor, t.shape, t.type, t.names, t.vectorized_axes, digest}, state}
  end

  defp canon(ref, state) when is_reference(ref) do
    case state.refs do
      %{^ref => n} ->
        {{:ref, n}, state}

      refs ->
        n = map_size(refs)
        {{:ref, n}, put_in(state.refs[ref], n)}
    end
  end

  defp canon(fun, state) when is_function(fun) do
    info = Function.info(fun)
    {{:fun, info[:module], info[:name], info[:arity]}, state}
  end

  defp canon(pid, state) when is_pid(pid) or is_port(pid), do: {:process, state}

  defp canon(%module{} = struct, state) do
    {fields, state} = canon(Map.from_struct(struct), state)
    {{:struct, module, fields}, state}
  end

  defp canon(map, state) when is_map(map) do
    # Values are visited in key order, so numbering does not depend on the
    # VM's map iteration order
    {keyed, state} =
      Enum.map_reduce(map, state, fn {k, v}, state ->
        {k, state} = canon(k, state)
        {{k, v}, state}
      end)

    {entries, state} =
      keyed
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map_reduce(state, fn {k, v}, state ->
        {v, state} = canon(v, state)
        {{k, v}, state}
      end)

    {{:map, entries}, state}
  end

  defp canon(list, state) when is_list(list) do
    canon_list(list, [], state)
  end

  defp canon(tuple, state) when is_tuple(tuple) do
    {elems, state} = canon(Tuple.to_list(tuple), state)
    {List.to_tuple(elems), state}
  end

  defp canon(other, state), do: {other, state}

  # Handles improper lists such as iodata
  defp canon_list([head | tail], acc, state) do
    {head, state} = canon(head, state)
    canon_list(tail, [head | acc], state)
  end

  defp canon_list([], acc, state), do: {Enum.reverse(acc), state}

  defp canon_list(tail, acc, state) do
    {tail, state} = canon(tail, state)
    {{:improper, Enum.reverse(acc), tail}, state}
  end
end
