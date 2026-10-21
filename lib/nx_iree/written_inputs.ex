defmodule NxIREE.WrittenInputs do
  @moduledoc false

  # IREE lets a dispatch write its result into the storage of an imported
  # argument (sort, scatter and in-place updates do), so a caller's buffer can
  # change under it. This reads the program as printed after IREE's stream
  # phase and returns the indices of the public function's arguments whose
  # buffers may be written. Each argument's imported resource is followed
  # through awaits, branches and execute regions; any use that is not known to
  # only read it counts as a write.

  @doc """
  Returns `{:ok, indices}` with the sorted argument indices of `function`
  that may be written in `stream_ir`, or `:unknown` when the function cannot
  be found.
  """
  def analyze(stream_ir, function \\ "main") do
    lines = String.split(stream_ir, "\n")

    with {:ok, body} <- function_body(lines, function) do
      written =
        for {resource, index} <- imports(body),
            written?(body, resource),
            uniq: true,
            do: index

      {:ok, Enum.sort(written)}
    end
  end

  defp function_body(lines, function) do
    header = "util.func public @#{function}("

    case Enum.split_while(lines, &(not String.contains?(&1, header))) do
      {_, [_signature | rest]} ->
        {:ok, Enum.take_while(rest, &(&1 != "  }"))}

      {_, []} ->
        :unknown
    end
  end

  defp imports(body) do
    for line <- body,
        [_, resource, index] <-
          [Regex.run(~r/^\s*(%[\w]+) = stream\.tensor\.import\b.*\s%arg(\d+) : !hal\./, line)] do
      {resource, String.to_integer(index)}
    end
  end

  defp written?(body, resource) do
    indexed = Enum.with_index(body)
    walk(indexed, [resource], MapSet.new())
  end

  defp walk(_indexed, [], _seen), do: false

  defp walk(indexed, [name | rest], seen) do
    if MapSet.member?(seen, name) do
      walk(indexed, rest, seen)
    else
      seen = MapSet.put(seen, name)
      token = token_regex(name)

      result =
        Enum.reduce_while(indexed, {:ok, []}, fn {line, i}, {:ok, aliases} ->
          if Regex.match?(token, line) and not definition?(line, name) do
            case classify(line, name, i, indexed) do
              :read -> {:cont, {:ok, aliases}}
              {:alias, more} -> {:cont, {:ok, more ++ aliases}}
              :write -> {:halt, :write}
            end
          else
            {:cont, {:ok, aliases}}
          end
        end)

      case result do
        :write -> true
        {:ok, aliases} -> walk(indexed, aliases ++ rest, seen)
      end
    end
  end

  # `%3` must not match `%30` or `%3#0`; `%3#0` must not match `%3#01`
  defp token_regex(name) do
    if String.contains?(name, "#"),
      do: ~r/#{Regex.escape(name)}(?!\d)/,
      else: ~r/#{Regex.escape(name)}(?![\w#])/
  end

  defp definition?(line, name) do
    base = name |> String.split("#") |> hd()
    trimmed = String.trim_leading(line)

    String.starts_with?(trimmed, "^") or
      Regex.match?(~r/^#{Regex.escape(base)}(:\d+)?(, %[\w]+)* = /, trimmed)
  end

  defp classify(line, name, i, indexed) do
    escaped = Regex.escape(name)

    cond do
      String.contains?(line, "stream.cmd.execute") ->
        case Regex.run(~r/#{escaped} as (%[\w]+)/, line) do
          [_, inner] -> if region_writes?(indexed, i, line, inner), do: :write, else: :read
          nil -> :write
        end

      match =
          Regex.run(~r/^\s*(%[\w]+)(?::(\d+))? = stream\.timepoint\.await .* => (.*) : /, line) ->
        [_, result, count, operands] = match
        operands = String.split(operands, ", ")

        aliases =
          for {operand, position} <- Enum.with_index(operands), operand == name do
            if count == "", do: result, else: "#{result}##{position}"
          end

        {:alias, aliases}

      match =
          Regex.run(
            ~r/^\s*(%[\w]+)(?:, %[\w]+)? = (?:stream\.resource\.subview|util\.optimization_barrier|stream\.timepoint\.barrier) #{escaped}\b/,
            line
          ) ->
        [_, result] = match
        {:alias, [result]}

      Regex.match?(~r/stream\.tensor\.export\b.* #{escaped} : /, line) ->
        :read

      Regex.match?(~r/stream\.resource\.size #{escaped} : /, line) ->
        :read

      Regex.match?(~r/^\s*cf\.(br|cond_br) /, line) ->
        branch_aliases(line, name, indexed)

      true ->
        :write
    end
  end

  # The lines of an execute region end at the closing brace at the indentation
  # of the execute op
  defp region_writes?(indexed, i, line, inner) do
    [indent] = Regex.run(~r/^\s*/, line)
    closing = indent <> "}"
    token = token_regex(inner)

    indexed
    |> Enum.drop(i + 1)
    |> Enum.take_while(fn {l, _} -> not String.starts_with?(l, closing) end)
    |> Enum.any?(fn {l, _} -> Regex.match?(token, l) and not region_read?(l, inner) end)
  end

  defp region_read?(line, inner) do
    escaped = Regex.escape(inner)

    Regex.match?(~r/^\s*ro #{escaped}\[/, line) or
      (Regex.match?(~r/stream\.cmd\.copy #{escaped}\[/, line) and
         not Regex.match?(~r/, #{escaped}\[/, line))
  end

  # Successor operands become the target block's arguments
  defp branch_aliases(line, name, indexed) do
    targets = Regex.scan(~r/\^(bb\d+)\(([^:)]*) :/, line)

    aliases =
      for [_, block, operands] <- targets,
          {operand, position} <- Enum.with_index(String.split(operands, ", ")),
          operand == name do
        block_argument(indexed, block, position)
      end

    if Enum.any?(aliases, &is_nil/1), do: :write, else: {:alias, aliases}
  end

  defp block_argument(indexed, block, position) do
    Enum.find_value(indexed, fn {line, _} ->
      case Regex.run(~r/^\s*\^#{block}\((.*)\):/, line) do
        [_, params] ->
          params
          |> String.split(", ")
          |> Enum.at(position)
          |> case do
            nil -> nil
            param -> param |> String.split(":") |> hd()
          end

        nil ->
          nil
      end
    end)
  end
end
