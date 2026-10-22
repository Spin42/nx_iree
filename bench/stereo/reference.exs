defmodule StereoPathReference do
  def aggregate(flat, {steps, lines, disparities}, p1, p2) do
    true = length(flat) == steps * lines * disparities
    previous = List.duplicate(List.duplicate(0, disparities), lines)

    {output, _} =
      flat
      |> Enum.chunk_every(lines * disparities)
      |> Enum.map_reduce(previous, fn plane, previous ->
        next =
          plane
          |> Enum.chunk_every(disparities)
          |> Enum.zip(previous)
          |> Enum.map(fn {costs, prior} ->
            minimum = Enum.min(prior)
            prior = List.to_tuple(prior)

            costs
            |> Enum.with_index()
            |> Enum.map(fn {cost, d} ->
              choices = [elem(prior, d), minimum + p2]
              choices = if d > 0, do: [elem(prior, d - 1) + p1 | choices], else: choices

              choices =
                if d + 1 < disparities,
                  do: [elem(prior, d + 1) + p1 | choices],
                  else: choices

              cost + Enum.min(choices) - minimum
            end)
          end)

        {List.flatten(next), next}
      end)

    List.flatten(output)
  end

  def fixtures do
    basic = [
      {"first pixel", {1, 2, 3}, [5, 1, 9, 0, 8, 2], 3, 10},
      {"one disparity", {4, 1, 1}, [7, 100, 4, 9], 3, 10},
      {"no disparity wrap", {3, 1, 4}, [0, 50, 50, 50, 10, 10, 10, 10, 8, 4, 7, 3], 2, 20},
      {"zero penalties", {3, 1, 3}, [9, 1, 7, 2, 10, 6, 8, 3, 4], 0, 0},
      {"large common offset", {3, 1, 3}, [100_000, 100_009, 100_025, 8, 1, 5, 6, 9, 3], 4, 15},
      {"large penalty overflow", {3, 1, 2}, [0, 999_999_999, 999_999_999, 999_999_999, 0, 0],
       999_999_999, 999_999_999},
      {"large boundary exclusion", {3, 1, 4},
       [
         0,
         999_999_999,
         999_999_999,
         999_999_999,
         999_999_999,
         999_999_999,
         999_999_999,
         999_999_999,
         0,
         0,
         0,
         0
       ], 0, 999_999_999},
      {"many independent lines", {17, 5, 32}, values({17, 5, 32}, 7), 8, 32},
      {"changed input", {17, 5, 32}, values({17, 5, 32}, 29), 8, 32}
    ]

    boundaries =
      for disparities <- [2, 31, 33, 63, 65, 97, 128] do
        shape = {5, 3, disparities}
        minima = {min(31, disparities - 1), min(32, disparities - 1), disparities - 1}

        costs =
          values(shape, disparities)
          |> Enum.with_index()
          |> Enum.map(fn {value, index} ->
            if index < 3 * disparities do
              if rem(index, disparities) == elem(minima, div(index, disparities)),
                do: 0,
                else: 200
            else
              value
            end
          end)

        {"disparity boundary #{disparities}", shape, costs, 20, 7}
      end

    basic ++ boundaries
  end

  def values(shape, seed) do
    count = shape |> Tuple.to_list() |> Enum.product()
    for i <- 0..(count - 1), do: rem(i * 73 + div(i, 13) * 19 + seed * 11, 251)
  end
end
