defmodule StereoPath do
  import Nx.Defn

  # Each line is independent; steps within a line follow SGM's recurrence.
  # Layout: {steps, lines, disparities}. Costs and penalties are s32 in [0, 1e9).
  defn aggregate(cost, p1, p2) do
    steps = Nx.axis_size(cost, 0)
    lines = Nx.axis_size(cost, 1)
    disparities = Nx.axis_size(cost, 2)
    previous = Nx.broadcast(Nx.tensor(0, type: :s32), {lines, disparities})
    output = Nx.broadcast(Nx.tensor(0, type: :s32), Nx.shape(cost))

    {_i, output, _previous, _cost, _p1, _p2} =
      while {i = 0, output, previous, cost, p1, p2}, i < steps do
        current = Nx.slice(cost, [i, 0, 0], [1, lines, disparities]) |> Nx.squeeze(axes: [0])
        minimum = Nx.reduce_min(previous, axes: [1], keep_axes: true)
        # Candidates above p2 cannot win; clamping also bounds penalty additions.
        normalized = Nx.min(previous - minimum, p2)

        lower =
          normalized
          |> Nx.pad(p2, [{0, 0, 0}, {1, 0, 0}])
          |> Nx.slice([0, 0], [lines, disparities])

        upper =
          normalized
          |> Nx.pad(p2, [{0, 0, 0}, {0, 1, 0}])
          |> Nx.slice([0, 1], [lines, disparities])

        transition =
          Nx.min(Nx.min(normalized, lower + p1), Nx.min(upper + p1, p2))

        next = current + transition
        output = Nx.put_slice(output, [i, 0, 0], Nx.new_axis(next, 0))
        {i + 1, output, next, cost, p1, p2}
      end

    output
  end
end
