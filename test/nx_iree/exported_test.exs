defmodule NxIREE.ExportedTest do
  use ExUnit.Case, async: true

  test "loads an unsigned export before the fresh VM has used Nx types" do
    # Given an exported template and a fresh runtime-only VM.
    directory =
      Path.join(
        System.tmp_dir!(),
        "nxiree-cold-export-" <> Base.encode16(:crypto.strong_rand_bytes(12))
      )

    File.mkdir!(directory)
    on_exit(fn -> File.rm_rf!(directory) end)
    payload = Path.join(directory, "unsigned.bin")
    File.write!(payload, :erlang.term_to_binary(%{output: Nx.template({1}, :u32)}))

    code = """
    Code.ensure_loaded!(NxIREE)
    data = NxIREE.Exported.decode(File.read!(hd(System.argv())))
    IO.puts(elem(data.output.type, 1))
    """

    paths =
      for module <- [NxIREE, Nx.Tensor, Complex],
          do: module |> :code.which() |> to_string() |> Path.dirname()

    args = ["--erl", "+S 2:2"] ++ Enum.flat_map(paths, &["-pa", &1]) ++ ["-e", code, payload]

    # When the first export is decoded without a compiler or GPU startup.
    {output, status} = System.cmd(System.find_executable("elixir"), args, stderr_to_stdout: true)

    # Then its type survives decoding without allowing arbitrary new atoms.
    assert {String.trim(output), status} == {"32", 0}
  end
end
