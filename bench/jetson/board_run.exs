# Evaluated in the board's main node: pauses any ncnn test processes, pins the
# GPU clock, runs /root/nxiree/run.exs (or NXIREE_SCRIPT) in a separate BEAM (so a NIF crash
# cannot take the main node down), then restores the clock and resumes.
sig = fn s ->
  for p <- Path.wildcard("/proc/[0-9]*"),
      String.starts_with?(String.trim(File.read!(p <> "/comm") |> to_string()), "test_") do
    System.cmd("kill", ["-#{s}", Path.basename(p)])
  end
end

sig.("STOP")

gpu = "/sys/class/devfreq/57000000.gpu"
saved = for f <- ~w(min_freq max_freq), do: {f, String.trim(File.read!("#{gpu}/#{f}"))}
pin = System.get_env("NXIREE_PIN_HZ", "768000000")
File.write!("#{gpu}/max_freq", pin)
File.write!("#{gpu}/min_freq", pin)

try do
  erts = hd(Path.wildcard("/srv/erlang/erts-*"))
  env = [
    {"ROOTDIR", "/srv/erlang"},
    {"BINDIR", erts <> "/bin"},
    {"EMU", "beam"},
    {"PROGNAME", "erl"},
    {"HOME", "/root"},
    {"NVK_I_WANT_A_BROKEN_VULKAN_DRIVER", "1"}
  ]

  # NXIREE_BOARD_ENV="K=V,K=V" adds variables such as NAK_DEBUG or VK_DRIVER_FILES
  extra =
    System.get_env("NXIREE_BOARD_ENV", "")
    |> String.split(",", trim: true)
    |> Enum.map(fn kv -> kv |> String.split("=", parts: 2) |> List.to_tuple() end)

  env = env ++ extra

  script = "/root/nxiree/" <> System.get_env("NXIREE_SCRIPT", "run.exs")
  eval = ~s|{ok,_}=application:ensure_all_started(elixir), 'Elixir.Code':eval_file(<<"#{script}">>), halt().|

  args = ["-boot", "/srv/erlang/releases/0.1.0/start_clean", "-boot_var", "RELEASE_LIB", "/srv/erlang/lib", "-noshell", "-eval", eval]
  File.rm("/root/nxiree.log")
  port = Port.open({:spawn_executable, erts <> "/bin/erlexec"}, [:binary, :exit_status, :stderr_to_stdout, args: args, env: Enum.map(env, fn {k, v} -> {~c"#{k}", ~c"#{v}"} end)])
  {:os_pid, os_pid} = Port.info(port, :os_pid)

  collect = fn collect, acc ->
    receive do
      {^port, {:data, d}} -> collect.(collect, acc <> d)
      {^port, {:exit_status, rc}} -> {acc, rc}
    after
      840_000 ->
        System.cmd("kill", ["-9", "#{os_pid}"])
        {acc, :timeout}
    end
  end

  {out, rc} = collect.(collect, "")

  # NXIREE_DUMP_OUT keeps the VM's output (e.g. NAK shader dumps) in a file
  case System.get_env("NXIREE_DUMP_OUT") do
    nil -> IO.puts(out <> "\nexit #{inspect(rc)}")
    path -> File.write!(path, out) && IO.puts("wrote #{byte_size(out)} bytes to #{path}\nexit #{inspect(rc)}")
  end

  IO.puts(File.read("/root/nxiree.log") |> elem(1) |> to_string())
after
  # min first, so it never exceeds the restored max
  File.write!("#{gpu}/min_freq", saved |> List.keyfind("min_freq", 0) |> elem(1))
  File.write!("#{gpu}/max_freq", saved |> List.keyfind("max_freq", 0) |> elem(1))
  sig.("CONT")
end

{d, _} = System.cmd("dmesg", [])
d |> String.split("\n") |> Enum.filter(&(&1 =~ ~r/ILLEGAL_INSTR|TRAP|MMU fault|page fault|nouveau.*(error|fail)/i)) |> Enum.take(-6) |> Enum.each(&IO.puts/1)
:ok
