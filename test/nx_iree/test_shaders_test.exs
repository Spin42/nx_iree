defmodule NxIREE.TestShadersTest do
  use ExUnit.Case, async: true

  alias NxIREE.TestShaders

  @glslang System.get_env("GLSLANG") || System.find_executable("glslangValidator") ||
             System.find_executable("glslang")
  @spirv_val System.get_env("SPIRV_VAL") || System.find_executable("spirv-val")

  @tag :tmp_dir
  @tag skip: is_nil(@glslang) or is_nil(@spirv_val)
  test "the embedded shaders validate and regenerate from their sources", %{tmp_dir: dir} do
    for {name, source, spirv} <- [
          {"single", TestShaders.single_source(), TestShaders.single()},
          {"pair", TestShaders.pair_source(), TestShaders.pair()},
          {"specialized", TestShaders.specialized_source(), TestShaders.specialized()}
        ] do
      source_path = Path.join(dir, name <> ".comp")
      embedded = Path.join(dir, name <> ".spv")
      generated = Path.join(dir, name <> "-generated.spv")
      File.write!(source_path, source)
      File.write!(embedded, spirv)

      args = ["-V", "--target-env", "vulkan1.1", "-S", "comp", source_path, "-o", generated]
      assert {_, 0} = System.cmd(@glslang, args, stderr_to_stdout: true)
      assert File.read!(generated) == spirv, "#{name} does not regenerate from its source"

      assert {_, 0} =
               System.cmd(@spirv_val, ["--target-env", "vulkan1.1", embedded], stderr_to_stdout: true)
    end
  end
end
