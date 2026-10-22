# NxIREE

Runs [Nx](https://github.com/elixir-nx/nx) functions through the
[IREE](https://iree.dev) compiler and runtime. Nx functions are lowered to
StableHLO by [EXLA](https://github.com/elixir-nx/nx/tree/main/exla), compiled
by IREE for a target (CPU, Vulkan, CUDA, Metal, ...) and run by IREE's
runtime, which is small enough for embedded Linux devices such as Nerves
targets. On the device, ahead-of-time compiled functions need neither EXLA
nor the IREE compiler.

## Installation

```elixir
def deps do
  [
    {:nx_iree, path: "../nx_iree"},
    # only needed where functions are compiled (the host)
    {:exla, "~> 1.0"}
  ]
end
```

The first build clones the IREE runtime (v3.13.0), builds it with CMake and
Ninja, and extracts `iree-compile` from the `iree-base-compiler` wheel. It
needs `git`, `cmake`, `ninja` and a C++17 compiler.

Build settings, all optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NX_IREE_VERSION` | `3.13.0` | IREE compiler wheel version |
| `NX_IREE_GIT_REV` | `v3.13.0` | IREE runtime git revision |
| `NX_IREE_SOURCE_DIR` | `~/.cache/nx_iree/iree-<rev>` | IREE checkout |
| `NX_IREE_COMPILER_DIR` | downloaded wheel in `cache/iree` | existing compiler tools or unpacked wheel directory to use |

## Running defn functions

```elixir
Nx.Defn.global_default_options(compiler: NxIREE.Compiler)
Nx.global_default_backend(NxIREE.Backend)

f = Nx.Defn.jit(fn a, b -> Nx.dot(a, b) |> Nx.exp() end)
f.(Nx.iota({3, 4}, type: :f32), Nx.iota({4, 2}, type: :f32))
```

`NxIREE.Compiler` accepts:

  * `:iree_compiler_flags` - extra `iree-compile` flags. The target backend
    follows the device unless `--iree-hal-target-backends` is given.
  * `:iree_runtime_options` - `[device: uri_or_device]`, the device to run on.

Compiled functions are cached per input shapes and options. Results stay on
the device as `NxIREE.Backend` tensors and are passed back in without copies
on the same device. `Nx.Defn.to_backend/1` returns the backend for the
configured device, and backend operations run on their inputs' device.

### Devices

`NxIREE.list_devices/0,1` lists devices by driver (`local-sync`,
`local-task`, `vulkan`, `cuda`, `metal`, ...). Without a `:device`, the first
GPU is used, else the first CPU device. Both can be configured:

```elixir
config :nx_iree,
  drivers: ["vulkan", "local-sync"],
  default_device: "vulkan://GPU-00000000030000000000000000000000"
```

Devices are created on first use, so applications that never call nx_iree
do not open GPUs.

## Ahead-of-time compilation

`NxIREE.export/3` compiles a function on the host for a given target and
returns a binary. `NxIREE.load_exported/2` loads it on a device and
`NxIREE.run/2` calls it, with only Nx and nx_iree's runtime installed:

```elixir
# host
flags = ["--iree-hal-target-backends=vulkan-spirv"]
bin = NxIREE.export(&MyModel.predict/2, [params_template, input_template], iree_compiler_flags: flags)
File.write!("model.bin", bin)

# device
model = NxIREE.load_exported(File.read!("model.bin"), device: "vulkan://...")
NxIREE.run(model, [params, input])
```

Exports for the CPU target a generic CPU of the host's architecture unless
told otherwise; for another device pass its triple and CPU, for example
`--iree-llvmcpu-target-triple=aarch64-linux-gnu --iree-llvmcpu-target-cpu=cortex-a57`.
Functions compiled for calls on the CPU use the host CPU's features.

`NxIREE.Serving.new/1` wraps a loaded export in the existing `Nx.Serving`
scheduler without compilation on the device:

```elixir
serving = NxIREE.Serving.new(model) |> Nx.Serving.streaming()
{:ok, _pid} = Nx.Serving.start_link(name: MyServing, serving: serving)
batches = Stream.map(inputs, fn {a, b} -> Nx.Batch.stack([{a, b}]) end)
results = Nx.Serving.batched_run(MyServing, batches)
```

Each entry is one complete invocation, so this adapter sets the maximum
batch size to one. Stream results are `{:batch, output, metadata}` with a
leading axis of size one; outputs stay on the device. Host inputs are
uploaded for the invocation, while caller-owned resident inputs are borrowed.
Reshape and squeeze use independent IREE buffer views without compilation,
GPU dispatch or tensor copies. Input-stream submission waits for the previous
invocation; applications must also consume results promptly. This provides
neither a global queue bound across clients nor a camera deadline guarantee.

## Module cache

Unchanged Nx code, Bumblebee servings included, runs on a device without
EXLA or the IREE compiler when its modules come from a cache compiled on the
host. With `:iree_module_cache` (or `config :nx_iree, module_cache: dir`),
each compiled function is stored under a key derived from its defn
expression, input shapes and compiler flags:

```elixir
defn_options = [
  compiler: NxIREE.Compiler,
  iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv"],
  iree_module_cache: Application.app_dir(:my_app, "priv/nx_iree_cache")
]

Bumblebee.Text.text_classification(model, tokenizer,
  compile: [batch_size: 4, sequence_length: 32],
  preallocate_params: true,
  defn_options: defn_options
)
```

On the host, starting the serving (or `Nx.Defn.compile/3` on the function)
compiles into the directory; ship it with the application and the device
loads the modules from it. The flags and Nx, Bumblebee and model versions
must be the same on both sides, since they shape the key. A function missing
from the cache raises on the device, naming the entry.

Preprocessing written as `defn` and called eagerly, such as Bumblebee's image
resizing, uses the global defn options: set them with
`Nx.Defn.global_default_options/1` to the same options so it is compiled and
cached as well. Use `preallocate_params: true` with Bumblebee servings, or
the parameters are uploaded to the device on every call.

## Host calls

IREE modules cannot call back into Elixir. Named `io_call`s that have no
callback and no `:hooks` entry do nothing and are removed, so Bumblebee's
text generation works; hooks with a callback (Bumblebee streaming),
`print_value`/`print_expr` and `Nx.runtime_call/4` raise when compiling.

## Nerves

Cross-compiling is automatic when `CROSSCOMPILE` is set, as in a Nerves
build: the runtime is built for the target with the Nerves toolchain, the
host tools it needs are built for the host, and `iree-compile` is left out of
the firmware. Functions for the device are compiled on the host, into a
module cache or with `NxIREE.export/3`. For Vulkan, the Nerves system must
provide a Vulkan loader and driver (for example Mesa's NVK on NVIDIA Tegra).

## Vulkan notes

For application-authored compute kernels, `NxIREE.Vulkan.export/4` imports
SPIR-V into the normal export/load/run API with device-resident tensors.
See [custom Vulkan kernels](docs/vulkan-kernels.md) and the
[portable stereo benchmark](bench/stereo/README.md).

To use a locally built compiler, set `NX_IREE_COMPILER_DIR` to the directory
containing `iree-compile` (for an IREE CMake build, its `tools/` directory).
A wheel's `iree/` directory also works. Mix selects it even when a compiler
was already installed; unsetting the variable restores the downloaded
compiler on the next build. Use a separate `:iree_module_cache` directory
when comparing compiler builds so previously compiled modules are not reused.
This setting is only for the host; Nerves cross builds omit the compiler.

  * Without a `--iree-vulkan-target`, NxIREE compiles for IREE's baseline
    profile plus 8-bit storage: on the plain baseline IREE misreads `u8`/`s8`
    tensors that vectorized dispatches read, such as attention masks.
  * Named targets (`--iree-vulkan-target=pascal`, `rdna2`, ...) fail to
    compile transformers in IREE 3.13: two embedding gathers added and then
    reduced, as in every BERT and GPT embedding layer, hit a SPIR-V
    legalization error. Their kernels also assume the architecture's subgroup
    size, so they are wrong on other GPUs. An explicit capability list must
    match the selected device. Device libraries can supply that list, such
    as `C100GPU.Target.compiler_flags/0` from `c100_gpu` for the C100.
  * `iree_tuning: {ProviderModule, :spec_name}` (a `NxIREE.Compiler` and
    `NxIREE.export/3` option) selects a transform-dialect spec from a library
    implementing `NxIREE.Tuning.fetch/1`. The callback returns
    `{:ok, mlir_text, supported_backends}` or `:error`. Cache keys contain
    the provider, spec name and SHA-256 of the text instead of a host path.
    Device-specific providers belong in separate libraries; the C100's
    Maxwell matmul provider is `{C100GPU.Tuning, :maxwell_matmul}` in
    [`c100_gpu`](../c100_gpu).
  * Command buffer memoization is off by default for GPU targets
    (`--iree-hal-memoization=false`): IREE 3.13 replays the first call's
    dynamic slice offsets from memoized Vulkan command buffers.
  * Software Vulkan devices (lavapipe, SwiftShader) are not opened unless
    `config :nx_iree, software_vulkan: true`; lavapipe's threads busy-wait.
  * SPIR-V has no f64: f64 inputs raise on Vulkan, so convert them to f32.
    The CPU backend keeps f64.
  * GPU results are within Vulkan's precision bounds rather than bit-exact
    with the CPU (f32 division is within 2.5 ulp, for example).

## Inputs and in-place operations

IREE compiles sort, scatter and slice updates in place on their arguments.
While compiling, NxIREE finds the arguments a module may write and passes
copies of those inputs, so caller tensors are never modified; read-only
inputs, such as model weights, are passed without copies.

The [native VGA stereo measurement](docs/reference/stereo-vga30-20261022/README.md)
uses the existing Nx.Serving adapter and Evision input conversion: 31.3
prepared-pair GPU FPS and paced 30 Hz network replay at 640×480 per eye/D64.
Four contexts sustain the replay with normal governors and otherwise idle
CPUs, at about 124 ms median processing latency. This is bounded diagnostic
replay evidence; camera capture and standard Evision GPU SGBM integration
remain open.
