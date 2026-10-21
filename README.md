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
| `NX_IREE_COMPILER_DIR` | `cache/iree` | where the compiler wheel is unpacked |

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
flags = ["--iree-hal-target-backends=vulkan-spirv", "--iree-vulkan-target=pascal"]
bin = NxIREE.export(&MyModel.predict/2, [params_template, input_template], iree_compiler_flags: flags)
File.write!("model.bin", bin)

# device
model = NxIREE.load_exported(File.read!("model.bin"), device: "vulkan://...")
NxIREE.run(model, [params, input])
```

## Nerves

Cross-compiling is automatic when `CROSSCOMPILE` is set, as in a Nerves
build: the runtime is built for the target with the Nerves toolchain, the
host tools it needs are built for the host, and `iree-compile` is left out of
the firmware. Functions for the device are compiled on the host with
`NxIREE.export/3`. For Vulkan, the Nerves system must provide a Vulkan loader
and driver (for example Mesa's NVK on NVIDIA Tegra).

## Vulkan notes

  * Without a `--iree-vulkan-target`, NxIREE compiles for IREE's baseline
    profile plus 8-bit storage: on the plain baseline IREE misreads `u8`/`s8`
    tensors that vectorized dispatches read, such as attention masks.
  * A named target (`--iree-vulkan-target=pascal`, `rdna3`, ...) enables
    faster kernels, which assume that architecture's subgroup size. Use the
    target matching the GPU; on other GPUs results can be wrong.
  * SPIR-V has no f64: f64 inputs raise on Vulkan, so convert them to f32.
    The CPU backend keeps f64.
  * GPU results are within Vulkan's precision bounds rather than bit-exact
    with the CPU (f32 division is within 2.5 ulp, for example).

## Inputs and in-place operations

IREE compiles sort, scatter and slice updates in place on their arguments.
While compiling, NxIREE finds the arguments a module may write and passes
copies of those inputs, so caller tensors are never modified; read-only
inputs, such as model weights, are passed without copies.
