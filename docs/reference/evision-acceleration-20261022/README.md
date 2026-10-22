# Standard Evision operations over a Vulkan-backed OpenCV runtime

The architectural direction is feasible for operations with an existing
OpenCV GPU implementation. The closest reusable route is:

```text
Evision keyword API → OpenCV UMat / OpenCL T-API → clvk → Vulkan → NVK/RADV
```

OpenCV's [configuration reference](https://docs.opencv.org/5.0/tutorials/introduction/config_reference/config_reference.html)
describes selection of its OpenCL kernels through UMat arguments. [clvk](https://github.com/kpet/clvk)
implements OpenCL over Vulkan. Its own [application status](https://github.com/kpet/clvk/blob/main/docs/supported-applications.md)
still lists OpenCV as untested, so compatibility and speed require actual
host/board tests before choosing it as a production dependency.

## Host Vulkan execution verified

The unchanged published Evision 1.0.0 / OpenCV 5.0.0 NIF now runs a chain of
standard keyword calls on the **AMD Radeon RX 7800 XT (RADV NAVI32)**:

```text
Evision.cvtColor → Evision.resize → Evision.medianBlur → Evision.remap
```

The reusable [benchmark](../../../bench/stereo/opencv/umat_pipeline.exs)
uploads a synthetic 640×272 BGR image, produces 320×136 grayscale output,
keeps intermediates as UMat and performs one final readback per timed frame.
Remap uses prepared fractional float32 coordinate maps. Each intermediate
is independently compared with the CPU operation outside timing. All four
stages match byte-for-byte for this fixture, and the input remains unchanged.
The test allows at most one intensity level for implementation rounding.

[Successful execution log](host-online-pipeline.log) records creation of
`RGB2Gray`, `resizeLN`, `medianFilter3` and `remap_2_32FC1` pipelines and
execution batches of four commands. This establishes actual GPU execution
through Evision, rather than merely OpenCL discovery or UMat construction.
No OpenCV kernel, Evision NIF, system ICD configuration or production
dependency was changed. No board workload or board configuration changed.

### BEAM integration failure and upstream configuration fix

clvk's external compiler is incompatible with the child-wait behavior of
the tested BEAM process. The compiler exits successfully and emits valid
SPIR-V, but clvk's caller gets `wait4(shell) = -1 ECHILD`, so `pclose`
returns -1. BEAM's observed ignored-signal mask includes SIGCHLD. clvk treats
the wait result as compilation failure; OpenCV then silently runs CPU code.
[Process evidence](host-compiler-wait.txt) includes both child exit 0 and
the failed wait. This explains this host failure; it does not establish
that every BEAM/OS combination behaves the same way.

Use clvk's existing **`-DCLVK_CLSPV_ONLINE_COMPILER=1`** build option, which
calls the compiler in process. No signal-handler override or custom process
wrapper is needed. Pinned upstream implementation:
[external wait](https://github.com/kpet/clvk/blob/3c155df22f9c30ef2c096469e9fc382d023aa899/src/utils.cpp#L49),
[failure check](https://github.com/kpet/clvk/blob/3c155df22f9c30ef2c096469e9fc382d023aa899/src/program.cpp#L1364),
[in-process compiler](https://github.com/kpet/clvk/blob/3c155df22f9c30ef2c096469e9fc382d023aa899/src/program.cpp#L1469).

The same benchmark with the OpenCV kernel cache disabled provides the
toggle proof: [external compiler](host-external-pipeline.log) fails all four
program builds and has no created GPU kernel pipelines;
[in-process compiler](host-online-pipeline.log) builds and dispatches all
four. Both return correct pixels and exit zero. Therefore process status
and the numerical marker alone must never be used for GPU acceptance.

Both binaries are official upstream CI artifacts from source commit
`3c155df22f9c30ef2c096469e9fc382d023aa899`, workflow run `39485071993`:

| Configuration | Artifact ID | Verified archive SHA256 |
| --- | --- | --- |
| external compiler | 12294835117 | `7d42037ce16c089aace83d6d09f3d61c57df2e849a4760d3baf1416ab52b6b1f` |
| in-process compiler | 12293192832 | `44509ae27b4a1ab021d473f9cbb15792cf2be7e74abb7837d4d1cd84e62ffeb9` |

The artifacts are Ubuntu 24.04 **x86_64**, cached under
`~/.cache/stereo-accel/`. They are not ARM64/Nerves deployment artifacts.
Evision remains the normal locked dependency in the isolated benchmark.
Reproduction instructions are in its [README](../../../bench/stereo/opencv/README.md#standard-umat-operations-over-vulkan).

### Performance and integration limits

A [separate 1,000-frame run](host-pipeline-timing.log), with clvk informational
logging disabled, measured 0.1191 ms/frame CPU and 0.4682 ms/frame UMat on this
small synthetic chain, including input Mat creation, upload and final
readback to an Elixir binary. GPU p50 was 0.395 ms and p95 1.044 ms. Map
creation, input generation, compilation and warmup are excluded. These are
host preprocessing timings, **not stereo throughput or C100 results**;
this establishes compatibility, not a speedup. It does not isolate which
cost dominates or predict SGBM speed.

Evision 1.0.0's `evision_to(... cv::UMat& ...)` in `c_src/evision.cpp`
copies each UMat argument using `copyTo`, including remap maps. This can
retain intermediates on device, but adds device copies between calls.
`Evision.UMat.get/1` already provides owning CPU readback. The next generic
integration work should measure these copies and preserve immutability
while removing unnecessary copying of read-only inputs or composing native
operations. A mature native pipeline interface should be preferred where
it provides that behavior. The current positional Mat guards also need
UMat support for equivalent positional syntax; keyword calls already work.

No complete OpenCV Vulkan backend or GPU SGBM implementation is supplied by
this spike. ARM64 clvk/compiler packaging, target capability checks and
target load stability remain outstanding.

## Verified binding behavior

The existing isolated Evision 1.0.0 ARM64 NIF was probed on C100 without
changing firmware, native libraries or power policy. The reusable source is
`bench/stereo/opencv/probe_acceleration.exs`. [Captured run](board.log):

```text
EVISION_HAVE_OPENCL: false
EVISION_USE_OPENCL: false
EVISION_UMAT_CONSTRUCTION: %Evision.UMat{...}
EVISION_MAT_CVTCOLOR: %Evision.Mat{...}
EVISION_UMAT_POSITIONAL_CVTCOLOR=no function clause matching in Evision.cvtColor/2
EVISION_UMAT_KEYWORD_CVTCOLOR: %Evision.UMat{...}
EVISION_ACCELERATION_PROBE_PASS
```

Thus UMat already exists and native keyword conversion works. The generated
positional guard rejects UMat; a new Elixir image-processing implementation
is unnecessary for this operation. Standard calls such as
`Evision.cvtColor(src: umat, code: 6)` work through the published binding.
This run had **no OpenCL device** and therefore proves API behavior only,
not GPU execution or acceleration. A UMat return alone never proves GPU use.

## The SGBM gap

OpenCV's [pinned StereoSGBM implementation](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/stereo/src/stereosgbm.cpp#L2049)
calls `leftarr.getMat()` / `rightarr.getMat()` and runs CPU implementations.
There is no accelerated branch in this compute method. The
[OpenCL SGBM request](https://github.com/opencv/opencv/issues/8195) remains open.
StereoBM has OpenCL code, but it is a different matcher and cannot silently
replace SGBM while claiming equal quality. A wrapper or build flag cannot
make the existing SGBM CPU implementation execute as Vulkan shaders.

A useful implementation direction is to verify existing OpenCV OpenCL
preprocessing/rectification/filter kernels over clvk first, keeping UMat
intermediates resident, then expose a verified GPU matcher through an OpenCV
module or StereoMatcher-compatible extension and generated Evision bindings.
Prefer a clearly licensed existing matcher; the prior
[library survey](../stereo-library-survey-20261022/README.md) records the
OpenCL SGM candidate's unresolved license and diagonal-path correctness gaps.
The present portable matcher remains a benchmark candidate, not a claim to
have supplied an OpenCV backend.

First integration acceptance should require haveOpenCL/useOpenCL true on the
selected GPU, actual dispatch evidence, CPU-vs-GPU output checks and transfer
accounting for a small operation chain. Check native UMat copy behavior and
unsupported-operation CPU fallback explicitly. Package reusable runtime code
in Buildroot/dedicated libraries; keep board policy separate. No clvk package,
new Evision build or upstream issue/PR was installed or published by this probe.
