# Evision/OpenCV stereo benchmark

Runs Evision 1.0.0 / OpenCV 5.0.0's CPU SGBM implementations on the same
50 rectified grayscale Sintel pairs as `../video`. This is a separate Mix
project because Evision declares Nx ~>0.12.1 while the main NxIREE workspace
uses Nx 1.0. Its dependencies do not change the main project's lockfile.

`OPENCV_MODES=3way` selects the four-thread CPU reference used in the latest
comparison. `VIDEO_REPEATS=1..20` repeats the full 50-pair clip inside one
measurement (default 1). The summary includes whole-loop `wall_fps` alongside
processing-only `fps`; only compare matching boundaries. Every repeated
output must agree, and one output file per distinct input is written after
timing. See the [concurrent CPU-work measurements](../../../docs/reference/stereo-concurrency-20261022/README.md).

From this directory:

```sh
mix deps.get
VIDEO_DATA=/path/to/alley_1 VIDEO_OUTPUT=/tmp/opencv-results \
  OPENCV_THREADS=4 mix run run.exs
```

For a runtime-only target, compile the dependency BEAM files on the host with
`TARGET_ARCH`, `TARGET_OS`, and `TARGET_ABI` selecting Evision's published
precompiled target. For generic ARM64 Linux these are `aarch64`, `linux`,
and `gnu`. `EVISION_ENABLE_CONTRIB=false` selects the non-contrib build.
The tested archive was `evision-nif_2.16-aarch64-linux-gnu-1.0.0.tar.gz`.
Compilation on an x86 host cannot load that ARM64 NIF; its dependencies still
compile to portable BEAM files. Verify the published archive checksum.

Stage the runtime `ebin`/`priv` directories for Evision, Nx, Complex,
Telemetry and CAStore, preserving shared-library aliases. Set
`EVISION_RUNTIME_ROOT` to their parent directory containing `lib`.
Run `run.exs` in an isolated child BEAM using the board tooling in `c100_gpu`.
The success marker is `EVISION_STEREO_PASS`. No GPU or CPU policy is changed.

The input files are prepared s32 grayscale 320×136 buffers. Conversion to
8-bit grayscale and disk access happen before timing. Each timed operation
creates the input Mats, calls native SGBM, converts fixed-point output to f32
and copies it to an Elixir binary. Three warmup pairs precede each complete
50-frame run. The native matcher is reused. Resource release follows the
VM's normal garbage collection policy. Image decode, resize, disk I/O,
resource guards, startup and warmup are excluded.

The modes are SGBM (five directions), HH (eight), 3WAY (three), and HH4
(four). Each produces one left disparity map with OpenCV's internal
right-consistency check. They do not independently aggregate two full views
like the Vulkan reference. OpenCV also uses a different matching cost,
median filtering and a fixed-point output with 1/16-pixel resolution.
Speckle filtering is disabled. Compare quality as well as FPS.

All modes use D=64, block size 3, P1=72, P2=288, uniqueness 10%, maximum
left/right difference 1 and prefilter cap 31. The published Evision 1.0.0
constructor accepts `P1`/`P2` but leaves their native values at zero.
Positional `setP1`/`setP2` calls work; the runner checks the native parameter
contract before benchmarking. Saved red/green evidence is in the report.

Evaluate predictions with the existing shared accuracy accumulator:

```sh
PYTHONPATH=bench/stereo/video uv run bench/stereo/opencv/quality.py \
  /path/to/alley_1 /tmp/opencv-results /tmp/opencv-quality.json
```

Run that command from the repository root. It measures full visible ground
truth and the common x≥64 region, since OpenCV rejects the left search-width
border. Rejected pixels count as failures in bad-or-rejected rates.

The [library survey and measured C100 results](../../../docs/reference/stereo-library-survey-20261022/README.md)
record dependency compatibility, quality and the remaining Vulkan goal.

## Standard UMat operations over Vulkan

`umat_pipeline.exs` uses published Evision calls for BGR-to-gray conversion,
resize, median filtering and remap. It checks each stage against the CPU
implementation, checks input preservation, and times complete upload through
final readback. It does not run SGBM. A UMat return or numerical pass alone
does not establish GPU execution: OpenCV can silently fall back to CPU.

With a clvk build using `-DCLVK_CLSPV_ONLINE_COMPILER=1`, run:

```sh
OPENCV_OPENCL_RUNTIME=/absolute/path/to/clvk/libOpenCL.so \
  OPENCV_OPENCL_DEVICE=:GPU:1 OPENCV_OPENCL_CACHE_ENABLE=0 CLVK_LOG=3 \
  EVISION_ENABLE_CONTRIB=false mix run --no-compile umat_pipeline.exs \
  > /tmp/umat-pipeline.log 2>&1
```

Choose the device index for the machine. The tested host's index 1 is the
RX 7800 XT; this is not a portable device identifier. Require pipeline and
execution-batch logs for all four kernels (`RGB2Gray`, `resizeLN`,
`medianFilter3`, `remap_2_32FC1`), no program build failures, and the final
`EVISION_UMAT_NUMERICAL_PASS` marker. `UMAT_ITERATIONS` defaults to 20;
`OPENCV_THREADS` defaults to four. Kernel compilation and three warmups are
excluded from timing, as are synthetic input and remap-map preparation.
Use `CLVK_LOG=1` for separate timing runs after validating execution.

The external-compiler clvk build fails its child wait inside the tested
BEAM runtime and OpenCV falls back to CPU. The upstream in-process compiler
option fixes this host integration without changing BEAM signal handling.
The [host evidence and remaining integration gaps](../../../docs/reference/evision-acceleration-20261022/README.md)
include red/green runs and the pinned, checksum-verified upstream artifacts.
