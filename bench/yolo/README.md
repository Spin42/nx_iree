# YOLOv5 on IREE's Vulkan backend

Applications run YOLO models the Nx way: `AxonOnnx.import/1` (with nearest
`Resize` support, the `resize-nearest` branch of `../axon_onnx`) and
`Axon.build(model, compiler: NxIREE.Compiler)`; the unmodified YOLOv5n export
gives the reference detections that way on EXLA and on Vulkan. This
directory is the benchmark that measures how far IREE itself can take the
model: it drives `iree-compile` on the ONNX graph directly with graph
rewrites and compiler options the Nx path does not apply yet, and is the
reference for moving them into `NxIREE.Compiler`.

Vendor-neutral recipe and tools for running an ultralytics YOLOv5 ONNX
export on any Vulkan GPU through IREE 3.13. Device-specific tuning specs live
with the device's own library (for example `c100_gpu/priv/tuning/`); every
step here works without one.

| File | Purpose |
| --- | --- |
| `prepare.sh` | runs the model rewrites below and imports the result |
| `cut_head.py` | cuts the Detect layer off and sets the input size; three channels-last head outputs |
| `split_concat_conv.py` | replaces 1x1 convolutions of channel concatenations by sums of convolutions |
| `pad_yolov5_onnx.py` | pads the stem to 4 input channels and the heads to 256 output channels |
| `stem_space_to_depth.py` | turns the 6x6 stride-2 stem into space-to-depth plus a 3x3 convolution |
| `split_conv_bias.py` | moves 3x3 stride-1 convolution biases into separate Adds, for Winograd |
| `to_fp32.py` | converts an fp16 export to fp32, for GPUs without fast fp16 |
| `compile.sh` | compiles for a `--iree-vulkan-target` preset; `F16=1`, `WINOGRAD=1`, optional tuning spec |
| `family_spec.py` | writes a tuning spec that overrides selected op families only |
| `blob.py`, `reference.py` | letterboxed input blob, and onnxruntime reference detections |
| `decode.py`, `compare.py` | host-side box decoding, and the detection-level check |
| `run.sh` | checks a module's detections against the reference, then times it |

Every rewrite leaves the model's values unchanged; each was checked against
onnxruntime on the original export (differences of 1e-5 from summation order).

## Steps

```sh
./prepare.sh yolov5s.onnx yolov5s-512x640 512 640        # fp16 export
F16=1 WINOGRAD=1 ./compile.sh yolov5s-512x640.mlir yolov5s.vmfb pascal [spec.mlir]
python3 blob.py image.jpg blob.bin 512 640 f16
python3 to_fp32.py yolov5s.onnx fp32.onnx && python3 cut_head.py fp32.onnx fp32-cut.onnx 512 640
python3 blob.py image.jpg blob32.bin 512 640 && python3 reference.py fp32-cut.onnx blob32.bin 512 640 ref.bin
ELEM=f16 SIZE=512x640 HEADS=1 ./run.sh yolov5s.vmfb blob.bin ref.bin [vulkan://<path>]
```

`--device=vulkan` takes the first Vulkan device, which on a desktop with an
integrated GPU is usually the integrated one; list them with
`iree-run-module --dump_devices=vulkan`. The reference and blobs are
device-independent and can be made on the development host. For the fp32
path, prepare the `to_fp32.py` output and compile without `F16=1`.

## Results

| GPU (driver) | Input | Build | ms per frame |
| --- | --- | --- | ---: |
| GM20B, Jetson Nano (NVK) | 640x512 | fp16, Winograd, C100 spec | 199 |
| GM20B, Jetson Nano (NVK) | 640x640 | fp16, Winograd, C100 spec | 242 |
| GM20B, Jetson Nano (NVK) | 640x640 | full export, fp32, padded only, `turing` | 738 |
| GM20B, Jetson Nano (NVK), YOLOv5n | 640x512 / 640x640 | fp16, Winograd, C100 YOLOv5s spec | 115 / 134 |
| RX 7800 XT (RADV) | 640x512 | fp16, Winograd, no spec | 5.6 |
| RX 7800 XT (RADV) | 640x640 | full export, fp32, padded only, `rdna3` | 8.3 |

The detections match onnxruntime's fp32 result in every row. On the GM20B
the steps were cumulative; from the padded full export in fp32 at 640x640:
fp16 with f16 accumulators and a spec 458 ms, Detect layer on the host
427 ms, a register-conscious spec 331 ms, Winograd 300 ms, 640x512 254 ms,
concatenations split 225 ms, Focus stem and pooling rule 199 ms.

## Why each step

* **im2col and channels-last.** IREE's direct SPIR-V lowering of NCHW 3x3
  convolutions is wrong (first visible at `/model.2/m/m.0/cv2/conv`, on RADV
  and NVK, IREE 3.11 and 3.13); channels-last direct convolutions are
  correct but slow.
* **f16 accumulators.** The ONNX importer lowers f16 convolutions with f32
  accumulators (`matmul_..._f16xf16xf32`), so a GPU with packed fp16
  arithmetic still runs f32 FMAs plus conversions. `F16=1` adds
  `--iree-input-demote-f32-to-f16`, which keeps accumulators in f16 as
  TensorRT's fp16 mode does. Scores move by up to about 0.02 and boxes by a
  few pixels, which is why `compare.py` checks detections, not values.
  Adreno, Mali and Maxwell/Pascal Tegra GPUs run fp16 at twice the f32 rate.
  The `turing` and `rdna3` presets cannot compile the f16 graph (their
  cooperative-matrix path rejects the stem); use `pascal` or another preset
  without cooperative matrices.
* **Detect on the host.** The exported Detect layer's reshapes and
  concatenation cost several layout copies on the GPU; `decode.py` does the
  same arithmetic on the CPU, as the ncnn and TensorRT YOLO demos do. It also
  makes the graph fully convolutional, so any input size works.
* **Input size.** Choose sizes whose feature-map row counts divide into the
  spec's tiles: 640x480 gives 15x20 and 30x40 maps (300 and 1200 rows),
  which no power-of-two tile divides, and ran slower than 640x640 on the
  GM20B; 640x512 letterboxes the same 4:3 frame and divides evenly.
* **Concatenations.** IREE keeps channel concatenations channel-major and
  transposes their result back for each 1x1 convolution; splitting the
  convolutions removes them (37 to 4 ms of transposes per frame on the
  GM20B).
* **Padding and the Focus stem.** K = 108 (stem) and N = 255 (heads) do not
  tile; padding fixes both. Rewriting the 6x6 stride-2 stem as
  space-to-depth plus a 3x3 convolution replaces its 6x6 im2col copy, the
  slowest dispatch, with a Winograd convolution.
* **Winograd.** `WINOGRAD=1` lowers 3x3 stride-1 convolutions with Winograd
  F(6x6, 3x3) and keeps the transformed filters as compile-time constants.
  IREE 3.13's rewrite is wrong for convolutions that carry a bias, hence
  `split_conv_bias.py`. The input and output transforms cost memory traffic,
  so measure it on each GPU.
* **Preset.** Run every preset that compiles and keep the fastest correct
  one; on NVK the `adreno` preset gave wrong output, which may not hold on a
  real Adreno, so always run `run.sh` first.

## Tuning a new GPU

1. Profile one frame with Tracy (`iree-tracy-capture`, a runtime built with
   tracing) and aggregate GPU time per dispatch name. Use the dispatch with
   the fewest occurrences in the capture as the frame count. Dispatch names
   carry the matmul shape (`matmul_MxNxK`), so GFLOPS per family follow
   directly.
2. Dump the compiler's choices with
   `--iree-hal-dump-executable-configurations-to=<dir>` and the SPIR-V with
   `--iree-hal-dump-executable-binaries-to=<dir>`. Compiling a dispatch with
   the driver's shader compiler (for Mesa, a no-op DRM shim) gives register
   counts and spills without the board.
3. Override one family at a time with `family_spec.py` and keep only the
   rules that win on the board. On the GM20B, what won was an 8x4 output
   block per thread, at most 128 registers, tiles that divide the shape, and
   for one pooling a thread layout along channels; one configuration for all
   matmuls was much slower than the defaults.

```sh
python3 family_spec.py spec.mlir "0<=600:16,128,32:32,2" "1=32:64,32,16:8,8"
```

   Rules match exact values or ranges of loop dimensions; the first matching
   rule wins.
