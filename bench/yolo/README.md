# YOLOv5 on IREE's Vulkan backend

Vendor-neutral recipe and tools for running an ultralytics YOLOv5 ONNX
export (640x640) on any Vulkan GPU through IREE 3.13. Device-specific
tuning lives with the device's own library (for example
`c100_gpu/priv/tuning/yolov5s_maxwell.mlir`); everything here works without it.

| File | Purpose |
| --- | --- |
| `to_fp32.py` | converts an fp16 export to fp32 |
| `pad_yolov5_onnx.py` | pads the stem to 4 input channels and the detect heads to 256 output channels; outputs unchanged |
| `compile.sh` | compiles with im2col convolutions and channels-last layout for a `--iree-vulkan-target` preset, optionally with a tuning spec; `F16=1` keeps everything, accumulators included, in f16 |
| `run.sh` | checks the output against a reference (f32 within 0.05, f16 at the detection level), then times it with `iree-benchmark-module` |
| `compare.py` | detection-level comparison of an output with the f32 reference |
| `family_spec.py` | writes a tuning spec that overrides selected matmul or generic op families only |
| `reference/` | `yolo_dump`: the input blob and OpenCV DNN's output for an image, the independent reference |

## Steps

```sh
python3 to_fp32.py yolov5s.onnx yolov5s-fp32.onnx           # for the f32 path only
python3 pad_yolov5_onnx.py yolov5s-fp32.onnx yolov5s-padded.onnx
iree-import-onnx yolov5s-padded.onnx -o yolov5s.mlir
./compile.sh yolov5s.mlir yolov5s.vmfb <preset> [spec.mlir]
yolo_dump yolov5s-fp32.onnx bus.jpg bus-blob.bin bus-expected.bin   # once, on any machine
./run.sh yolov5s.vmfb bus-blob.bin bus-expected.bin [vulkan://<n>]
# fp16: pad and import the fp16 export, F16=1 ./compile.sh ..., then
# ELEM=f16 ./run.sh yolov5s.vmfb bus-blob-f16.bin bus-expected.bin
```

`--device=vulkan` takes the first Vulkan device, which on a desktop with an
integrated GPU is usually the integrated one. List the devices with
`iree-run-module --dump_devices=vulkan` and pass
`vulkan://<path>` from that list.

The blob and reference are device-independent, so they can be produced on
the development host and copied to the board with the module.

## Why each step

* **im2col and channels-last are required for correctness.** IREE's direct
  SPIR-V lowering of NCHW 3x3 convolutions writes values to the wrong
  positions (first visible at `/model.2/m/m.0/cv2/conv`). This reproduces on
  RADV and on NVK, with IREE 3.11 and 3.13, so it is a compiler bug, not a
  driver one. Channels-last alone fixes correctness but the direct
  convolutions remain slow (about 300 ms per layer on a Maxwell GPU).
* **fp16 needs f16 accumulators.** The ONNX importer lowers f16
  convolutions with f32 accumulators (`matmul_..._f16xf16xf32`), so a GPU
  with packed fp16 arithmetic still runs f32 FMAs and pays for the
  conversions; on the GM20B this made the fp16 export slower than fp32.
  `F16=1` adds `--iree-input-demote-f32-to-f16`, which keeps the
  accumulators in f16 as TensorRT's fp16 mode does, and the matmuls then
  compile to packed f16 FMAs. Accuracy stays at the detection level (the
  same candidates and classes for the bus image) while box coordinates move
  by a few pixels, so check f16 modules with `ELEM=f16 run.sh` (it uses
  `compare.py`). Mobile GPUs (Adreno, Mali) and Maxwell/Pascal Tegra run
  fp16 at twice the f32 rate, so this is the first thing to try there. The
  `turing` and `rdna3` presets cannot compile the f16 graph (their
  cooperative-matrix path rejects the stem); use `pascal` or another preset
  without cooperative matrices.
* **Padding.** The stem's im2col matmul has K = 6x6x3 = 108 and the detect
  heads have N = 255; neither divides into the vector widths the SPIR-V
  matmul pipeline needs, so both fall back to a scalar path. Padding in the
  ONNX graph is the only place it works: IREE's `pad-linalg-ops`
  preprocessing runs before im2col and never sees these matmuls, and a Slice
  directly on a padded convolution breaks channels-last propagation, so the
  slice sits behind a Reshape.
* **Preset choice matters more than expected.** Run every preset that
  compiles and keep the fastest correct one. Measured so far:

| GPU (driver) | Fastest preset | ms per frame | Other presets |
| --- | --- | ---: | --- |
| GM20B, Jetson Nano (NVK), fp16 + `F16=1` | `pascal` | 706, 458 with C100 f16 spec | fp16 without `F16=1`: 886 |
| GM20B, Jetson Nano (NVK), fp32 | `turing` | 738, 589 with C100 spec | `pascal` +5 %, `rdna*` +50 %, `valhall4` +60 %, `adreno` wrong output |
| RX 7800 XT (RADV), fp32 | `rdna3` | 8.3 | `turing` 9.6; unpadded `rdna3` 9.6; fp16 + `F16=1` with `pascal` 10.8 |
| Ryzen 7800X3D integrated, 2 CUs (RADV) | `rdna3` | 199 | `turing` 296; unpadded `rdna3` 227 |

  The `adreno` preset's wrong output was measured on NVK, not on Adreno
  hardware. The capabilities the preset assumes do not match the GM20B, so
  check it on a real Adreno before relying on it, and always run `run.sh`'s
  correctness step first.

## Tuning a new GPU

1. Profile one frame with Tracy (`iree-tracy-capture`, a runtime built with
   tracing) and aggregate GPU time per dispatch name. Use the dispatch with
   the fewest occurrences in the capture as the frame count. The dispatch
   names carry the matmul shape (`matmul_MxNxK`), so GFLOPS per family follow
   directly.
2. Dump the compiler's choices with
   `--iree-hal-dump-executable-configurations-to=<dir>` to see the default
   tiles for the slow families.
3. Override one family at a time with `family_spec.py` and keep only the
   rules that win on the board. On the GM20B, the defaults for most families
   were already the best, and applying one configuration to every matmul was
   much slower than the defaults. The gains came from small workgroups with
   an 8x4 output block per thread in f32 (two families) and an 8x8 block in
   f16 (four families), for example:

```sh
python3 family_spec.py spec.mlir "0=400:16,128,32:32,2" "1=32:64,32,16:8,16"
```

   The first rule matching an op wins, so list each family once.
