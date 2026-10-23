# YOLOv5 on IREE's Vulkan backend

Vendor-neutral recipe and tools for running an ultralytics YOLOv5 ONNX
export (640x640) on any Vulkan GPU through IREE 3.13. Device-specific
tuning lives with the device's own library (for example
`c100_gpu/priv/tuning/yolov5s_maxwell.mlir`); everything here works without it.

| File | Purpose |
| --- | --- |
| `to_fp32.py` | converts an fp16 export to fp32 |
| `pad_yolov5_onnx.py` | pads the stem to 4 input channels and the detect heads to 256 output channels; outputs unchanged |
| `compile.sh` | compiles with im2col convolutions and channels-last layout for a `--iree-vulkan-target` preset, optionally with a tuning spec |
| `run.sh` | checks the output against a reference within 0.05, then times it with `iree-benchmark-module` |
| `family_spec.py` | writes a tuning spec that overrides selected matmul or generic op families only |
| `reference/` | `yolo_dump`: the input blob and OpenCV DNN's output for an image, the independent reference |

## Steps

```sh
python3 to_fp32.py yolov5s.onnx yolov5s-fp32.onnx           # only for fp16 exports
python3 pad_yolov5_onnx.py yolov5s-fp32.onnx yolov5s-padded.onnx
iree-import-onnx yolov5s-padded.onnx -o yolov5s.mlir
./compile.sh yolov5s.mlir yolov5s.vmfb <preset> [spec.mlir]
yolo_dump yolov5s-fp32.onnx bus.jpg bus-blob.bin bus-expected.bin   # once, on any machine
./run.sh yolov5s.vmfb bus-blob.bin bus-expected.bin [vulkan://<n>]
```

The blob and reference are device-independent, so they can be produced on
the development host and copied to the board with the module.

## Why each step

* **im2col and channels-last are required for correctness.** IREE's direct
  SPIR-V lowering of NCHW 3x3 convolutions writes values to the wrong
  positions (first visible at `/model.2/m/m.0/cv2/conv`). This reproduces on
  RADV and on NVK, with IREE 3.11 and 3.13, so it is a compiler bug, not a
  driver one. Channels-last alone fixes correctness but the direct
  convolutions remain slow (about 300 ms per layer on a Maxwell GPU).
* **fp32.** The fp16 export exceeds 48 KB of workgroup memory in several
  dispatches; devices with more shared memory may accept it, so try fp16
  again on a new GPU.
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
| GM20B, Jetson Nano (NVK) | `turing` | 738, 589 with C100 spec | `pascal` +5 %, `rdna*` +50 %, `valhall4` +60 %, `adreno` wrong output |
| RX 7800 XT (RADV) | `rdna3` | 200 | `turing` 295; unpadded `rdna3` 224 |

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
   much slower than the defaults. The gains came from two families with an
   8x4 output block per thread on small workgroups:

```sh
python3 family_spec.py spec.mlir "0=400:16,128,32:32,2" "1=32:64,32,16:8,16"
```

   The first rule matching an op wins, so list each family once.
