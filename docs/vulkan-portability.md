# Running the Vulkan work on another GPU

The Vulkan workloads in this repository are vendor-neutral. The C100
(GM20B, NVK) is the board they were tuned on, but nothing outside
`../c100_gpu` assumes it. This page lists what carries over to another
Vulkan board unchanged, what was tuned for the GM20B and has to be measured
again, and the order to bring a new board up in.

## Carries over unchanged

| Asset | Location | Requirements |
| --- | --- | --- |
| NxIREE compiler and runtime | `lib/`, `c_src/` | Vulkan 1.1 compute; `--iree-vulkan-target=<preset>` per device |
| YOLOv5 recipe and tools | `bench/yolo/` | fp32 export; correctness checked against OpenCV DNN |
| Census SGBM graph and shaders | `bench/stereo/census/` | `GL_KHR_shader_subgroup_shuffle` and `_shuffle_relative` in compute, subgroup size 16 to 128 |
| OpenCV `vkstereo` module | `../opencv_vkstereo` | the same shaders, any Vulkan 1.1 device; `VKSTEREO_DEVICE` picks one |
| Matmul spec generator | `bench/tuning/matmul_spec.py`, `bench/yolo/family_spec.py` | IREE 3.13 spec syntax |

The stereo shaders shuffle only within aligned groups of
`DISPARITIES / VIDEO_DP_BLOCK` lanes (8 at D = 64) inside 128-wide
workgroups, so any subgroup size that is a power of two between 16 and 128
works. With D = 128 and the default block, the group is 16 lanes.

## Tuned for the GM20B, measure again

* **IREE target preset.** `turing` was fastest for YOLO on the GM20B and
  `rdna3` on an RX 7800 XT. Try every preset that compiles, starting with the
  one named after the GPU's family, and keep the fastest correct one.
  `run.sh`'s correctness step catches presets whose assumed capabilities the
  device lacks.
* **Tuning specs.** `c100_gpu/priv/tuning/*.mlir` are GM20B numbers. Rebuild
  them from a profile with `family_spec.py`, following `bench/yolo/README.md`.
* **Stereo layout parameters.** `VIDEO_DP_BLOCK`, `VIDEO_SELECT_LANES`, the
  row-tiled volume (`volume_tile`) and the fusion mode were chosen from GM20B
  memory-transaction measurements. Re-sweep them; a GPU with larger caches
  or wider memory transactions may prefer other values.
* **Allocator and transfers.** Host-cached memory with explicit
  flush/invalidate was a large win on the Jetson's shared memory. Check
  which memory types the new device exposes before assuming the same.

## Bringing a new board up

1. `vulkaninfo`: record the driver, the subgroup size and supported stages,
   the shared-memory limit and the memory types.
2. Run the `vkstereo` test and benchmark from `../opencv_vkstereo`. They
   need no IREE compiler and check bit-exact output, so they validate the
   driver's subgroup and storage-buffer behaviour first.
3. Compile and check YOLO with `bench/yolo` for each candidate preset.
4. Profile with Tracy, then write device tuning in a library of the board's
   own, as `../c100_gpu` does for the C100.
