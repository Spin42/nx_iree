// Standalone IREE reproducer for wrong masked attention scores on Vulkan
// (IREE 3.13, default `vp_android_baseline_2022` target, any device).
//
// The key-mask bias reaches the batch matmul as a separate i8 tensor. The
// SPIR-V backend vectorizes its load into `memref<2xvector<4xi8>>`, the
// narrow-type emulation does not handle that `vector.transfer_read`, and
// ConvertToSPIRV then widens `vector<4xi8>` to `vector<4xi32>` because the
// target has no 8-bit storage: each mask byte is read 4 bytes apart.
//
//   iree-compile --iree-hal-target-backends=vulkan-spirv \
//     --iree-input-type=stablehlo_xla vulkan_i8_mask_repro.mlir -o repro.vmfb
//   iree-run-module --module=repro.vmfb --device=vulkan \
//     --input=1x4x8x8xf32=0 --input=1x4x8x8xf32=0 "--input=8xi8=[1 1 1 1 1 0 0 0]"
//
//   expected (llvm-cpu, or with 8-bit storage enabled):
//     [0 0 0 0 0 -3.40282E+38 -3.40282E+38 -3.40282E+38] ...
//   got:
//     [0 0 -3.40282E+38 -3.40282E+38 -3.40282E+38 -3.40282E+38 ...] ...
//
// Passes with --iree-vulkan-target=rdna3 / llvmpipe, or with
// --iree-vulkan-target-features=spirv:v1.3,cap:Shader,cap:StorageBuffer8BitAccess,cap:Int8,ext:SPV_KHR_8bit_storage
// and with an i32 mask instead of i8.
module {
  func.func public @main(%q: tensor<1x4x8x8xf32>, %k: tensor<1x4x8x8xf32>, %mask: tensor<8xi8>) -> tensor<1x4x8x8xf32> {
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %min = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %c0 = stablehlo.constant dense<0> : tensor<i8>
    %0 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x8x8xf32>
    %2 = stablehlo.multiply %1, %0 : tensor<1x4x8x8xf32>
    %3 = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<i8>) -> tensor<8xi8>
    %4 = stablehlo.compare NE, %mask, %3 : (tensor<8xi8>, tensor<8xi8>) -> tensor<8xi1>
    %5 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %6 = stablehlo.broadcast_in_dim %min, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.select %4, %5, %6 : tensor<8xi1>, tensor<8xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [3] : (tensor<8xf32>) -> tensor<1x4x8x8xf32>
    %9 = stablehlo.add %2, %8 : tensor<1x4x8x8xf32>
    return %9 : tensor<1x4x8x8xf32>
  }
}
