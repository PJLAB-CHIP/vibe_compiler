// RUN: not wafer-opt --mlir-disable-threading --pass-pipeline='builtin.module(wafer-lower-groups-to-target-llvm)' --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 | FileCheck %s --implicit-check-not=llvm.func --implicit-check-not=llvm.call --implicit-check-not=wafer.group --implicit-check-not='tensor<' --implicit-check-not=bufferization.to_

module {
  func.func @selected_before_late_failure(
      %lhs: tensor<4xf32>, %rhs: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %group = wafer.group ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>,
         %arg2: tensor<4xf32>):
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
          outs(%arg2 : tensor<4xf32>) {
        ^bb0(%left: f32, %right: f32, %old: f32):
          %value = arith.addf %left, %right : f32
          linalg.yield %value : f32
        } -> tensor<4xf32>
      wafer.group.yield %sum : tensor<4xf32>
    } : tensor<4xf32>
    return %group : tensor<4xf32>
  }

  func.func @late_unsupported_target_kind() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
    wafer.instr.tdma_data_move #wafer.instr_data_move_kind<transpose> %src into %dst
        {source_shape = array<i64: 1, 1, 2, 3>,
         dest_shape = array<i64: 1, 1, 3, 2>,
         permutation = array<i64: 0, 1, 3, 2>}
        : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
       to memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
    return
  }
}

// CHECK: unsupported_target_instr: transform-like tdma_data_move kind reached target LLVM lowering
// CHECK: IR Dump After LowerInstrToTargetLLVMPass Failed
// CHECK: module {
// CHECK-LABEL: func.func @selected_before_late_failure
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.wdma
// CHECK-LABEL: func.func @late_unsupported_target_kind
// CHECK: wafer.instr.tdma_data_move <transpose>
