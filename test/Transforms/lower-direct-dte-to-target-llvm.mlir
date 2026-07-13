// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{logical-rank=0 transport-status-argument-index=0})' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{logical-rank=0 transport-status-argument-index=0})' %s | mlir-translate --mlir-to-llvmir | FileCheck --check-prefix=LLVMIR %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}

  func.func @main(%status: i64) {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %send = wafer.instr.dte_send %src
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = collective_permute, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_receiver_offset = 65792, completion = sender_wait_receiver_fsm>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %recv = wafer.instr.dte_recv %dst
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 10, phase = collective_permute, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_receiver_offset = 65792, completion = sender_wait_receiver_fsm>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %send, %recv : !async.token, !async.token
    return
  }
}

// CHECK-LABEL: llvm.func @main
// CHECK: llvm.call @wafer_tx81_direct_dte_begin
// CHECK: %[[SEND:.*]] = llvm.call @wafer_tx81_direct_dte_send_prepare
// CHECK-SAME: : (i64, i64, i32, i32, i32, i32, i32) -> i64
// CHECK: %[[RECV:.*]] = llvm.call @wafer_tx81_direct_dte_recv_prepare
// CHECK-SAME: : (i64, i32, i32, i32, i32) -> i64
// CHECK: llvm.call @wafer_tx81_direct_dte_wait(%[[SEND]])
// CHECK: llvm.call @wafer_tx81_direct_dte_wait(%[[RECV]])
// CHECK: llvm.call @wafer_tx81_direct_dte_finish
// CHECK: llvm.return

// LLVMIR: call void @wafer_tx81_direct_dte_begin
// LLVMIR: call i64 @wafer_tx81_direct_dte_send_prepare
// LLVMIR: call i64 @wafer_tx81_direct_dte_recv_prepare
// LLVMIR: call void @wafer_tx81_direct_dte_wait
// LLVMIR: call void @wafer_tx81_direct_dte_finish
