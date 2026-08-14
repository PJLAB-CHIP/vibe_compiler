// RUN: split-file %s %t
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{physical-card-id=0 physical-tile-id=0 transport-status-argument-index=0})' %t/missing-binding.mlir 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{physical-card-id=0 physical-tile-id=0 transport-status-argument-index=0})' %t/missing-selector.mlir 2>&1 | FileCheck %s --check-prefix=SELECTOR

//--- missing-binding.mlir
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["rank"], shape = array<i64: 16>}
  func.func @main(%status: i64) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
}

// MISSING: unsupported_target_transport: Direct DTE send is missing an accepted physical binding

//--- missing-selector.mlir
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["rank"], shape = array<i64: 16>}
  func.func @main(%status: i64) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = selector_table, remote_receiver_address = 65792, route_bindings = [0, 65792, 0, 1, 66048, 1], completion = sender_wait_receiver_fsm>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
}

// SELECTOR: Direct DTE selector-table binding requires route selector operand
