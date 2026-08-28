// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"],
       shape = array<i64: 1>}

  func.func @peer_tiles(
      %boundary: memref<8xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %boundary : memref<8xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<8xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<8xf32, #wafer.memory<ddr, tensor>>):
      %send_buffer = memref.alloc()
          : memref<8xf32, #wafer.memory<spm, tensor>>
      %recv_buffer = memref.alloc()
          : memref<8xf32, #wafer.memory<spm, tensor>>
      %recv = wafer.tile.peer_recv %recv_buffer
          {peer = 1 : i64, bytes = 32 : i64,
           message = #wafer.dte_message<communication = 41, round = 0, slice = 3>}
          : memref<8xf32, #wafer.memory<spm, tensor>> -> !async.token
      %send = wafer.tile.peer_send %send_buffer
          {peer = 1 : i64, bytes = 32 : i64,
           message = #wafer.dte_message<communication = 42, round = 0, slice = 3>}
          : memref<8xf32, #wafer.memory<spm, tensor>> -> !async.token
      async.await %recv : !async.token
      async.await %send : !async.token
      wafer.tile.yield %arg0
          : memref<8xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// CHECK-LABEL: func.func @peer_tiles
// CHECK: %[[SEND_BUFFER:.+]] = memref.alloc
// CHECK: %[[RECV_BUFFER:.+]] = memref.alloc
// CHECK: %[[RECV:.+]] = wafer.instr.dte_recv %[[RECV_BUFFER]]
// CHECK-SAME: message = #wafer.dte_message<communication = 41, round = 0, slice = 3>
// CHECK-SAME: peer = 1 : i64
// CHECK: %[[SEND:.+]] = wafer.instr.dte_send %[[SEND_BUFFER]]
// CHECK-SAME: message = #wafer.dte_message<communication = 42, round = 0, slice = 3>
// CHECK-SAME: peer = 1 : i64
// CHECK: wafer.instr.dte_wait %[[RECV]]
// CHECK: wafer.instr.dte_wait %[[SEND]]
// CHECK-NOT: wafer.tile.peer_
// CHECK-NOT: async.await
