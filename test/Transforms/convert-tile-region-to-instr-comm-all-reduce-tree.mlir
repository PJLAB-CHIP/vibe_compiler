// RUN: wafer-opt --wafer-convert-tile-region-to-instr='all-reduce-schedule=tree' %s | FileCheck %s
// RUN: wafer-opt --wafer-convert-tile-region-to-instr='all-reduce-schedule=tree' %s \
// RUN:   | wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66560' \
// RUN:   | FileCheck --check-prefix=SPM %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 4>,
       policy = "all_available",
       endpoints = array<i64>}

  func.func @all_reduce_tree_rank_two(
      %boundary: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
          {local_rank = 2 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64}
          : (memref<4xf32, #wafer.memory<spm, tensor>>,
             memref<4xf32, #wafer.memory<spm, tensor>>)
         -> memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// CHECK-LABEL: func.func @all_reduce_tree_rank_two
// CHECK: %[[INPUT:.+]] = memref.alloc
// CHECK: %[[RECV:.+]] = memref.alloc
// CHECK: %[[ACC:.+]] = memref.alloc
// CHECK: wafer.instr.gather_scatter %[[INPUT]] to %[[ACC]]
// CHECK-SAME: byte_count = 16 : i64
// CHECK: wafer.instr.local_drain
// CHECK: %[[RECV_CHILD:.+]] = wafer.instr.dte_recv %[[RECV]]
// CHECK-SAME: peer = 3 : i64
// CHECK: wafer.instr.dte_wait %[[RECV_CHILD]]
// CHECK: wafer.instr.elementwise <add> %[[ACC]], %[[RECV]] into %[[ACC]]
// CHECK: wafer.instr.local_drain
// CHECK: %[[SEND_PARENT:.+]] = wafer.instr.dte_send %[[ACC]]
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND_PARENT]]
// CHECK: %[[RECV_FINAL:.+]] = wafer.instr.dte_recv %[[ACC]]
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[RECV_FINAL]]
// CHECK: %[[SEND_CHILD:.+]] = wafer.instr.dte_send %[[ACC]]
// CHECK-SAME: peer = 3 : i64
// CHECK: wafer.instr.dte_wait %[[SEND_CHILD]]
// CHECK-NOT: wafer.tile.all_reduce

// SPM-LABEL: func.func @all_reduce_tree_rank_two
// SPM: wafer.instr.dte_recv
// SPM: wafer.instr.dte_wait
// SPM: wafer.instr.elementwise <add>
// SPM: wafer.instr.dte_send
