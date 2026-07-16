// RUN: wafer-opt --wafer-convert-tile-region-to-instr='all-gather-schedule=direct' %s | FileCheck %s
// RUN: wafer-opt --wafer-convert-tile-region-to-instr='all-gather-schedule=direct' %s \
// RUN:   | wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66816' \
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

  func.func @all_gather_direct(%boundary: memref<16xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<16xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<16xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<16xf32, #wafer.memory<ddr, tensor>>):
      %local = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %gather = memref.alloc()
          : memref<16xf32, #wafer.memory<spm, tensor>>
      wafer.tile.all_gather %local into %gather
          {local_rank = 1 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64,
           communication_id = 11 : i64}
          : memref<4xf32, #wafer.memory<spm, tensor>>
         -> memref<16xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<16xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// CHECK-LABEL: func.func @all_gather_direct
// CHECK: %[[LOCAL:.+]] = memref.alloc
// CHECK: %[[GATHER:.+]] = memref.alloc
// CHECK: %[[SLOT1:.+]] = memref.subview %[[GATHER]][4] [4] [1]
// CHECK: %[[LOCAL_COMM:.+]] = memref.alloc
// CHECK: wafer.instr.gather_scatter %[[LOCAL]] to %[[LOCAL_COMM]]
// CHECK-SAME: byte_count = 16 : i64
// CHECK: wafer.instr.gather_scatter %[[LOCAL_COMM]] to %[[SLOT1]]
// CHECK-SAME: byte_count = 16 : i64
// CHECK: wafer.instr.local_fence
// CHECK: %[[SLOT0:.+]] = memref.subview %[[GATHER]][0] [4] [1]
// CHECK: %[[RECV0_BUF:.+]] = memref.alloc
// CHECK: %[[SEND0:.+]] = wafer.instr.dte_send %[[LOCAL_COMM]]
// CHECK-SAME: message = #wafer.dte_message<communication = 11, phase = all_gather_direct, round = 1, slice = 1>
// CHECK-SAME: peer = 2 : i64
// CHECK: %[[RECV0:.+]] = wafer.instr.dte_recv %[[RECV0_BUF]]
// CHECK-SAME: message = #wafer.dte_message<communication = 11, phase = all_gather_direct, round = 1, slice = 0>
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND0]], %[[RECV0]]
// CHECK: wafer.instr.gather_scatter %[[RECV0_BUF]] to %[[SLOT0]]
// CHECK: %[[SLOT3:.+]] = memref.subview %[[GATHER]][12] [4] [1]
// CHECK: %[[RECV1_BUF:.+]] = memref.alloc
// CHECK: %[[SEND1:.+]] = wafer.instr.dte_send %[[LOCAL_COMM]]
// CHECK-SAME: peer = 3 : i64
// CHECK: %[[RECV1:.+]] = wafer.instr.dte_recv %[[RECV1_BUF]]
// CHECK-SAME: peer = 3 : i64
// CHECK: wafer.instr.dte_wait %[[SEND1]], %[[RECV1]]
// CHECK: wafer.instr.gather_scatter %[[RECV1_BUF]] to %[[SLOT3]]
// CHECK: %[[SLOT2:.+]] = memref.subview %[[GATHER]][8] [4] [1]
// CHECK: %[[RECV2_BUF:.+]] = memref.alloc
// CHECK: %[[SEND2:.+]] = wafer.instr.dte_send %[[LOCAL_COMM]]
// CHECK-SAME: peer = 0 : i64
// CHECK: %[[RECV2:.+]] = wafer.instr.dte_recv %[[RECV2_BUF]]
// CHECK-SAME: peer = 2 : i64
// CHECK: wafer.instr.dte_wait %[[SEND2]], %[[RECV2]]
// CHECK: wafer.instr.gather_scatter %[[RECV2_BUF]] to %[[SLOT2]]
// CHECK-NEXT: wafer.instr.local_fence
// CHECK-NOT: wafer.tile.all_gather

// SPM-LABEL: func.func @all_gather_direct
// SPM: wafer.instr.dte_send
// SPM: wafer.instr.dte_recv
// SPM: wafer.instr.dte_wait
