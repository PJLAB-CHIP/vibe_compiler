// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s
// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s \
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

  func.func @reduce_scatter_phases(
      %boundary: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<16xf32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.reduce_scatter #wafer.reduce_kind<sum> %input using %recv
          {axis = 0 : i64, local_rank = 1 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64,
           communication_id = 18 : i64}
          : (memref<16xf32, #wafer.memory<spm, tensor>>,
             memref<4xf32, #wafer.memory<spm, tensor>>)
         -> memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// CHECK-LABEL: func.func @reduce_scatter_phases
// CHECK-SAME: %[[BOUNDARY:[^:]+]]:
// CHECK: %[[INPUT:.+]] = memref.alloc
// CHECK: %[[RECV:.+]] = memref.alloc
// CHECK: %[[ACC:.+]] = memref.alloc
// CHECK: %[[LOCAL_SLOT:.+]] = memref.subview %[[INPUT]][4] [4] [1]
// CHECK: wafer.instr.gather_scatter %[[LOCAL_SLOT]] to %[[ACC]]
// CHECK-SAME: byte_count = 16 : i64
// CHECK: wafer.instr.local_fence
// CHECK: %[[SLOT2:.+]] = memref.subview %[[INPUT]][8] [4] [1]
// CHECK: %[[SEND0:.+]] = wafer.instr.dte_send %[[SLOT2]]
// CHECK-SAME: message = #wafer.dte_message<communication = 18, phase = reduce_scatter_direct, round = 1, slice = 2>
// CHECK-SAME: peer = 2 : i64
// CHECK: %[[RECV0:.+]] = wafer.instr.dte_recv %[[RECV]]
// CHECK-SAME: message = #wafer.dte_message<communication = 18, phase = reduce_scatter_direct, round = 1, slice = 1>
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND0]], %[[RECV0]]
// CHECK: wafer.instr.elementwise <add> %[[ACC]], %[[RECV]] into %[[ACC]]
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[SLOT3:.+]] = memref.subview %[[INPUT]][12] [4] [1]
// CHECK: %[[SEND1:.+]] = wafer.instr.dte_send %[[SLOT3]]
// CHECK-SAME: peer = 3 : i64
// CHECK: %[[RECV1:.+]] = wafer.instr.dte_recv %[[RECV]]
// CHECK-SAME: peer = 3 : i64
// CHECK: wafer.instr.dte_wait %[[SEND1]], %[[RECV1]]
// CHECK: wafer.instr.elementwise <add> %[[ACC]], %[[RECV]] into %[[ACC]]
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[SLOT0:.+]] = memref.subview %[[INPUT]][0] [4] [1]
// CHECK: %[[SEND2:.+]] = wafer.instr.dte_send %[[SLOT0]]
// CHECK-SAME: peer = 0 : i64
// CHECK: %[[RECV2:.+]] = wafer.instr.dte_recv %[[RECV]]
// CHECK-SAME: peer = 2 : i64
// CHECK: wafer.instr.dte_wait %[[SEND2]], %[[RECV2]]
// CHECK: wafer.instr.elementwise <add> %[[ACC]], %[[RECV]] into %[[ACC]]
// CHECK-NEXT: wafer.instr.local_fence
// CHECK-NOT: wafer.tile.reduce_scatter

// SPM-LABEL: func.func @reduce_scatter_phases
// SPM: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<
// SPM: wafer.instr.dte_send
// SPM: wafer.instr.dte_recv
// SPM: wafer.instr.dte_wait
// SPM: wafer.instr.elementwise <add>
