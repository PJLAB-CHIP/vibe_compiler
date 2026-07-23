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

  func.func @all_reduce_ring(%boundary: memref<4xi32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xi32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xi32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xi32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<4xi32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xi32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
          {local_rank = 1 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64,
           communication_id = 16 : i64}
          : (memref<4xi32, #wafer.memory<spm, tensor>>,
             memref<4xi32, #wafer.memory<spm, tensor>>)
         -> memref<4xi32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xi32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// CHECK-LABEL: func.func @all_reduce_ring
// CHECK-SAME: %[[BOUNDARY:[^:]+]]:
// CHECK: %[[INPUT:.+]] = memref.alloc
// CHECK: %[[RECV:.+]] = memref.alloc
// CHECK: %[[ACC:.+]] = memref.alloc
// CHECK: wafer.instr.gather_scatter %[[INPUT]] to %[[ACC]]
// CHECK-SAME: byte_count = 16 : i64
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[ACC1:.+]] = memref.subview %[[ACC]][1] [1] [1]
// CHECK: %[[RECV0BUF:.+]] = memref.subview %[[RECV]][0] [1] [1]
// CHECK: %[[ACC0:.+]] = memref.subview %[[ACC]][0] [1] [1]
// CHECK: %[[SEND0:.+]] = wafer.instr.dte_send %[[ACC1]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 0, slice = 1>
// CHECK-SAME: peer = 2 : i64
// CHECK: %[[RECV0:.+]] = wafer.instr.dte_recv %[[RECV0BUF]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 0, slice = 0>
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND0]], %[[RECV0]]
// CHECK-NEXT: wafer.instr.elementwise <add> %[[ACC0]], %[[RECV0BUF]] into %[[ACC0]]
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[RECV3BUF:.+]] = memref.subview %[[RECV]][3] [1] [1]
// CHECK: %[[ACC3:.+]] = memref.subview %[[ACC]][3] [1] [1]
// CHECK: %[[SEND1:.+]] = wafer.instr.dte_send %[[ACC0]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 1, slice = 0>
// CHECK-SAME: peer = 2 : i64
// CHECK: %[[RECV1:.+]] = wafer.instr.dte_recv %[[RECV3BUF]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 1, slice = 3>
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND1]], %[[RECV1]]
// CHECK-NEXT: wafer.instr.elementwise <add> %[[ACC3]], %[[RECV3BUF]] into %[[ACC3]]
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[RECV2BUF:.+]] = memref.subview %[[RECV]][2] [1] [1]
// CHECK: %[[ACC2:.+]] = memref.subview %[[ACC]][2] [1] [1]
// CHECK: %[[SEND2:.+]] = wafer.instr.dte_send %[[ACC3]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 2, slice = 3>
// CHECK-SAME: peer = 2 : i64
// CHECK: %[[RECV2:.+]] = wafer.instr.dte_recv %[[RECV2BUF]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 2, slice = 2>
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND2]], %[[RECV2]]
// CHECK-NEXT: wafer.instr.elementwise <add> %[[ACC2]], %[[RECV2BUF]] into %[[ACC2]]
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[RECV1BUF:.+]] = memref.subview %[[RECV]][1] [1] [1]
// CHECK: %[[SEND3:.+]] = wafer.instr.dte_send %[[ACC2]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 3, slice = 2>
// CHECK-SAME: peer = 2 : i64
// CHECK: %[[RECV3:.+]] = wafer.instr.dte_recv %[[RECV1BUF]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 3, slice = 1>
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND3]], %[[RECV3]]
// CHECK-NEXT: wafer.instr.gather_scatter %[[RECV1BUF]] to %[[ACC1]]
// CHECK-SAME: byte_count = 4 : i64
// CHECK-SAME: inner_bytes = 4 : i64
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[SEND4:.+]] = wafer.instr.dte_send %[[ACC1]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 4, slice = 1>
// CHECK-SAME: peer = 2 : i64
// CHECK: %[[RECV4:.+]] = wafer.instr.dte_recv %[[RECV0BUF]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 4, slice = 0>
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND4]], %[[RECV4]]
// CHECK-NEXT: wafer.instr.gather_scatter %[[RECV0BUF]] to %[[ACC0]]
// CHECK-SAME: byte_count = 4 : i64
// CHECK-SAME: inner_bytes = 4 : i64
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[SEND5:.+]] = wafer.instr.dte_send %[[ACC0]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 5, slice = 0>
// CHECK-SAME: peer = 2 : i64
// CHECK: %[[RECV5:.+]] = wafer.instr.dte_recv %[[RECV3BUF]]
// CHECK-SAME: bytes = 4 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 16, phase = all_reduce_ring, round = 5, slice = 3>
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait %[[SEND5]], %[[RECV5]]
// CHECK-NEXT: wafer.instr.gather_scatter %[[RECV3BUF]] to %[[ACC3]]
// CHECK-SAME: byte_count = 4 : i64
// CHECK-SAME: inner_bytes = 4 : i64
// CHECK-NEXT: wafer.instr.local_fence
// CHECK-NEXT: wafer.instr.local_fence
// CHECK-NOT: wafer.instr.elementwise <add>
// CHECK-NOT: wafer.instr.dte_send
// CHECK-NOT: wafer.instr.dte_recv
// CHECK-NOT: wafer.tile.all_reduce

// SPM-LABEL: func.func @all_reduce_ring
// SPM: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<
// SPM: wafer.instr.dte_send
// SPM: wafer.instr.dte_recv
// SPM: wafer.instr.dte_wait
// SPM: wafer.instr.elementwise <add>
