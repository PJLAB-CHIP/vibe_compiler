// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s
// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s \
// RUN:   | wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=74752' \
// RUN:   | FileCheck --check-prefix=SPM %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 16>,
       policy = "all_available",
       endpoints = array<i64>}

  func.func @all_gather_ring_inner_axis(
      %boundary: memref<16x64xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<16x64xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<16x64xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<16x64xf32, #wafer.memory<ddr, tensor>>):
      %local = memref.alloc()
          : memref<16x4xf32, #wafer.memory<spm, tensor>>
      %gather = memref.alloc()
          : memref<16x64xf32, #wafer.memory<spm, tensor>>
      wafer.tile.all_gather %local into %gather
          {local_rank = 14 : i64, group_size = 16 : i64,
           communication_id = 14 : i64,
           rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                                  8, 9, 10, 11, 12, 13, 14, 15>,
           bytes = 256 : i64}
          : memref<16x4xf32, #wafer.memory<spm, tensor>>
         -> memref<16x64xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<16x64xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// CHECK-LABEL: func.func @all_gather_ring_inner_axis
// CHECK: %[[LOCAL:.+]] = memref.alloc
// CHECK-SAME: memref<16x4xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[GATHER:.+]] = memref.alloc
// CHECK-SAME: memref<16x64xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[LOCAL_SLOT:.+]] = memref.subview %[[GATHER]][0, 56] [16, 4] [1, 1]
// CHECK-SAME: memref<16x4xf32, strided<[64, 1], offset: 56>, #wafer.memory<spm, tensor>>
// CHECK: %[[LOCAL_COMM:.+]] = memref.alloc
// CHECK-SAME: memref<16x4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[LOCAL]] to %[[LOCAL_COMM]]
// CHECK-SAME: inner_bytes = 256 : i64
// CHECK: wafer.instr.gather_scatter %[[LOCAL_COMM]] to %[[LOCAL_SLOT]]
// CHECK-SAME: dst_strides = array<i64: 256, 0, 0>
// CHECK-SAME: inner_bytes = 16 : i64
// CHECK: %[[PEER_SLOT:.+]] = memref.subview %[[GATHER]][0, 60] [16, 4] [1, 1]
// CHECK-SAME: memref<16x4xf32, strided<[64, 1], offset: 60>, #wafer.memory<spm, tensor>>
// CHECK: %[[RECV_BUF:.+]] = memref.alloc
// CHECK-SAME: memref<16x4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.ncc_join [0]
// CHECK: %[[SEND:.+]] = wafer.instr.dte_send %[[LOCAL_COMM]]
// CHECK-SAME: bytes = 256 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 14, phase = all_gather_ring, round = 0, slice = 14>
// CHECK-SAME: peer = 13 : i64
// CHECK: %[[RECV:.+]] = wafer.instr.dte_recv %[[RECV_BUF]]
// CHECK-SAME: bytes = 256 : i64
// CHECK-SAME: message = #wafer.dte_message<communication = 14, phase = all_gather_ring, round = 0, slice = 15>
// CHECK-SAME: peer = 15 : i64
// CHECK: wafer.instr.dte_wait %[[SEND]], %[[RECV]]
// CHECK: wafer.instr.gather_scatter %[[RECV_BUF]] to %[[PEER_SLOT]]
// CHECK-SAME: dst_strides = array<i64: 256, 0, 0>
// CHECK-SAME: inner_bytes = 16 : i64
// CHECK: wafer.instr.ncc_join [0]
// CHECK-NOT: wafer.instr.local_fence
// CHECK-NOT: wafer.tile.all_gather

// SPM-LABEL: func.func @all_gather_ring_inner_axis
// SPM: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<
// SPM-SAME: memref<16x4xf32, #wafer.memory<spm, tensor>>
// SPM: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<
// SPM-SAME: memref<16x64xf32, #wafer.memory<spm, tensor>>
// SPM: wafer.instr.dte_send
// SPM: wafer.instr.dte_recv
// SPM: wafer.instr.gather_scatter
