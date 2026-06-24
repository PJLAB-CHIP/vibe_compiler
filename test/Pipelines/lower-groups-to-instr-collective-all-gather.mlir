// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-memory-planned-instr)' %s | FileCheck --check-prefix=PLANNED %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-ddr-memory-planned-instr)' %s | FileCheck --check-prefix=DDR-PLANNED %s

func.func @collective_all_gather_to_instr(%input: tensor<4xf32>,
                                          %out: tensor<8xf32>)
    -> tensor<8xf32> {
  %group = wafer.group ins(%input : tensor<4xf32>)
      outs(%out : tensor<8xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<8xf32>):
    %ag = wafer.linalg_ext.collective.all_gather
        ins(%arg0 : tensor<4xf32>)
        outs(%arg1 : tensor<8xf32>)
        {axis = 0 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<8xf32>
    wafer.group.yield %ag : tensor<8xf32>
  } : tensor<8xf32>
  return %group : tensor<8xf32>
}

// CHECK-LABEL: func.func @collective_all_gather_to_instr
// CHECK-NOT: wafer.group
// CHECK-NOT: wafer.linalg_ext.collective.all_gather
// CHECK-NOT: wafer.tile.all_gather
// CHECK: %[[INPUT:.+]] = memref.alloc
// CHECK: wafer.instr.rdma {{%.*}} to %[[INPUT]]
// CHECK: %[[GATHER:.+]] = memref.alloc
// CHECK: %[[LOCAL_SLOT:.+]] = memref.subview %[[GATHER]][0] [4] [1]
// CHECK: wafer.instr.gather_scatter %[[INPUT]] to %[[LOCAL_SLOT]]
// CHECK: wafer.instr.local_fence
// CHECK: %[[PEER_SLOT:.+]] = memref.subview %[[GATHER]][4] [4] [1]
// CHECK: %[[SEND:.+]] = wafer.instr.dte_send %[[LOCAL_SLOT]]
// CHECK-SAME: peer = 1 : i64
// CHECK: %[[RECV:.+]] = wafer.instr.dte_recv %[[PEER_SLOT]]
// CHECK-SAME: peer = 1 : i64
// CHECK: wafer.instr.dte_wait %[[SEND]], %[[RECV]]
// CHECK: wafer.instr.wdma %[[GATHER]]

// PLANNED-LABEL: func.func @collective_all_gather_to_instr
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<
// PLANNED: wafer.instr.local_fence
// PLANNED: wafer.instr.dte_send
// PLANNED: wafer.instr.dte_recv
// PLANNED: wafer.instr.dte_wait
// PLANNED: wafer.instr.wdma

// DDR-PLANNED-LABEL: func.func @collective_all_gather_to_instr
// DDR-PLANNED-NOT: wafer.group
// DDR-PLANNED-NOT: wafer.tile.all_gather
// DDR-PLANNED: wafer.spm.offset
// DDR-PLANNED: wafer.instr.local_fence
// DDR-PLANNED: wafer.instr.dte_send
// DDR-PLANNED: wafer.instr.dte_recv
// DDR-PLANNED: wafer.instr.dte_wait
// DDR-PLANNED: wafer.instr.wdma
