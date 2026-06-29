// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-memory-planned-instr)' %s | FileCheck --check-prefix=PLANNED %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-ddr-memory-planned-instr)' %s | FileCheck --check-prefix=DDR-PLANNED %s

func.func @collective_permute_to_instr(%input: tensor<4xf32>,
                                       %out: tensor<4xf32>)
    -> tensor<4xf32> {
  %group = wafer.group ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
    %cp = wafer.linalg_ext.collective.collective_permute
        ins(%arg0 : tensor<4xf32>)
        outs(%arg1 : tensor<4xf32>)
        {source_target_pairs = array<i64: 0, 1, 2, 0>}
        -> tensor<4xf32>
    wafer.group.yield %cp : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// CHECK-LABEL: func.func @collective_permute_to_instr
// CHECK-NOT: wafer.group
// CHECK-NOT: wafer.linalg_ext.collective.collective_permute
// CHECK: %[[INPUT:.+]] = memref.alloc
// CHECK: wafer.instr.rdma {{%.*}} to %[[INPUT]]
// CHECK: %[[RESULT:.+]] = memref.alloc
// CHECK: wafer.instr.fill %[[RESULT]]
// CHECK: %[[SEND:.+]] = wafer.instr.dte_send %[[INPUT]]
// CHECK-SAME: peer = 1 : i64
// CHECK: %[[RECV:.+]] = wafer.instr.dte_recv %[[RESULT]]
// CHECK-SAME: peer = 2 : i64
// CHECK: wafer.instr.dte_wait %[[SEND]], %[[RECV]]
// CHECK: wafer.instr.wdma %[[RESULT]]

// PLANNED-LABEL: func.func @collective_permute_to_instr
// PLANNED-NOT: wafer.group
// PLANNED-NOT: wafer.linalg_ext.collective.collective_permute
// PLANNED: wafer.spm.offset
// PLANNED: wafer.instr.fill
// PLANNED: wafer.instr.dte_send
// PLANNED: wafer.instr.dte_recv
// PLANNED: wafer.instr.dte_wait
// PLANNED: wafer.instr.wdma

// DDR-PLANNED-LABEL: func.func @collective_permute_to_instr
// DDR-PLANNED-NOT: wafer.group
// DDR-PLANNED-NOT: wafer.linalg_ext.collective.collective_permute
// DDR-PLANNED: wafer.spm.offset
// DDR-PLANNED: wafer.instr.fill
// DDR-PLANNED: wafer.instr.dte_send
// DDR-PLANNED: wafer.instr.dte_recv
// DDR-PLANNED: wafer.instr.dte_wait
// DDR-PLANNED: wafer.instr.wdma
