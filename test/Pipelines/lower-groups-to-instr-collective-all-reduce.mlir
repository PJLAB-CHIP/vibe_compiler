// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-memory-planned-instr)' %s | FileCheck --check-prefix=PLANNED %s

func.func @collective_all_reduce_to_instr(%input: tensor<4xf32>,
                                          %out: tensor<4xf32>)
    -> tensor<4xf32> {
  %group = wafer.group ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
    %ar = wafer.linalg_ext.collective.all_reduce
        ins(%arg0 : tensor<4xf32>)
        outs(%arg1 : tensor<4xf32>)
        {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          wafer.linalg_ext.collective.yield %sum : f32
        } {rank_group = array<i64: 0, 1>} -> tensor<4xf32>
    wafer.group.yield %ar : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// CHECK-LABEL: func.func @collective_all_reduce_to_instr
// CHECK-NOT: wafer.group
// CHECK-NOT: wafer.linalg_ext.collective.all_reduce
// CHECK-NOT: wafer.tile.all_reduce
// CHECK: %[[INPUT:.+]] = memref.alloc
// CHECK: wafer.instr.rdma {{%.*}} to %[[INPUT]]
// CHECK: %[[RECV_BUF:.+]] = memref.alloc
// CHECK: %[[ACC:.+]] = memref.alloc
// CHECK: %[[FORWARD:.+]] = memref.alloc
// CHECK: wafer.instr.gather_scatter %[[INPUT]] to %[[ACC]]
// CHECK: wafer.instr.gather_scatter %[[INPUT]] to %[[FORWARD]]
// CHECK: wafer.instr.local_drain
// CHECK: %[[SEND:.+]] = wafer.instr.dte_send %[[FORWARD]]
// CHECK-SAME: peer = 1 : i64
// CHECK: %[[RECV:.+]] = wafer.instr.dte_recv %[[RECV_BUF]]
// CHECK-SAME: peer = 1 : i64
// CHECK: wafer.instr.dte_wait %[[SEND]], %[[RECV]]
// CHECK: wafer.instr.elementwise <add> %[[ACC]], %[[RECV_BUF]] into %[[ACC]]
// CHECK: wafer.instr.wdma

// PLANNED-LABEL: func.func @collective_all_reduce_to_instr
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<
// PLANNED: wafer.instr.dte_send
// PLANNED: wafer.instr.dte_recv
// PLANNED: wafer.instr.dte_wait
// PLANNED: wafer.instr.elementwise <add>
