// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-memory-planned-instr)' %s | FileCheck --check-prefix=PLANNED %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-ddr-memory-planned-instr)' %s | FileCheck --check-prefix=DDR-PLANNED %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-tile-region)' %s | FileCheck --check-prefix=TILE %s

func.func @all_to_all_to_instr(%input: tensor<4x4xf32>,
                               %out: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  %group = wafer.group ins(%input : tensor<4x4xf32>)
      outs(%out : tensor<2x8xf32>) {
  ^bb0(%arg0: tensor<4x4xf32>, %arg1: tensor<2x8xf32>):
    %a2a = wafer.linalg_ext.collective.all_to_all
        ins(%arg0 : tensor<4x4xf32>)
        outs(%arg1 : tensor<2x8xf32>)
        {split_axis = 0 : i64, concat_axis = 1 : i64,
         split_count = 2 : i64, rank_group = array<i64: 0, 1>,
         channel_id = 21 : i64}
        -> tensor<2x8xf32>
    wafer.group.yield %a2a : tensor<2x8xf32>
  } : tensor<2x8xf32>
  return %group : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @all_to_all_to_instr
// CHECK-NOT: wafer.group
// CHECK-NOT: wafer.linalg_ext.collective.all_to_all
// CHECK: wafer.instr.rdma
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.local_fence
// CHECK: %[[SEND:.+]] = wafer.instr.dte_send
// CHECK-SAME: message = #wafer.dte_message<communication = 21, phase = all_to_all, round = 1, slice = 1>
// CHECK-SAME: peer = 1 : i64
// CHECK: %[[RECV:.+]] = wafer.instr.dte_recv
// CHECK-SAME: message = #wafer.dte_message<communication = 21, phase = all_to_all, round = 1, slice = 0>
// CHECK-SAME: peer = 1 : i64
// CHECK: wafer.instr.dte_wait %[[SEND]], %[[RECV]]
// CHECK: wafer.instr.wdma

// PLANNED-LABEL: func.func @all_to_all_to_instr
// PLANNED-NOT: wafer.group
// PLANNED-NOT: wafer.linalg_ext.collective.all_to_all
// PLANNED: wafer.spm.offset
// PLANNED: wafer.instr.local_fence
// PLANNED: wafer.instr.dte_send
// PLANNED: wafer.instr.dte_recv
// PLANNED: wafer.instr.dte_wait
// PLANNED: wafer.instr.wdma

// DDR-PLANNED-LABEL: func.func @all_to_all_to_instr
// DDR-PLANNED-NOT: wafer.group
// DDR-PLANNED-NOT: wafer.linalg_ext.collective.all_to_all
// DDR-PLANNED: wafer.spm.offset
// DDR-PLANNED: wafer.instr.local_fence
// DDR-PLANNED: wafer.instr.dte_send
// DDR-PLANNED: wafer.instr.dte_recv
// DDR-PLANNED: wafer.instr.dte_wait
// DDR-PLANNED: wafer.instr.wdma

// TILE-LABEL: func.func @all_to_all_to_instr
// TILE-NOT: wafer.group
// TILE-NOT: wafer.linalg_ext.collective.all_to_all
// TILE: %[[SEND:.+]] = wafer.instr.dte_send
// TILE-SAME: -> !async.token
// TILE: %[[RECV:.+]] = wafer.instr.dte_recv
// TILE-SAME: -> !async.token
// TILE: wafer.instr.dte_wait %[[SEND]], %[[RECV]]
