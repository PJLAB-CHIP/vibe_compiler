// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-selected-instr)' %s | FileCheck --check-prefix=SELECTED --implicit-check-not='tensor<' --implicit-check-not=bufferization.to_ --implicit-check-not=wafer.group %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-target-llvm)' %s | FileCheck --check-prefix=TARGET --implicit-check-not=wafer. --implicit-check-not=memref. --implicit-check-not=func.func --implicit-check-not='tensor<' --implicit-check-not=bufferization. %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-target-llvm)' %s | mlir-translate --mlir-to-llvmir | FileCheck --check-prefix=LLVMIR %s

// The two-add payload makes a full-width tile exceed the SPM planning window,
// so selection must refine both traversal dimensions and materialize the tail.
func.func @complete_non_divisible_grid(
    %lhs: tensor<2x100001xi64>, %rhs: tensor<2x100001xi64>,
    %extra: tensor<2x100001xi64>,
    %out: tensor<2x100001xi64>) -> tensor<2x100001xi64> {
  %group = wafer.group
      ins(%lhs, %rhs, %extra
          : tensor<2x100001xi64>, tensor<2x100001xi64>,
            tensor<2x100001xi64>)
      outs(%out : tensor<2x100001xi64>) {
  ^bb0(%lhs_arg: tensor<2x100001xi64>, %rhs_arg: tensor<2x100001xi64>,
       %extra_arg: tensor<2x100001xi64>,
       %out_arg: tensor<2x100001xi64>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs_arg, %rhs_arg, %extra_arg
          : tensor<2x100001xi64>, tensor<2x100001xi64>,
            tensor<2x100001xi64>)
        outs(%out_arg : tensor<2x100001xi64>) {
    ^bb0(%left: i64, %right: i64, %extra_value: i64, %old: i64):
      %partial = arith.addi %left, %right : i64
      %value = arith.addi %partial, %extra_value : i64
      linalg.yield %value : i64
    } -> tensor<2x100001xi64>
    wafer.group.yield %sum : tensor<2x100001xi64>
  } : tensor<2x100001xi64>
  return %group : tensor<2x100001xi64>
}

// SELECTED-LABEL: func.func @complete_non_divisible_grid
// SELECTED-SAME: memref<2x100001xi64, #wafer.memory<ddr, tensor>>
// SELECTED: memref.subview {{%.*}}[0, 0] [1, 50001] [1, 1]
// SELECTED: wafer.instr.elementwise <add>
// SELECTED: %[[STORE_00:.+]] = memref.subview {{%.*}}[0, 0] [1, 50001] [1, 1]
// SELECTED-NEXT: wafer.instr.wdma {{%.*}} to %[[STORE_00]]
// SELECTED: memref.subview {{%.*}}[0, 50001] [1, 50000] [1, 1]
// SELECTED: wafer.instr.elementwise <add>
// SELECTED: %[[STORE_01:.+]] = memref.subview {{%.*}}[0, 50001] [1, 50000] [1, 1]
// SELECTED-NEXT: wafer.instr.wdma {{%.*}} to %[[STORE_01]]
// SELECTED: memref.subview {{%.*}}[1, 0] [1, 50001] [1, 1]
// SELECTED: wafer.instr.elementwise <add>
// SELECTED: %[[STORE_10:.+]] = memref.subview {{%.*}}[1, 0] [1, 50001] [1, 1]
// SELECTED-NEXT: wafer.instr.wdma {{%.*}} to %[[STORE_10]]
// SELECTED: memref.subview {{%.*}}[1, 50001] [1, 50000] [1, 1]
// SELECTED: wafer.instr.elementwise <add>
// SELECTED: %[[STORE_11:.+]] = memref.subview {{%.*}}[1, 50001] [1, 50000] [1, 1]
// SELECTED-NEXT: wafer.instr.wdma {{%.*}} to %[[STORE_11]]
// SELECTED: wafer.tile.yield

// TARGET-LABEL: llvm.func @complete_non_divisible_grid
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_elementwise_add
// TARGET: llvm.call @wafer_tx81_wdma
// TARGET: llvm.call @wafer_tx81_local_fence
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_elementwise_add
// TARGET: llvm.call @wafer_tx81_wdma
// TARGET: llvm.call @wafer_tx81_local_fence
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_elementwise_add
// TARGET: llvm.call @wafer_tx81_wdma
// TARGET: llvm.call @wafer_tx81_local_fence
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_elementwise_add
// TARGET: llvm.call @wafer_tx81_wdma
// TARGET: llvm.call @wafer_tx81_local_fence
// TARGET: llvm.return

// LLVMIR-DAG: declare void @wafer_tx81_rdma
// LLVMIR-DAG: declare void @wafer_tx81_elementwise_add
// LLVMIR-DAG: declare void @wafer_tx81_wdma
// LLVMIR-DAG: declare void @wafer_tx81_local_fence
// LLVMIR-LABEL: define void @complete_non_divisible_grid
// LLVMIR: call void @wafer_tx81_elementwise_add
// LLVMIR: call void @wafer_tx81_elementwise_add
// LLVMIR: call void @wafer_tx81_elementwise_add
// LLVMIR: call void @wafer_tx81_elementwise_add
// LLVMIR: ret void
