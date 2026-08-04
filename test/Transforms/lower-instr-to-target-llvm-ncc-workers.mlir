// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/join.mlir | FileCheck %s --check-prefix=JOIN
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/nonzero-issue.mlir | FileCheck %s --check-prefix=WORKERS

//--- join.mlir

func.func @join_workers_zero_and_two() {
  wafer.instr.ncc_join [0, 2]
  return
}

// JOIN: llvm.func @wafer_tx81_ncc_join(i32)
// JOIN-LABEL: llvm.func @join_workers_zero_and_two
// JOIN: %[[MASK:.+]] = llvm.mlir.constant(5 : i32) : i32
// JOIN: llvm.call @wafer_tx81_ncc_join(%[[MASK]]) : (i32) -> ()
// JOIN: llvm.return

//--- nonzero-issue.mlir

func.func @nonzero_issue_worker(
    %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %zero = arith.constant 0.000000e+00 : f16
  wafer.instr.fill %buffer, %zero
      {worker = #wafer.ncc_worker<worker1>}
      : memref<4xf16, #wafer.memory<spm, tensor>>, f16
  wafer.instr.rdma %input to %buffer
      {byte_count = 8 : i64, inner_bytes = 8 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>,
       worker = #wafer.ncc_worker<worker2>}
      : memref<4xf16, #wafer.memory<ddr, tensor>>
     to memref<4xf16, #wafer.memory<spm, tensor>>
  return
}

// WORKERS: llvm.func @wafer_tx81_memset_v3(i64, i32, i32, i32, i32)
// WORKERS: llvm.func @wafer_tx81_rdma_v3(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// WORKERS-LABEL: llvm.func @nonzero_issue_worker
// WORKERS: %[[WORKER1:.+]] = llvm.mlir.constant(1 : i32) : i32
// WORKERS-NEXT: llvm.call @wafer_tx81_memset_v3({{.*}}%[[WORKER1]])
// WORKERS: %[[FORMAT:.+]] = llvm.mlir.constant(2 : i32) : i32
// WORKERS-NEXT: %[[WORKER2:.+]] = llvm.mlir.constant(2 : i32) : i32
// WORKERS-NEXT: llvm.call @wafer_tx81_rdma_v3({{.*}}%[[FORMAT]], %[[WORKER2]])
