// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/join.mlir | FileCheck %s --check-prefix=JOIN
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/nonzero-issue.mlir 2>&1 | FileCheck %s --check-prefix=NONZERO

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

func.func @nonzero_issue_worker() {
  %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %zero = arith.constant 0.000000e+00 : f16
  wafer.instr.fill %buffer, %zero
      {worker = #wafer.ncc_worker<worker1>}
      : memref<4xf16, #wafer.memory<spm, tensor>>, f16
  return
}

// NONZERO: unsupported_target_worker: current target operator ABI does not encode NCC issue worker 'worker1'; only worker0 is lowerable
