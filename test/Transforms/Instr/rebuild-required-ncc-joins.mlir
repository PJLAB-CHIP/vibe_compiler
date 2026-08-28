// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-rebuild-required-ncc-joins))' %s | FileCheck %s

module {
  func.func @rebuild_removes_stale_joins(%value: f32) {
    %buffer = memref.alloc()
        : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.fill %buffer, %value
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.ncc_join [0, 1]
    wafer.instr.ncc_join [2]
    return
  }

  func.func @optional_loop_branch_issue(
      %lower: index, %upper: index, %condition: i1, %value: f32) {
    %step = arith.constant 1 : index
    scf.for %iv = %lower to %upper step %step {
      scf.if %condition {
        %buffer = memref.alloc()
            : memref<4xf32, #wafer.memory<spm, tensor>>
        wafer.instr.fill %buffer, %value
            {worker = #wafer.ncc_worker<worker0>}
            : memref<4xf32, #wafer.memory<spm, tensor>>, f32
      }
      %later = memref.alloc()
          : memref<8xf32, #wafer.memory<spm, tensor>>
      wafer.instr.fill %later, %value
          {worker = #wafer.ncc_worker<worker0>}
          : memref<8xf32, #wafer.memory<spm, tensor>>, f32
    }
    return
  }
}

// CHECK-LABEL: func.func @rebuild_removes_stale_joins
// CHECK: wafer.instr.fill
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: return
// CHECK-NOT: wafer.instr.ncc_join

// CHECK-LABEL: func.func @optional_loop_branch_issue
// CHECK: scf.for
// CHECK: scf.if
// CHECK: wafer.instr.fill
// CHECK: }
// CHECK: wafer.instr.fill
// CHECK-NEXT: }
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: return
