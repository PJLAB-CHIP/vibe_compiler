// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %s > %t.min-a
// RUN: FileCheck %s < %t.min-a
// RUN: FileCheck %s --check-prefix=WDMA < %t.min-a
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %s > %t.min-b
// RUN: diff -u %t.min-a %t.min-b
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=first-legal tile-search-effort=quick})' %s > %t.first-a
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=first-legal tile-search-effort=quick})' %s > %t.first-b
// RUN: diff -u %t.first-a %t.first-b

// Bounded rank search evaluates the mandatory-only, one-peer, two-peer and
// all-peer partitions.  Cost ranking should retain the one-region variant so
// all three matmuls reuse one activation RDMA/SPM layout.  Repeated min-cost
// and first-legal runs also cover deterministic partition deduplication.
func.func @shared_activation_matmuls(
    %activation: tensor<4x8xf16>,
    %gate_weight: tensor<8x16xf16>,
    %up_weight: tensor<8x16xf16>,
    %side_weight: tensor<8x16xf16>)
    -> (tensor<4x16xf16>, tensor<4x16xf16>, tensor<4x16xf16>) {
  %zero = arith.constant 0.0 : f16
  %gate_empty = tensor.empty() : tensor<4x16xf16>
  %gate_init = linalg.fill ins(%zero : f16)
      outs(%gate_empty : tensor<4x16xf16>) -> tensor<4x16xf16>
  %gate = linalg.matmul
      ins(%activation, %gate_weight : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%gate_init : tensor<4x16xf16>) -> tensor<4x16xf16>
  %up_empty = tensor.empty() : tensor<4x16xf16>
  %up_init = linalg.fill ins(%zero : f16)
      outs(%up_empty : tensor<4x16xf16>) -> tensor<4x16xf16>
  %up = linalg.matmul
      ins(%activation, %up_weight : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%up_init : tensor<4x16xf16>) -> tensor<4x16xf16>
  %side_empty = tensor.empty() : tensor<4x16xf16>
  %side_init = linalg.fill ins(%zero : f16)
      outs(%side_empty : tensor<4x16xf16>) -> tensor<4x16xf16>
  %side = linalg.matmul
      ins(%activation, %side_weight : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%side_init : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %gate, %up, %side
      : tensor<4x16xf16>, tensor<4x16xf16>, tensor<4x16xf16>
}

// CHECK-LABEL: func.func @shared_activation_matmuls
// CHECK-COUNT-1: wafer.tile.region
// One full static tile has no time-domain loop.  The direct region body still
// proves that all three roots share the activation's one RDMA/SPM layout.
// CHECK-NOT: scf.for
// CHECK: %[[ACT:.+]] = memref.alloc() {{.*}} : memref<4x8xf16, #wafer.memory<spm, tensor>>
// CHECK-NEXT: wafer.instr.rdma {{%.*}} to %[[ACT]]
// CHECK: wafer.instr.rdma
// CHECK: %[[ACT_CX:.+]] = memref.alloc() {{.*}} : memref<4x8xf16, #wafer.memory<spm, cx>>
// CHECK-NEXT: wafer.instr.gather_scatter %[[ACT]] to %[[ACT_CX]]
// CHECK: wafer.instr.gemm %[[ACT_CX]],
// CHECK: wafer.instr.gemm %[[ACT_CX]],
// CHECK: wafer.instr.gemm %[[ACT_CX]],
// CHECK-NOT: scf.for
// CHECK: wafer.tile.yield
// CHECK-NOT: wafer.tile.region

// WDMA-LABEL: func.func @shared_activation_matmuls
// WDMA-COUNT-3: wafer.instr.wdma
// WDMA-NOT: wafer.instr.wdma
// WDMA: return
