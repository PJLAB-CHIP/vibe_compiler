// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{target-profile=wafer-tx81-single-card-kernel-v1 default-ddr-arena-argument-index=1})' %s | FileCheck %s

module {
  func.func @arena_bound(
      %external: memref<16xf32, #wafer.memory<ddr, tensor>>,
      %arena: i64) {
    %scratch = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<256>}
        : memref<16xf32, #wafer.memory<ddr, tensor>>
    return
  }
}

// CHECK-LABEL: llvm.func @arena_bound
// CHECK-SAME: ({{.*}}: i64, {{.*}}: i64)
// CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(256 : i64) : i64
// CHECK: llvm.add {{.*}}, %[[OFFSET]] : i64
// CHECK: llvm.return
