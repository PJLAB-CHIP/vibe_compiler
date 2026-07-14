// RUN: not wafer-opt --mlir-disable-threading --wafer-lower-instr-to-target-llvm --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=MISSING --implicit-check-not=llvm.func --implicit-check-not=llvm.call
// RUN: not wafer-opt --mlir-disable-threading --wafer-lower-instr-to-target-llvm='target-profile=unknown-profile' --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN --implicit-check-not=llvm.func --implicit-check-not=llvm.call

module {
  func.func @source_is_unchanged(
      %arg: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    return
  }
}

// MISSING: missing required target-profile for target LLVM conversion
// MISSING: IR Dump After LowerInstrToTargetLLVMPass Failed
// MISSING: func.func @source_is_unchanged
// MISSING-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>

// UNKNOWN: invalid target-profile for target LLVM conversion: unknown target profile 'unknown-profile'
// UNKNOWN: IR Dump After LowerInstrToTargetLLVMPass Failed
// UNKNOWN: func.func @source_is_unchanged
// UNKNOWN-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
