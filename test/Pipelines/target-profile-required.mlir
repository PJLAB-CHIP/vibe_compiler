// RUN: not wafer-opt --mlir-disable-threading --pass-pipeline='builtin.module(wafer-lower-groups-to-target-llvm)' --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=MISSING --implicit-check-not=llvm.func --implicit-check-not=llvm.call --implicit-check-not=memref.
// RUN: not wafer-opt --mlir-disable-threading --pass-pipeline='builtin.module(wafer-lower-groups-to-target-llvm{target-profile=unknown-profile})' --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN --implicit-check-not=llvm.func --implicit-check-not=llvm.call --implicit-check-not=memref.

module {
  func.func @pipeline_source_is_unchanged(%arg: tensor<4xf32>)
      -> tensor<4xf32> {
    return %arg : tensor<4xf32>
  }
}

// MISSING: missing required target-profile for target LLVM conversion
// MISSING: IR Dump After LowerInstrToTargetLLVMPass Failed
// MISSING: func.func @pipeline_source_is_unchanged
// MISSING-SAME: tensor<4xf32>

// UNKNOWN: invalid target-profile for target LLVM conversion: unknown target profile 'unknown-profile'
// UNKNOWN: IR Dump After LowerInstrToTargetLLVMPass Failed
// UNKNOWN: func.func @pipeline_source_is_unchanged
// UNKNOWN-SAME: tensor<4xf32>
