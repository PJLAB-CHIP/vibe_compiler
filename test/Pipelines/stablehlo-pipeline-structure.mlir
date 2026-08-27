// REQUIRES: stablehlo
// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' --dump-pass-pipeline -o /dev/null %s 2>&1 | FileCheck %s

module {
  func.func @main() {
    return
  }
}

// CHECK: builtin.module(wafer-normalize-stablehlo-collectives,wafer-fold-default-stablehlo-execution-ids,wafer-fold-constant-integer-tensor-casts,wafer-lower-static-stablehlo-concatenate,stablehlo-legalize-to-linalg{{.*}},wafer-fold-static-tensor-ops,canonicalize{{.*}})
