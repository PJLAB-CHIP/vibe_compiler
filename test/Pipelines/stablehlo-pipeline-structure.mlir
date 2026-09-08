// REQUIRES: stablehlo
// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' --dump-pass-pipeline -o /dev/null %s 2>&1 | FileCheck %s

module {
  func.func @main() {
    return
  }
}

// CHECK: Pass Manager with 12 passes:
// CHECK-NEXT: builtin.module(inline{{.*}},wafer-normalize-stablehlo-collectives,wafer-fold-default-stablehlo-execution-ids,wafer-fold-constant-integer-tensor-casts,wafer-lower-static-stablehlo-concatenate,func.func(wafer-promote-stablehlo-logistic),stablehlo-legalize-to-linalg{{.*}},wafer-fold-static-tensor-ops,canonicalize{{.*}},func.func(wafer-fold-convolution-input-casts),func.func(wafer-form-attention-ops),func.func(wafer-normalize-structured-tensor-graph{{.*}}))
