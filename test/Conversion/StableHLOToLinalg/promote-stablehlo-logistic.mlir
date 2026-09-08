// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-promote-stablehlo-logistic))' %s | FileCheck %s --check-prefix=HLO
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-promote-stablehlo-logistic,wafer-promote-stablehlo-logistic))' %s | FileCheck %s --check-prefix=HLO
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s --check-prefix=LINALG

// HLO-LABEL: func.func @half
// HLO: %[[X:.*]] = stablehlo.convert %arg0 : (tensor<2x4x1024xf16>) -> tensor<2x4x1024xf32>
// HLO-NEXT: %[[Y:.*]] = stablehlo.logistic %[[X]] : tensor<2x4x1024xf32>
// HLO-NEXT: %[[Z:.*]] = stablehlo.convert %[[Y]] : (tensor<2x4x1024xf32>) -> tensor<2x4x1024xf16>
// HLO-NEXT: return %[[Z]]
// LINALG-LABEL: func.func @half
// LINALG: arith.extf {{.*}} : f16 to f32
// LINALG: arith.negf {{.*}} : f32
// LINALG: math.exp {{.*}} : f32
// LINALG: arith.addf {{.*}} : f32
// LINALG: arith.divf {{.*}} : f32
// LINALG: arith.truncf {{.*}} : f32 to f16
// LINALG: return {{.*}} : tensor<2x4x1024xf16>
func.func @half(%arg0: tensor<2x4x1024xf16>) -> tensor<2x4x1024xf16> {
  %0 = stablehlo.logistic %arg0 : tensor<2x4x1024xf16>
  return %0 : tensor<2x4x1024xf16>
}

// HLO-LABEL: func.func @bfloat_tail
// HLO: stablehlo.convert %arg0 : (tensor<2x4x1025xbf16>) -> tensor<2x4x1025xf32>
// HLO-NEXT: stablehlo.logistic {{.*}} : tensor<2x4x1025xf32>
// HLO-NEXT: stablehlo.convert {{.*}} : (tensor<2x4x1025xf32>) -> tensor<2x4x1025xbf16>
// HLO-NEXT: return
// LINALG-LABEL: func.func @bfloat_tail
// LINALG: arith.extf {{.*}} : bf16 to f32
// LINALG: math.exp {{.*}} : f32
// LINALG: arith.truncf {{.*}} : f32 to bf16
func.func @bfloat_tail(%arg0: tensor<2x4x1025xbf16>) -> tensor<2x4x1025xbf16> {
  %0 = stablehlo.logistic %arg0 : tensor<2x4x1025xbf16>
  return %0 : tensor<2x4x1025xbf16>
}

// HLO-LABEL: func.func @half_tail
// HLO: stablehlo.convert %arg0 : (tensor<2x4x1031xf16>) -> tensor<2x4x1031xf32>
// HLO-NEXT: stablehlo.logistic {{.*}} : tensor<2x4x1031xf32>
// HLO-NEXT: stablehlo.convert {{.*}} : (tensor<2x4x1031xf32>) -> tensor<2x4x1031xf16>
// HLO-NEXT: return
func.func @half_tail(%arg0: tensor<2x4x1031xf16>) -> tensor<2x4x1031xf16> {
  %0 = stablehlo.logistic %arg0 : tensor<2x4x1031xf16>
  return %0 : tensor<2x4x1031xf16>
}

// HLO-LABEL: func.func @wide
// HLO-NEXT: %[[A:.*]] = stablehlo.logistic %arg0 : tensor<2x4x1024xf32>
// HLO-NEXT: %[[B:.*]] = stablehlo.logistic %arg1 : tensor<2x4x1024xf64>
// HLO-NEXT: return %[[A]], %[[B]]
func.func @wide(%arg0: tensor<2x4x1024xf32>, %arg1: tensor<2x4x1024xf64>) -> (tensor<2x4x1024xf32>, tensor<2x4x1024xf64>) {
  %0 = stablehlo.logistic %arg0 : tensor<2x4x1024xf32>
  %1 = stablehlo.logistic %arg1 : tensor<2x4x1024xf64>
  return %0, %1 : tensor<2x4x1024xf32>, tensor<2x4x1024xf64>
}

// HLO-LABEL: func.func @explicit_primitives
// HLO-NEXT: stablehlo.negate %arg0 : tensor<2x4x1025xf16>
// HLO-NEXT: stablehlo.exponential {{.*}} : tensor<2x4x1025xf16>
// HLO-NEXT: stablehlo.add {{.*}} : tensor<2x4x1025xf16>
// HLO-NEXT: stablehlo.divide {{.*}} : tensor<2x4x1025xf16>
// HLO-NEXT: return
// LINALG-LABEL: func.func @explicit_primitives
// LINALG: math.exp {{.*}} : f16
// LINALG: arith.addf {{.*}} : f16
// LINALG: arith.divf {{.*}} : f16
func.func @explicit_primitives(%arg0: tensor<2x4x1025xf16>, %arg1: tensor<2x4x1025xf16>) -> tensor<2x4x1025xf16> {
  %0 = stablehlo.negate %arg0 : tensor<2x4x1025xf16>
  %1 = stablehlo.exponential %0 : tensor<2x4x1025xf16>
  %2 = stablehlo.add %1, %arg1 : tensor<2x4x1025xf16>
  %3 = stablehlo.divide %arg1, %2 : tensor<2x4x1025xf16>
  return %3 : tensor<2x4x1025xf16>
}
