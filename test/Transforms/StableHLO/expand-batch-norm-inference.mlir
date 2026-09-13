// REQUIRES: stablehlo
// RUN: split-file %s %t
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-expand-stablehlo-batch-norm-inference))' %t/positive.mlir | FileCheck %s --check-prefix=HLO --implicit-check-not=stablehlo.batch_norm_inference
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-expand-stablehlo-batch-norm-inference,wafer-expand-stablehlo-batch-norm-inference))' %t/positive.mlir | FileCheck %s --check-prefix=HLO --implicit-check-not=stablehlo.batch_norm_inference
// RUN: wafer-opt --wafer-lower-stablehlo-to-linalg %t/positive.mlir | FileCheck %s --check-prefix=LINALG --implicit-check-not=stablehlo.
// RUN: not wafer-opt --pass-pipeline='builtin.module(func.func(wafer-expand-stablehlo-batch-norm-inference))' %t/dynamic.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: not wafer-opt --pass-pipeline='builtin.module(func.func(wafer-expand-stablehlo-batch-norm-inference))' %t/quantized.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-expand-stablehlo-batch-norm-inference))' %t/other.mlir | FileCheck %s --check-prefix=OTHER

//--- positive.mlir
// HLO-LABEL: func.func @feature_middle
// HLO: %[[EPS:.*]] = stablehlo.constant dense<1.250000e-01> : tensor<4xf16>
// HLO-NEXT: %[[VAR:.*]] = stablehlo.add %arg4, %[[EPS]] : tensor<4xf16>
// HLO-NEXT: %[[STD:.*]] = stablehlo.sqrt %[[VAR]] : tensor<4xf16>
// HLO-NEXT: %[[MEAN:.*]] = stablehlo.broadcast_in_dim %arg3, dims = [1]
// HLO-NEXT: %[[CENTER:.*]] = stablehlo.subtract %arg0, %[[MEAN]] : tensor<2x4x1024xf16>
// HLO-NEXT: %[[STD_B:.*]] = stablehlo.broadcast_in_dim %[[STD]], dims = [1]
// HLO-NEXT: %[[NORM:.*]] = stablehlo.divide %[[CENTER]], %[[STD_B]] : tensor<2x4x1024xf16>
// HLO-NEXT: %[[SCALE:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1]
// HLO-NEXT: %[[SCALED:.*]] = stablehlo.multiply %[[SCALE]], %[[NORM]] : tensor<2x4x1024xf16>
// HLO-NEXT: %[[BIAS:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [1]
// HLO-NEXT: %[[Y:.*]] = stablehlo.add %[[SCALED]], %[[BIAS]] : tensor<2x4x1024xf16>
// HLO-NEXT: return %[[Y]] : tensor<2x4x1024xf16>
// LINALG-LABEL: func.func @feature_middle
// LINALG: arith.addf {{.*}} : f16
// LINALG: math.sqrt {{.*}} : f16
// LINALG: arith.subf {{.*}} : f16
// LINALG: arith.divf {{.*}} : f16
// LINALG: arith.mulf {{.*}} : f16
// LINALG: arith.addf {{.*}} : f16
// LINALG: return {{.*}} : tensor<2x4x1024xf16>
func.func @feature_middle(%x: tensor<2x4x1024xf16>, %s: tensor<4xf16>, %b: tensor<4xf16>, %m: tensor<4xf16>, %v: tensor<4xf16>) -> tensor<2x4x1024xf16> {
  %y = "stablehlo.batch_norm_inference"(%x, %s, %b, %m, %v) {epsilon = 0.125 : f32, feature_index = 1 : i64} : (tensor<2x4x1024xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) -> tensor<2x4x1024xf16>
  return %y : tensor<2x4x1024xf16>
}

// HLO-LABEL: func.func @feature_last_tail
// HLO: stablehlo.broadcast_in_dim %arg3, dims = [2]
// HLO: stablehlo.subtract {{.*}} : tensor<2x1025x4xf32>
// HLO: stablehlo.broadcast_in_dim {{.*}}, dims = [2]
// HLO: stablehlo.divide {{.*}} : tensor<2x1025x4xf32>
// HLO: stablehlo.broadcast_in_dim %arg1, dims = [2]
// HLO: stablehlo.multiply {{.*}} : tensor<2x1025x4xf32>
// HLO: stablehlo.broadcast_in_dim %arg2, dims = [2]
// HLO: stablehlo.add {{.*}} : tensor<2x1025x4xf32>
// LINALG-LABEL: func.func @feature_last_tail
// LINALG: math.sqrt {{.*}} : f32
// LINALG: arith.divf {{.*}} : f32
// LINALG: return {{.*}} : tensor<2x1025x4xf32>
func.func @feature_last_tail(%x: tensor<2x1025x4xf32>, %s: tensor<4xf32>, %b: tensor<4xf32>, %m: tensor<4xf32>, %v: tensor<4xf32>) -> tensor<2x1025x4xf32> {
  %y = "stablehlo.batch_norm_inference"(%x, %s, %b, %m, %v) {epsilon = 0.125 : f32, feature_index = 2 : i64} : (tensor<2x1025x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x1025x4xf32>
  return %y : tensor<2x1025x4xf32>
}

// HLO-LABEL: func.func @feature_first_tail
// HLO: stablehlo.broadcast_in_dim %arg3, dims = [0]
// HLO: stablehlo.subtract {{.*}} : tensor<4x2x3x1031xbf16>
// HLO: stablehlo.broadcast_in_dim {{.*}}, dims = [0]
// HLO: stablehlo.divide {{.*}} : tensor<4x2x3x1031xbf16>
// HLO: stablehlo.broadcast_in_dim %arg1, dims = [0]
// HLO: stablehlo.multiply {{.*}} : tensor<4x2x3x1031xbf16>
// HLO: stablehlo.broadcast_in_dim %arg2, dims = [0]
// HLO: stablehlo.add {{.*}} : tensor<4x2x3x1031xbf16>
// LINALG-LABEL: func.func @feature_first_tail
// LINALG: math.sqrt {{.*}} : bf16
// LINALG: arith.divf {{.*}} : bf16
// LINALG: return {{.*}} : tensor<4x2x3x1031xbf16>
func.func @feature_first_tail(%x: tensor<4x2x3x1031xbf16>, %s: tensor<4xbf16>, %b: tensor<4xbf16>, %m: tensor<4xbf16>, %v: tensor<4xbf16>) -> tensor<4x2x3x1031xbf16> {
  %y = "stablehlo.batch_norm_inference"(%x, %s, %b, %m, %v) {epsilon = 0.125 : f32, feature_index = 0 : i64} : (tensor<4x2x3x1031xbf16>, tensor<4xbf16>, tensor<4xbf16>, tensor<4xbf16>, tensor<4xbf16>) -> tensor<4x2x3x1031xbf16>
  return %y : tensor<4x2x3x1031xbf16>
}

// HLO-LABEL: func.func @wide
// HLO: stablehlo.constant dense<1.250000e-01> : tensor<4xf64>
// HLO: stablehlo.divide {{.*}} : tensor<2x1024x4xf64>
// HLO: return {{.*}} : tensor<2x1024x4xf64>
// LINALG-LABEL: func.func @wide
// LINALG: math.sqrt {{.*}} : f64
// LINALG: arith.divf {{.*}} : f64
// LINALG: return {{.*}} : tensor<2x1024x4xf64>
func.func @wide(%x: tensor<2x1024x4xf64>, %s: tensor<4xf64>, %b: tensor<4xf64>, %m: tensor<4xf64>, %v: tensor<4xf64>) -> tensor<2x1024x4xf64> {
  %y = "stablehlo.batch_norm_inference"(%x, %s, %b, %m, %v) {epsilon = 0.125 : f32, feature_index = 2 : i64} : (tensor<2x1024x4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<2x1024x4xf64>
  return %y : tensor<2x1024x4xf64>
}

//--- dynamic.mlir
// REJECT: error: failed to legalize operation 'stablehlo.batch_norm_inference'
func.func @dynamic(%x: tensor<2x4x?xf32>, %f: tensor<4xf32>) -> tensor<2x4x?xf32> {
  %y = "stablehlo.batch_norm_inference"(%x, %f, %f, %f, %f) {epsilon = 0.125 : f32, feature_index = 1 : i64} : (tensor<2x4x?xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4x?xf32>
  return %y : tensor<2x4x?xf32>
}

//--- quantized.mlir
!q = !quant.uniform<i8:f32, 1.0>
func.func @quantized(%x: tensor<2x4x1024x!q>, %f: tensor<4x!q>) -> tensor<2x4x1024x!q> {
  %y = "stablehlo.batch_norm_inference"(%x, %f, %f, %f, %f) {epsilon = 0.125 : f32, feature_index = 1 : i64} : (tensor<2x4x1024x!q>, tensor<4x!q>, tensor<4x!q>, tensor<4x!q>, tensor<4x!q>) -> tensor<2x4x1024x!q>
  return %y : tensor<2x4x1024x!q>
}

//--- other.mlir
// OTHER-LABEL: func.func @training
// OTHER-NEXT: {{.*}} = "stablehlo.batch_norm_training"
// OTHER-NEXT: return
func.func @training(%x: tensor<2x4x1024xf32>, %f: tensor<4xf32>) -> (tensor<2x4x1024xf32>, tensor<4xf32>, tensor<4xf32>) {
  %y:3 = "stablehlo.batch_norm_training"(%x, %f, %f) {epsilon = 0.125 : f32, feature_index = 1 : i64} : (tensor<2x4x1024xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<2x4x1024xf32>, tensor<4xf32>, tensor<4xf32>)
  return %y#0, %y#1, %y#2 : tensor<2x4x1024xf32>, tensor<4xf32>, tensor<4xf32>
}
// OTHER-LABEL: func.func @explicit
// OTHER-NEXT: {{.*}} = stablehlo.subtract
// OTHER-NEXT: {{.*}} = stablehlo.divide
// OTHER-NEXT: return
func.func @explicit(%x: tensor<2x4x1031xf16>, %a: tensor<2x4x1031xf16>, %b: tensor<2x4x1031xf16>) -> tensor<2x4x1031xf16> {
  %c = stablehlo.subtract %x, %a : tensor<2x4x1031xf16>
  %y = stablehlo.divide %c, %b : tensor<2x4x1031xf16>
  return %y : tensor<2x4x1031xf16>
}
