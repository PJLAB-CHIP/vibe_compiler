// RUN: wafer-opt --split-input-file --wafer-lower-stablehlo-to-linalg %s | FileCheck %s

module {
  func.func @fp16_1024(%arg0: tensor<24xf16>, %arg1: tensor<24x16x3x3xf16>, %arg2: tensor<1x16x8x1024xf16>) -> tensor<1x24x8x1024xf16> {
    %0 = stablehlo.convert %arg2 : (tensor<1x16x8x1024xf16>) -> tensor<1x16x8x1024xf32>
    %1 = stablehlo.convert %arg1 : (tensor<24x16x3x3xf16>) -> tensor<24x16x3x3xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x8x1024xf32>, tensor<24x16x3x3xf32>) -> tensor<1x24x8x1024xf32>
    %3 = stablehlo.convert %arg0 : (tensor<24xf16>) -> tensor<24xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<24xf32>) -> tensor<1x24x8x1024xf32>
    %5 = stablehlo.add %2, %4 : tensor<1x24x8x1024xf32>
    %6 = stablehlo.convert %5 : (tensor<1x24x8x1024xf32>) -> tensor<1x24x8x1024xf16>
    return %6 : tensor<1x24x8x1024xf16>
  }
}

// CHECK-LABEL: func.func @fp16_1024
// CHECK: tensor.pad %arg2
// CHECK: } : tensor<1x16x8x1024xf16> to tensor<1x16x10x1026xf16>
// CHECK: ins({{.*}} : tensor<1x16x10x1026xf16>, tensor<24x16x3x3xf16>)
// CHECK: ^bb0(%[[X:[^: ]+]]: f16, %[[W:[^: ]+]]: f16, %[[A:[^: ]+]]: f32):
// CHECK: %[[XF:[^ ]+]] = arith.extf %[[X]] : f16 to f32
// CHECK: %[[WF:[^ ]+]] = arith.extf %[[W]] : f16 to f32
// CHECK: arith.mulf %[[XF]], %[[WF]] : f32
// CHECK: arith.addf %[[A]], {{.*}} : f32
// CHECK: linalg.yield {{.*}} : f32
// CHECK: arith.addf {{.*}} : f32
// CHECK: arith.truncf {{.*}} : f32 to f16

// -----

module {
  func.func @bf16_1025(%arg0: tensor<24xbf16>, %arg1: tensor<24x16x3x3xbf16>, %arg2: tensor<1x16x8x1025xbf16>) -> tensor<1x24x8x1025xbf16> {
    %0 = stablehlo.convert %arg2 : (tensor<1x16x8x1025xbf16>) -> tensor<1x16x8x1025xf32>
    %1 = stablehlo.convert %arg1 : (tensor<24x16x3x3xbf16>) -> tensor<24x16x3x3xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x8x1025xf32>, tensor<24x16x3x3xf32>) -> tensor<1x24x8x1025xf32>
    %3 = stablehlo.convert %arg0 : (tensor<24xbf16>) -> tensor<24xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<24xf32>) -> tensor<1x24x8x1025xf32>
    %5 = stablehlo.add %2, %4 : tensor<1x24x8x1025xf32>
    %6 = stablehlo.convert %5 : (tensor<1x24x8x1025xf32>) -> tensor<1x24x8x1025xbf16>
    return %6 : tensor<1x24x8x1025xbf16>
  }
}

// CHECK-LABEL: func.func @bf16_1025
// CHECK: tensor.pad %arg2
// CHECK: } : tensor<1x16x8x1025xbf16> to tensor<1x16x10x1027xbf16>
// CHECK: ins({{.*}} : tensor<1x16x10x1027xbf16>, tensor<24x16x3x3xbf16>)
// CHECK: ^bb0(%[[X:[^: ]+]]: bf16, %[[W:[^: ]+]]: bf16, %[[A:[^: ]+]]: f32):
// CHECK: %[[XF:[^ ]+]] = arith.extf %[[X]] : bf16 to f32
// CHECK: %[[WF:[^ ]+]] = arith.extf %[[W]] : bf16 to f32
// CHECK: arith.mulf %[[XF]], %[[WF]] : f32
// CHECK: arith.addf %[[A]], {{.*}} : f32
// CHECK: linalg.yield {{.*}} : f32
// CHECK: arith.addf {{.*}} : f32
// CHECK: arith.truncf {{.*}} : f32 to bf16

// -----

module {
  func.func @f16_1031(%arg0: tensor<24xf16>, %arg1: tensor<24x16x3x3xf16>, %arg2: tensor<1x16x8x1031xf16>) -> tensor<1x24x8x1031xf16> {
    %0 = stablehlo.convert %arg2 : (tensor<1x16x8x1031xf16>) -> tensor<1x16x8x1031xf32>
    %1 = stablehlo.convert %arg1 : (tensor<24x16x3x3xf16>) -> tensor<24x16x3x3xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x8x1031xf32>, tensor<24x16x3x3xf32>) -> tensor<1x24x8x1031xf32>
    %3 = stablehlo.convert %arg0 : (tensor<24xf16>) -> tensor<24xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<24xf32>) -> tensor<1x24x8x1031xf32>
    %5 = stablehlo.add %2, %4 : tensor<1x24x8x1031xf32>
    %6 = stablehlo.convert %5 : (tensor<1x24x8x1031xf32>) -> tensor<1x24x8x1031xf16>
    return %6 : tensor<1x24x8x1031xf16>
  }
}

// CHECK-LABEL: func.func @f16_1031
// CHECK: tensor.pad %arg2
// CHECK: } : tensor<1x16x8x1031xf16> to tensor<1x16x10x1033xf16>
// CHECK: ins({{.*}} : tensor<1x16x10x1033xf16>, tensor<24x16x3x3xf16>)
// CHECK: ^bb0(%[[X:[^: ]+]]: f16, %[[W:[^: ]+]]: f16, %[[A:[^: ]+]]: f32):
// CHECK: %[[XF:[^ ]+]] = arith.extf %[[X]] : f16 to f32
// CHECK: %[[WF:[^ ]+]] = arith.extf %[[W]] : f16 to f32
// CHECK: arith.mulf %[[XF]], %[[WF]] : f32
// CHECK: arith.addf %[[A]], {{.*}} : f32
// CHECK: linalg.yield {{.*}} : f32
// CHECK: arith.addf {{.*}} : f32
// CHECK: arith.truncf {{.*}} : f32 to f16
