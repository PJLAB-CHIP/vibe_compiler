// RUN: not wafer-opt --wafer-form-groups --wafer-materialize-multi-tile-no-comm %s 2>&1 | FileCheck %s

module {
  func.func @two_tile_matmul(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%out : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// CHECK: materialize multi-tile no-comm requires exactly one wafer.placement.map
