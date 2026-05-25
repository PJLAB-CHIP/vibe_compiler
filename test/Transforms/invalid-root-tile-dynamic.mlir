// RUN: not wafer-opt --wafer-check-root-tile-candidates %s 2>&1 | FileCheck %s

module {
  func.func @dynamic_root_tile(
      %lhs: tensor<?x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<?x16xf16>) -> tensor<?x16xf16> {
    %0 = wafer.group ins(%lhs, %rhs : tensor<?x8xf16>, tensor<8x16xf16>)
                     outs(%out : tensor<?x16xf16>) {
    ^bb0(%glhs: tensor<?x8xf16>,
         %grhs: tensor<8x16xf16>,
         %gout: tensor<?x16xf16>):
      %1 = linalg.matmul
          ins(%glhs, %grhs : tensor<?x8xf16>, tensor<8x16xf16>)
          outs(%gout : tensor<?x16xf16>) -> tensor<?x16xf16>
      wafer.group_yield %1 : tensor<?x16xf16>
    } : tensor<?x16xf16>
    return %0 : tensor<?x16xf16>
  }
}

// CHECK: wafer.group
// CHECK: has no feasible root tile candidate
// CHECK: dynamic result shape
