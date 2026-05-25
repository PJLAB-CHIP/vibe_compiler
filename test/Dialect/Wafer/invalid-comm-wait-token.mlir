// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %not_token = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  wafer.comm.wait %not_token : tensor<4xf32>
}

// CHECK: comm wait operands must be async tokens
