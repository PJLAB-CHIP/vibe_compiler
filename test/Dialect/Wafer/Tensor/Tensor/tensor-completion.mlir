// RUN: wafer-opt %s | FileCheck %s
// RUN: wafer-opt --mlir-print-op-generic %s | wafer-opt | FileCheck %s

func.func @forward_selected_states(
    %maximum: tensor<2x1025x64xf16>,
    %sum: tensor<2x1025x64xf16>)
    -> (tensor<2x1025x64xf16>, tensor<2x1025x64xf16>) {
  %0:2 = wafer.tensor.completion %maximum, %sum
      participants([0, 1])
      : (tensor<2x1025x64xf16>, tensor<2x1025x64xf16>)
     -> (tensor<2x1025x64xf16>, tensor<2x1025x64xf16>)
  return %0#0, %0#1
      : tensor<2x1025x64xf16>, tensor<2x1025x64xf16>
}

// CHECK-LABEL: func.func @forward_selected_states
// CHECK: %[[STATES:.+]]:2 = wafer.tensor.completion
// CHECK-SAME: participants([0, 1])
// CHECK: return %[[STATES]]#0, %[[STATES]]#1
