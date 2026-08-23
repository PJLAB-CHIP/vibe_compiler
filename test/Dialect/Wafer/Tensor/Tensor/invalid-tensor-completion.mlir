// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @unsorted_participants(%state: tensor<2x1031x64xbf16>)
    -> tensor<2x1031x64xbf16> {
  %0 = wafer.tensor.completion %state participants([1, 0])
      : (tensor<2x1031x64xbf16>) -> tensor<2x1031x64xbf16>
  return %0 : tensor<2x1031x64xbf16>
}

// CHECK: 'wafer.tensor.completion' op requires a sorted unique nonempty NCC worker participant set
