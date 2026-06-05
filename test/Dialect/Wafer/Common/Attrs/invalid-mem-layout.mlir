// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module attributes {
  wafer.memory = #wafer.memory<spm, linear>
} {}

// CHECK: invalid Wafer physical memory layout family specification
