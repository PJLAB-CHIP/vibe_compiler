// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module attributes {
  wafer.target = #wafer.target<tx8>
} {}

// CHECK: expected ::wafer::Target to be one of: wafer_v0
