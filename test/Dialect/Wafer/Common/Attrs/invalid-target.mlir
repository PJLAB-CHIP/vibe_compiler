// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module attributes {
  wafer.target = #wafer.target<unknown>
} {}

// CHECK: expected ::wafer::Target to be one of: wafer
