// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module attributes {
  wafer.mem_layout = #wafer.mem_layout<linear>
} {}

// CHECK: expected ::wafer::MemLayout to be one of: tensor, ntensor, cx, ncx
