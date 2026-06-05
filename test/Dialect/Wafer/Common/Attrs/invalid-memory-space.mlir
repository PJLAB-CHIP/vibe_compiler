// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module attributes {
  wafer.memory = #wafer.memory<register_file, tensor>
} {
}

// CHECK: invalid Wafer addressable storage space specification
