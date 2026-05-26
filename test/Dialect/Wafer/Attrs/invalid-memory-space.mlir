// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module attributes {
  wafer.memory_space = #wafer.memory_space<register_file>
} {
}

// CHECK: expected ::wafer::MemorySpace to be one of: spm, ddr
