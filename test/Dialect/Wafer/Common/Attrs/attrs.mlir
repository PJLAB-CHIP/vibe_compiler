// RUN: wafer-opt %s | FileCheck %s

module attributes {
  wafer.target = #wafer.target<wafer>,
  wafer.memory_space = #wafer.memory_space<spm>,
  wafer.mem_layout = #wafer.mem_layout<tensor>,
  wafer.placement = #wafer.placement<single_tile>
} {
}

// CHECK-DAG: wafer.target = #wafer.target<wafer>
// CHECK-DAG: wafer.memory_space = #wafer.memory_space<spm>
// CHECK-DAG: wafer.mem_layout = #wafer.mem_layout<tensor>
// CHECK-DAG: wafer.placement = #wafer.placement<single_tile>
