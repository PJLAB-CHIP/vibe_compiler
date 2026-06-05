// RUN: wafer-opt %s | FileCheck %s

module attributes {
  wafer.target = #wafer.target<wafer>,
  wafer.memory = #wafer.memory<spm, tensor>,
  wafer.placement = #wafer.placement<single_tile>
} {
}

// CHECK-DAG: wafer.target = #wafer.target<wafer>
// CHECK-DAG: wafer.memory = #wafer.memory<spm, tensor>
// CHECK-DAG: wafer.placement = #wafer.placement<single_tile>
