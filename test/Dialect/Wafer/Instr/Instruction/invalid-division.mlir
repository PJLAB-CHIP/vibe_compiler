// RUN: not wafer-opt %s 2>&1 | FileCheck %s
// Native divide is retired; source division lowers to recip + mul.
// CHECK: error:
// CHECK: wafer.instr.elementwise <div>
func.func @native_divide(%lhs: memref<1x2x1024xf32, #wafer.memory<spm, tensor>>,
                        %rhs: memref<1x2x1024xf32, #wafer.memory<spm, tensor>>,
                        %dest: memref<1x2x1024xf32, #wafer.memory<spm, tensor>>) {
  wafer.instr.elementwise <div> %lhs, %rhs into %dest : memref<1x2x1024xf32, #wafer.memory<spm, tensor>>, memref<1x2x1024xf32, #wafer.memory<spm, tensor>> into memref<1x2x1024xf32, #wafer.memory<spm, tensor>>
  return
}
