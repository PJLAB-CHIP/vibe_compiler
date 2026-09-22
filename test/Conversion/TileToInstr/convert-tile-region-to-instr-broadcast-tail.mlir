// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr,wafer-plan-spm-memory{spm-limit=8388608},wafer-lower-instr-to-target-llvm)' %s | FileCheck %s
// Two batches with seven valid tail lanes: each uses a bounded eight-lane
// scratch, a VuV64 main group and a VuV8 tail. The actual SPM planner and
// target conversion must consume the address-preserving physical views.
// CHECK-COUNT-4: llvm.call @wafer_tx81_elementwise_sub
// CHECK-NOT: memref.memory_space_cast
module { func.func @main() { wafer.tile.region() -> () {
%a = memref.alloc() : memref<2x1025x263xbf16, #wafer.memory<spm, ncx>>
%b = memref.alloc() : memref<2x263xbf16, #wafer.memory<spm, tensor>>
wafer.tile.elementwise_into <sub> %a, %b into %a {indexing_maps = [affine_map<(a,b,c)->(a,b,c)>, affine_map<(a,b,c)->(a,c)>, affine_map<(a,b,c)->(a,b,c)>]} : memref<2x1025x263xbf16, #wafer.memory<spm, ncx>>, memref<2x263xbf16, #wafer.memory<spm, tensor>> into memref<2x1025x263xbf16, #wafer.memory<spm, ncx>>
wafer.tile.yield
} return } }
