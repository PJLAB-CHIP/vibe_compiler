// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s
// Scalar buffers exercise a genuine rank-zero fill; destinations are rank three.
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar = affine_map<(d0, d1, d2) -> ()>
func.func private @escape(memref<f32, #wafer.memory<spm, tensor>>)

func.func @mask_1024() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %predicate = "builtin.unrealized_conversion_cast"() : () -> memref<2x1024x32xi1, #wafer.memory<spm, tensor>>
    %data = "builtin.unrealized_conversion_cast"() : () -> memref<2x1024x32xf32, #wafer.memory<spm, tensor>>
    %value = arith.constant 0xFF800000 : f32
    %scalar = memref.alloc() : memref<f32, #wafer.memory<spm, tensor>>
    wafer.tile.fill %scalar, %value : memref<f32, #wafer.memory<spm, tensor>>, f32
    %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
        %predicate, %scalar, %data {indexing_maps = [#id, #scalar, #id, #id]}
        : (memref<2x1024x32xi1, #wafer.memory<spm, tensor>>, memref<f32, #wafer.memory<spm, tensor>>, memref<2x1024x32xf32, #wafer.memory<spm, tensor>>) -> memref<2x1024x32xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @mask_1024
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.fill {{.*}}fill_domain = #wafer.fill_domain<physical_footprint>{{.*}} : memref<2x1024x32xf32, #wafer.memory<spm, tensor>>, f32
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.gather_scatter
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.mask_move
// CHECK-NOT: wafer.instr.gather_scatter

func.func @mask_1025() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %predicate = "builtin.unrealized_conversion_cast"() : () -> memref<2x1025x32xi1, #wafer.memory<spm, tensor>>
    %data = "builtin.unrealized_conversion_cast"() : () -> memref<2x1025x32xf16, #wafer.memory<spm, tensor>>
    %value = arith.constant 0x8000 : f16
    %scalar = memref.alloc() : memref<f16, #wafer.memory<spm, tensor>>
    wafer.tile.fill %scalar, %value : memref<f16, #wafer.memory<spm, tensor>>, f16
    %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
        %predicate, %scalar, %data {indexing_maps = [#id, #scalar, #id, #id]}
        : (memref<2x1025x32xi1, #wafer.memory<spm, tensor>>, memref<f16, #wafer.memory<spm, tensor>>, memref<2x1025x32xf16, #wafer.memory<spm, tensor>>) -> memref<2x1025x32xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @mask_1025
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.fill {{.*}}fill_domain = #wafer.fill_domain<physical_footprint>{{.*}} : memref<2x1025x32xf16, #wafer.memory<spm, tensor>>, f16
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.gather_scatter
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.mask_move
// CHECK-NOT: wafer.instr.gather_scatter

func.func @mask_1031() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %predicate = "builtin.unrealized_conversion_cast"() : () -> memref<2x1031x32xi1, #wafer.memory<spm, tensor>>
    %data = "builtin.unrealized_conversion_cast"() : () -> memref<2x1031x32xbf16, #wafer.memory<spm, tensor>>
    %value = arith.constant 0x7FC1 : bf16
    %scalar = memref.alloc() : memref<bf16, #wafer.memory<spm, tensor>>
    wafer.tile.fill %scalar, %value : memref<bf16, #wafer.memory<spm, tensor>>, bf16
    %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
        %predicate, %scalar, %data {indexing_maps = [#id, #scalar, #id, #id]}
        : (memref<2x1031x32xi1, #wafer.memory<spm, tensor>>, memref<bf16, #wafer.memory<spm, tensor>>, memref<2x1031x32xbf16, #wafer.memory<spm, tensor>>) -> memref<2x1031x32xbf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @mask_1031
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.fill {{.*}}fill_domain = #wafer.fill_domain<physical_footprint>{{.*}} : memref<2x1031x32xbf16, #wafer.memory<spm, tensor>>, bf16
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.gather_scatter
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.mask_move
// CHECK-NOT: wafer.instr.gather_scatter

func.func @escaped_value() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %predicate = "builtin.unrealized_conversion_cast"() : () -> memref<2x1024x32xi1, #wafer.memory<spm, tensor>>
    %data = "builtin.unrealized_conversion_cast"() : () -> memref<2x1024x32xf32, #wafer.memory<spm, tensor>>
    %value = arith.constant 0xFF800000 : f32
    %scalar = memref.alloc() : memref<f32, #wafer.memory<spm, tensor>>
    wafer.tile.fill %scalar, %value : memref<f32, #wafer.memory<spm, tensor>>, f32
    func.call @escape(%scalar) : (memref<f32, #wafer.memory<spm, tensor>>) -> ()
    %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
        %predicate, %scalar, %data {indexing_maps = [#id, #scalar, #id, #id]}
        : (memref<2x1024x32xi1, #wafer.memory<spm, tensor>>, memref<f32, #wafer.memory<spm, tensor>>, memref<2x1024x32xf32, #wafer.memory<spm, tensor>>) -> memref<2x1024x32xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @escaped_value
// CHECK: call @escape
// CHECK: wafer.instr.gather_scatter {{.*}}src_iterations = array<i64: 65536, 1, 1>
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.mask_move


func.func @dynamic_fill(%scalar_value: f32) {
  %token = arith.constant false
  %unused = wafer.tile.region(%token, %scalar_value : i1, f32) -> (i1) {
  ^bb0(%tile_token: i1, %value: f32):
    %predicate = "builtin.unrealized_conversion_cast"() : () -> memref<2x1024x32xi1, #wafer.memory<spm, tensor>>
    %data = "builtin.unrealized_conversion_cast"() : () -> memref<2x1024x32xf32, #wafer.memory<spm, tensor>>
    %scalar = memref.alloc() : memref<f32, #wafer.memory<spm, tensor>>
    wafer.tile.fill %scalar, %value : memref<f32, #wafer.memory<spm, tensor>>, f32
    %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
        %predicate, %scalar, %data {indexing_maps = [#id, #scalar, #id, #id]}
        : (memref<2x1024x32xi1, #wafer.memory<spm, tensor>>, memref<f32, #wafer.memory<spm, tensor>>, memref<2x1024x32xf32, #wafer.memory<spm, tensor>>) -> memref<2x1024x32xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @dynamic_fill
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.fill {{.*}}fill_domain = #wafer.fill_domain<physical_footprint>{{.*}} : memref<2x1024x32xf32, #wafer.memory<spm, tensor>>, f32
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.gather_scatter
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.mask_move
// CHECK-NOT: wafer.instr.gather_scatter
