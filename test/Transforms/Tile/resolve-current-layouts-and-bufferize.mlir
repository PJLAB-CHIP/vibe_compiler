// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-resolve-layouts-and-bufferize)' %s | FileCheck %s

#id = affine_map<(b, m, n) -> (b, m, n)>

module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @aligned(%input: tensor<2x1024x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x1024x64xf16>)
          -> (tensor<2x1024x64xf16>) {
      ^bb0(%local: tensor<2x1024x64xf16>):
        %empty = tensor.empty() : tensor<2x1024x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<2x1024x64xf16>)
            outs(%empty : tensor<2x1024x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<2x1024x64xf16>
        wafer.tile.yield %mapped : tensor<2x1024x64xf16>
      }
      return
    }
  }
  wafer.tile.module card_id = 0 tile_id = 1 {
    func.func @ragged(%input: tensor<2x1025x64xbf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x1025x64xbf16>)
          -> (tensor<2x1025x64xbf16>) {
      ^bb0(%local: tensor<2x1025x64xbf16>):
        %empty = tensor.empty() : tensor<2x1025x64xbf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<2x1025x64xbf16>)
            outs(%empty : tensor<2x1025x64xbf16>) {
          ^bb1(%value: bf16, %old: bf16):
            %next = arith.addf %value, %value : bf16
            linalg.yield %next : bf16
        } -> tensor<2x1025x64xbf16>
        wafer.tile.yield %mapped : tensor<2x1025x64xbf16>
      }
      return
    }
  }
}

// CHECK-LABEL: wafer.tile.module card_id = 0 tile_id = 0
// CHECK: func.func @aligned(%[[INPUT:.+]]: memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>)
// CHECK: %[[BOUNDARY:.+]] = bufferization.to_tensor %[[INPUT]]
// CHECK: wafer.tile.region(%[[BOUNDARY]] : tensor<2x1024x64xf16>)
// CHECK: ^bb0(%[[LOCAL:.+]]: tensor<2x1024x64xf16>):
// CHECK: %[[LOCAL_BUFFER:.+]] = bufferization.to_memref %[[LOCAL]] : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[DEST:.+]] = memref.alloc() {{.*}} : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
// CHECK: linalg.generic {{.*}} ins(%[[LOCAL_BUFFER]] {{.*}}) outs(%[[DEST]]
// CHECK: %[[RESULT_TENSOR:.+]] = bufferization.to_tensor %[[DEST]]
// CHECK: wafer.tile.yield %[[RESULT_TENSOR]] : tensor<2x1024x64xf16>
// CHECK-NOT: tensor.empty

// CHECK-LABEL: wafer.tile.module card_id = 0 tile_id = 1
// CHECK: func.func @ragged(%{{.+}}: memref<2x1025x64xbf16, #wafer.memory<ddr, tensor>>)
// CHECK: memref<2x1025x64xbf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: tensor.empty
