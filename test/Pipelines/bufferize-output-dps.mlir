// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-bufferize-instr-functions)' %s | FileCheck %s

#id = affine_map<(b, m, n) -> (b, m, n)>

func.func @aligned_output(
    %input: tensor<2x1024x64xf16>,
    %output: tensor<2x1024x64xf16>) -> tensor<2x1024x64xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %result = scf.for %batch = %c0 to %c2 step %c1
      iter_args(%acc = %output) -> tensor<2x1024x64xf16> {
    %input_slice = tensor.extract_slice %input[%batch, 0, 0]
        [1, 1024, 64] [1, 1, 1]
        : tensor<2x1024x64xf16> to tensor<1x1024x64xf16>
    %output_slice = tensor.extract_slice %acc[%batch, 0, 0]
        [1, 1024, 64] [1, 1, 1]
        : tensor<2x1024x64xf16> to tensor<1x1024x64xf16>
    %mapped = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input_slice : tensor<1x1024x64xf16>)
        outs(%output_slice : tensor<1x1024x64xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<1x1024x64xf16>
    %next = tensor.insert_slice %mapped into %acc[%batch, 0, 0]
        [1, 1024, 64] [1, 1, 1]
        : tensor<1x1024x64xf16> into tensor<2x1024x64xf16>
    scf.yield %next : tensor<2x1024x64xf16>
  }
  return %result : tensor<2x1024x64xf16>
}

// CHECK-LABEL: func.func @aligned_output(
// CHECK-SAME: memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>
// CHECK-SAME: memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>) {
// CHECK-NOT: memref.copy
// CHECK: scf.for
// CHECK: return

func.func @necessary_alias_copy_aligned(
    %output: tensor<2x1024x64xf16>, %index: index)
    -> (tensor<2x1024x64xf16>, f16) {
  %c0 = arith.constant 0 : index
  %value = arith.constant 1.000000e+00 : f16
  %updated = tensor.insert %value into %output[%c0, %index, %c0]
      : tensor<2x1024x64xf16>
  %old = tensor.extract %output[%c0, %index, %c0]
      : tensor<2x1024x64xf16>
  return %updated, %old : tensor<2x1024x64xf16>, f16
}

// CHECK-LABEL: func.func @necessary_alias_copy_aligned(
// CHECK: %[[COPY:.+]] = memref.alloc()
// CHECK: memref.copy %{{.+}}, %[[COPY]]
// CHECK: memref.store
// CHECK: return %[[COPY]],

func.func @necessary_alias_copy_ragged(
    %output: tensor<2x1025x64xf16>, %index: index)
    -> (tensor<2x1025x64xf16>, f16) {
  %c0 = arith.constant 0 : index
  %value = arith.constant 1.000000e+00 : f16
  %updated = tensor.insert %value into %output[%c0, %index, %c0]
      : tensor<2x1025x64xf16>
  %old = tensor.extract %output[%c0, %index, %c0]
      : tensor<2x1025x64xf16>
  return %updated, %old : tensor<2x1025x64xf16>, f16
}

// CHECK-LABEL: func.func @necessary_alias_copy_ragged(
// CHECK: %[[RAGGED_COPY:.+]] = memref.alloc()
// CHECK: memref.copy %{{.+}}, %[[RAGGED_COPY]]
// CHECK: memref.store
// CHECK: return %[[RAGGED_COPY]],

func.func @ragged_output(
    %input: tensor<2x1025x64xf16>,
    %output: tensor<2x1025x64xf16>) -> tensor<2x1025x64xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %result = scf.for %batch = %c0 to %c2 step %c1
      iter_args(%acc = %output) -> tensor<2x1025x64xf16> {
    %input_slice = tensor.extract_slice %input[%batch, 0, 0]
        [1, 1025, 64] [1, 1, 1]
        : tensor<2x1025x64xf16> to tensor<1x1025x64xf16>
    %output_slice = tensor.extract_slice %acc[%batch, 0, 0]
        [1, 1025, 64] [1, 1, 1]
        : tensor<2x1025x64xf16> to tensor<1x1025x64xf16>
    %mapped = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input_slice : tensor<1x1025x64xf16>)
        outs(%output_slice : tensor<1x1025x64xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<1x1025x64xf16>
    %next = tensor.insert_slice %mapped into %acc[%batch, 0, 0]
        [1, 1025, 64] [1, 1, 1]
        : tensor<1x1025x64xf16> into tensor<2x1025x64xf16>
    scf.yield %next : tensor<2x1025x64xf16>
  }
  return %result : tensor<2x1025x64xf16>
}

// CHECK-LABEL: func.func @ragged_output(
// CHECK-SAME: memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>
// CHECK-SAME: memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>) {
// CHECK-NOT: memref.copy
// CHECK: scf.for
// CHECK: return
