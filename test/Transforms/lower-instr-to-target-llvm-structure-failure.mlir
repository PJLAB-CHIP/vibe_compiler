// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | FileCheck %s --check-prefix=POS
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/indirect.mlir 2>&1 | FileCheck %s --check-prefix=INDIRECT
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/recursive.mlir 2>&1 | FileCheck %s --check-prefix=RECURSIVE
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/external.mlir 2>&1 | FileCheck %s --check-prefix=EXTERNAL
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/alias-mismatch.mlir 2>&1 | FileCheck %s --check-prefix=ALIAS
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/scalar-arg.mlir 2>&1 | FileCheck %s --check-prefix=SCALAR
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/ddr-alloc.mlir 2>&1 | FileCheck %s --check-prefix=DDR-ALLOC
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/mask-address-overflow.mlir 2>&1 | FileCheck %s --check-prefix=MASK-ADDRESS
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/factorize.mlir 2>&1 | FileCheck %s --check-prefix=FACTORIZE
// RUN: not wafer-opt --mlir-disable-threading --wafer-lower-instr-to-target-llvm --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %t/dte-atomic.mlir 2>&1 | FileCheck %s --check-prefix=DTE-ATOMIC --implicit-check-not=llvm.func --implicit-check-not=llvm.call
// RUN: not wafer-opt --mlir-disable-threading --wafer-lower-instr-to-target-llvm --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %t/late-atomic.mlir 2>&1 | FileCheck %s --check-prefix=LATE-ATOMIC --implicit-check-not=llvm.func --implicit-check-not=llvm.call

//--- positive.mlir

func.func @false_branch() {
  %false = arith.constant false
  scf.if %false {
    wafer.instr.local_fence
  } else {
    wafer.instr.local_fence
  }
  return
}

func.func @true_branch() {
  %true = arith.constant true
  scf.if %true {
    wafer.instr.local_fence
  } else {
    wafer.instr.local_fence
  }
  return
}

func.func @zero_trip_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c0 step %c1 {
    wafer.instr.local_fence
  }
  return
}

func.func @one_trip_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    wafer.instr.local_fence
  }
  return
}

func.func @two_trip_nested() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c2 step %c1 {
    scf.if %true {
      wafer.instr.local_fence
    }
  }
  return
}

func.func @diamond_cfg() {
  %cond = arith.constant true
  cf.cond_br %cond, ^left, ^right
^left:
  wafer.instr.local_fence
  cf.br ^merge
^right:
  wafer.instr.local_fence
  cf.br ^merge
^merge:
  wafer.instr.local_fence
  return
}

func.func private @forward_alias(
    %arg: memref<4xf16, #wafer.memory<ddr, tensor>>)
    -> memref<4xf16, #wafer.memory<ddr, tensor>> {
  wafer.instr.local_fence
  return %arg : memref<4xf16, #wafer.memory<ddr, tensor>>
}

func.func @direct_call(
    %arg: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %alias = func.call @forward_alias(%arg)
      : (memref<4xf16, #wafer.memory<ddr, tensor>>)
     -> memref<4xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %alias to %spm
      {byte_count = 8 : i64, inner_bytes = 8 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4xf16, #wafer.memory<ddr, tensor>>
     to memref<4xf16, #wafer.memory<spm, tensor>>
  return
}

func.func @nested_subview(
    %arg: memref<8xf16, #wafer.memory<ddr, tensor>>) {
  %first = memref.subview %arg[2] [4] [1]
      : memref<8xf16, #wafer.memory<ddr, tensor>>
     to memref<4xf16, strided<[1], offset: 2>, #wafer.memory<ddr, tensor>>
  %second = memref.subview %first[1] [2] [1]
      : memref<4xf16, strided<[1], offset: 2>, #wafer.memory<ddr, tensor>>
     to memref<2xf16, strided<[1], offset: 3>, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<2xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %second to %spm
      {byte_count = 4 : i64, inner_bytes = 4 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<2xf16, strided<[1], offset: 3>, #wafer.memory<ddr, tensor>>
     to memref<2xf16, #wafer.memory<spm, tensor>>
  return
}

func.func @tile_boundary_yield(
    %arg: memref<4xf16, #wafer.memory<ddr, tensor>>)
    -> memref<4xf16, #wafer.memory<ddr, tensor>> {
  %result = wafer.tile.region(
      %arg : memref<4xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%boundary: memref<4xf16, #wafer.memory<ddr, tensor>>):
    wafer.instr.local_fence
    wafer.tile.yield %boundary
        : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
  return %result : memref<4xf16, #wafer.memory<ddr, tensor>>
}

func.func @loop_carried_alias(
    %arg: memref<4xf16, #wafer.memory<ddr, tensor>>)
    -> memref<4xf16, #wafer.memory<ddr, tensor>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %result = scf.for %i = %c0 to %c2 step %c1
      iter_args(%carried = %arg)
      -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    scf.yield %carried : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
  return %result : memref<4xf16, #wafer.memory<ddr, tensor>>
}

func.func @mask_uint32_boundary() {
  %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %mask = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<4294967288>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.mask_move %src, %mask into %dst
      : memref<4xf16, #wafer.memory<spm, tensor>>,
        memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<4xf16, #wafer.memory<spm, tensor>>
  return
}

// POS-LABEL: llvm.func @false_branch
// POS: llvm.cond_br
// POS: llvm.call @wafer_tx81_local_fence
// POS: llvm.call @wafer_tx81_local_fence
// POS-LABEL: llvm.func @true_branch
// POS: llvm.cond_br
// POS-LABEL: llvm.func @zero_trip_loop
// POS: llvm.cond_br
// POS-LABEL: llvm.func @one_trip_loop
// POS: llvm.cond_br
// POS-LABEL: llvm.func @two_trip_nested
// POS: llvm.cond_br
// POS: llvm.cond_br
// POS-LABEL: llvm.func @diamond_cfg
// POS: llvm.cond_br
// POS: llvm.br
// POS: llvm.br
// POS-LABEL: llvm.func @forward_alias(
// POS-SAME: %[[ALIAS_ARG:.+]]: i64) -> i64
// POS: llvm.call @wafer_tx81_local_fence
// POS: llvm.return %[[ALIAS_ARG]] : i64
// POS-LABEL: llvm.func @direct_call(
// POS-SAME: %[[CALL_ARG:.+]]: i64)
// POS: %[[CALL_RESULT:.+]] = llvm.call @forward_alias(%[[CALL_ARG]]) : (i64) -> i64
// POS: llvm.call @wafer_tx81_rdma(%[[CALL_RESULT]],
// POS-LABEL: llvm.func @nested_subview(
// POS-SAME: %[[NESTED_BASE:.+]]: i64)
// POS: %[[FIRST_DELTA:.+]] = llvm.mlir.constant(4 : i64) : i64
// POS: %[[FIRST_ADDR:.+]] = llvm.add %[[NESTED_BASE]], %[[FIRST_DELTA]] : i64
// POS: %[[SECOND_DELTA:.+]] = llvm.mlir.constant(2 : i64) : i64
// POS: %[[SECOND_ADDR:.+]] = llvm.add %[[FIRST_ADDR]], %[[SECOND_DELTA]] : i64
// POS: llvm.call @wafer_tx81_rdma(%[[SECOND_ADDR]],
// POS-LABEL: llvm.func @tile_boundary_yield
// POS-NOT: wafer.tile
// POS: llvm.call @wafer_tx81_local_fence
// POS: llvm.return
// POS-LABEL: llvm.func @loop_carried_alias
// POS: llvm.cond_br
// POS: llvm.br
// POS: llvm.return
// POS-LABEL: llvm.func @mask_uint32_boundary
// POS: llvm.call @wafer_tx81_mask_move({{.*}}) : (i64, i32, i64, i32, i32) -> ()

// LLVMIR-LABEL: define void @false_branch()
// LLVMIR: br i1
// LLVMIR-LABEL: define void @two_trip_nested()
// LLVMIR: br i1
// LLVMIR-LABEL: define void @diamond_cfg()
// LLVMIR: br i1
// LLVMIR-LABEL: define i64 @forward_alias(
// LLVMIR-SAME: i64 %[[ARG:.+]])
// LLVMIR: ret i64 %[[ARG]]
// LLVMIR-LABEL: define void @direct_call(
// LLVMIR-SAME: i64 %[[CALL_ARG:.+]])
// LLVMIR: %[[CALL:.+]] = call i64 @forward_alias(i64 %[[CALL_ARG]])
// LLVMIR: call void @wafer_tx81_rdma(i64 %[[CALL]],
// LLVMIR-LABEL: define void @nested_subview(
// LLVMIR: %[[FIRST:.+]] = add i64 %{{.+}}, 4
// LLVMIR: %[[SECOND:.+]] = add i64 %[[FIRST]], 2
// LLVMIR: call void @wafer_tx81_rdma(i64 %[[SECOND]],
// LLVMIR-LABEL: define void @tile_boundary_yield
// LLVMIR: ret void
// LLVMIR-LABEL: define void @loop_carried_alias
// LLVMIR: phi i64
// LLVMIR: ret void
// LLVMIR-LABEL: define void @mask_uint32_boundary()
// LLVMIR: call void @wafer_tx81_mask_move(i64 65536, i32 -8, i64 65792,

//--- indirect.mlir

func.func private @indirect_callee() {
  return
}

func.func @reject_indirect() {
  %callee = func.constant @indirect_callee : () -> ()
  func.call_indirect %callee() : () -> ()
  return
}

// INDIRECT: unsupported_target_call: indirect or unknown call-like operation 'func.call_indirect' is not supported

//--- recursive.mlir

func.func @reject_recursive() {
  func.call @reject_recursive() : () -> ()
  return
}

// RECURSIVE: unsupported_target_call: recursive call to @reject_recursive is not supported

//--- external.mlir

func.func private @external(
    memref<4xf16, #wafer.memory<ddr, tensor>>)

// EXTERNAL: unsupported_target_call: external func.func declarations are not accepted

//--- alias-mismatch.mlir

func.func @reject_alias_mismatch(
    %lhs: memref<4xf16, #wafer.memory<ddr, tensor>>,
    %rhs: memref<4xf16, #wafer.memory<ddr, tensor>>)
    -> memref<4xf16, #wafer.memory<ddr, tensor>> {
  %cond = arith.constant true
  cf.cond_br %cond, ^left, ^right
^left:
  cf.br ^merge(%lhs : memref<4xf16, #wafer.memory<ddr, tensor>>)
^right:
  cf.br ^merge(%rhs : memref<4xf16, #wafer.memory<ddr, tensor>>)
^merge(%result: memref<4xf16, #wafer.memory<ddr, tensor>>):
  return %result : memref<4xf16, #wafer.memory<ddr, tensor>>
}

// ALIAS: unsupported_target_alias: DDR result #0 must provably alias one DDR function argument

//--- scalar-arg.mlir

func.func @reject_scalar_arg(%condition: i1) {
  return
}

// SCALAR: unsupported_target_function: argument #0 must be a Wafer DDR memref target binding

//--- ddr-alloc.mlir

func.func @reject_ddr_alloc() {
  %ddr = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<4096>}
      : memref<4xf16, #wafer.memory<ddr, tensor>>
  return
}

// DDR-ALLOC: unsupported_target_address: compiler-managed DDR allocation requires an explicit arena base binding

//--- mask-address-overflow.mlir

func.func @reject_mask_address_overflow() {
  %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %mask = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<4294967289>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.mask_move %src, %mask into %dst
      : memref<4xf16, #wafer.memory<spm, tensor>>,
        memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<4xf16, #wafer.memory<spm, tensor>>
  return
}

// MASK-ADDRESS: target_abi_narrowing: mask physical address range [4294967289, 4294967296] must fit uint32_t

//--- factorize.mlir

func.func @reject_factorize() {
  %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %dst0 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %dst1 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %dst2 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<factorize>
      %src into %dst0, %dst1, %dst2 {elem_count = 4 : i64}
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<4xf16, #wafer.memory<spm, tensor>>,
         memref<4xf16, #wafer.memory<spm, tensor>>,
         memref<4xf16, #wafer.memory<spm, tensor>>
  return
}

// FACTORIZE: unsupported_target_operation: peripheral factorize lacks a proven production target semantic profile

//--- dte-atomic.mlir

func.func @valid_before_transport_failure() {
  wafer.instr.local_fence
  return
}

func.func @transport_failure() {
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %token = wafer.instr.dte_send %spm {peer = 1 : i64, bytes = 8 : i64}
      : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
  return
}

// DTE-ATOMIC: unsupported_target_transport: DTE instruction requires a physical transport/endpoint binding and target CRT support
// DTE-ATOMIC: module {
// DTE-ATOMIC: func.func @valid_before_transport_failure
// DTE-ATOMIC: wafer.instr.local_fence
// DTE-ATOMIC: func.func @transport_failure
// DTE-ATOMIC: wafer.instr.dte_send

//--- late-atomic.mlir

func.func @valid_before_late_failure() {
  wafer.instr.local_fence
  return
}

func.func @late_failure() {
  %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<transpose> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 3, 2>,
       permutation = array<i64: 0, 1, 3, 2>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
  return
}

// LATE-ATOMIC: unsupported_target_instr: transform-like tdma_data_move kind reached target LLVM lowering
// LATE-ATOMIC: module {
// LATE-ATOMIC: func.func @valid_before_late_failure
// LATE-ATOMIC: wafer.instr.local_fence
// LATE-ATOMIC: func.func @late_failure
// LATE-ATOMIC: wafer.instr.tdma_data_move
