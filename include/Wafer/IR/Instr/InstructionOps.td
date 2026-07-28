//===- InstructionOps.td - Wafer instruction op definitions ---*- tablegen -*-===//

#ifndef WAFER_IR_INSTR_INSTRUCTIONOPS_TD
#define WAFER_IR_INSTR_INSTRUCTIONOPS_TD

class Wafer_InstrOp<string mnemonic, list<Trait> traits>
    : Wafer_Op<mnemonic, traits # [
        DeclareOpInterfaceMethods<WaferInstructionOpInterface>
      ]>;

class Wafer_NCCInstrOp<string mnemonic, list<Trait> traits>
    : Wafer_InstrOp<mnemonic, traits # [
        DeclareOpInterfaceMethods<WaferNCCIssueOpInterface>
      ]>;

class Wafer_InstrDMAOp<string mnemonic, list<Trait> traits>
    : Wafer_NCCInstrOp<mnemonic, traits> {
  let hasVerifier = 1;
}

def Wafer_InstrRDMAOp : Wafer_InstrDMAOp<"instr.rdma", [
    MemoryEffects<[MemRead<Wafer_DDRResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "instruction-level DDR to SPM RDMA movement";
  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    I64Attr:$byte_count,
    I64Attr:$inner_bytes,
    OptionalAttr<I64Attr>:$src_offset,
    OptionalAttr<I64Attr>:$dst_offset,
    DenseI64ArrayAttr:$src_strides,
    DenseI64ArrayAttr:$src_iterations,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let assemblyFormat =
      "$source `to` $dest attr-dict `:` type($source) `to` type($dest)";
}

def Wafer_InstrWDMAOp : Wafer_InstrDMAOp<"instr.wdma", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_DDRResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "instruction-level SPM to DDR WDMA movement";
  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    I64Attr:$byte_count,
    I64Attr:$inner_bytes,
    OptionalAttr<I64Attr>:$src_offset,
    OptionalAttr<I64Attr>:$dst_offset,
    DenseI64ArrayAttr:$dst_strides,
    DenseI64ArrayAttr:$dst_iterations,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let assemblyFormat =
      "$source `to` $dest attr-dict `:` type($source) `to` type($dest)";
}

def Wafer_InstrGatherScatterOp : Wafer_InstrDMAOp<"instr.gather_scatter", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "instruction-level TDMA SPM gather/scatter movement";
  let description = [{
    `src_offset` and `dst_offset` are byte offsets within the source and
    destination operand buffers. They do not assign physical SPM base addresses;
    SPM memory planning still owns the final buffer base/range.
  }];
  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    I64Attr:$byte_count,
    I64Attr:$inner_bytes,
    OptionalAttr<I64Attr>:$src_offset,
    OptionalAttr<I64Attr>:$dst_offset,
    DenseI64ArrayAttr:$src_strides,
    DenseI64ArrayAttr:$src_iterations,
    DenseI64ArrayAttr:$dst_strides,
    DenseI64ArrayAttr:$dst_iterations,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let assemblyFormat =
      "$source `to` $dest attr-dict `:` type($source) `to` type($dest)";
}

def Wafer_InstrFillOp : Wafer_NCCInstrOp<"instr.fill", [
    MemoryEffects<[MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "instruction-level TDMA scalar fill";
  let arguments = (ins
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    AnyType:$value,
    OptionalAttr<Wafer_FillDomainAttr>:$fill_domain,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$dest `,` $value attr-dict `:` type($dest) `,` type($value)";
}

def Wafer_InstrElementwiseOp : Wafer_NCCInstrOp<"instr.elementwise", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level CT elementwise compute";
  let arguments = (ins
    Wafer_InstrElementwiseKindAttr:$kind,
    Arg<Variadic<AnyMemRef>, "", [MemRead]>:$inputs,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $inputs `into` $dest attr-dict `:` type($inputs) `into` type($dest)";
}

def Wafer_InstrBit2FpOp : Wafer_NCCInstrOp<"instr.bit2fp", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level CT bit-mask to floating mask conversion";
  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `into` $dest attr-dict `:` type($source) `to` type($dest)";
}

def Wafer_InstrMaskMoveOp : Wafer_NCCInstrOp<"instr.mask_move", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level CT masked SPM data move";
  let description = [{
    When the mask element is true/non-zero, copy the corresponding source
    element into dest. When the mask element is false/zero, dest keeps the
    value already present there. This mirrors the TX81 `mask_move` target op
    and is not a generic ternary select instruction.
  }];
  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemRead]>:$mask,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `,` $mask `into` $dest attr-dict `:` type($source) `,` type($mask) `into` type($dest)";
}

def Wafer_InstrReduceOp : Wafer_NCCInstrOp<"instr.reduce", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level CT local reduce";
  let arguments = (ins
    Wafer_InstrReduceKindAttr:$kind,
    Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    I64Attr:$dim,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input `into` $dest attr-dict `:` type($input) `into` type($dest)";
}

def Wafer_InstrConvertOp : Wafer_NCCInstrOp<"instr.convert", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level CT dtype conversion";
  let arguments = (ins
    Wafer_InstrConvertKindAttr:$kind,
    Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    OptionalAttr<I64Attr>:$zero_point,
    OptionalAttr<I64Attr>:$rounding_mode,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $source `into` $dest attr-dict `:` type($source) `to` type($dest)";
}

def Wafer_InstrGemmOp : Wafer_NCCInstrOp<"instr.gemm", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level NE GEMM";
  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    I64Attr:$m,
    I64Attr:$k,
    I64Attr:$n,
    OptionalAttr<Wafer_GemmOrientationAttr>:$lhs_orientation,
    OptionalAttr<Wafer_GemmOrientationAttr>:$rhs_orientation,
    OptionalAttr<I64Attr>:$batch_count,
    OptionalAttr<DenseI64ArrayAttr>:$lhs_batch_dims,
    OptionalAttr<I64Attr>:$lhs_m_dim,
    OptionalAttr<I64Attr>:$lhs_contracting_dim,
    OptionalAttr<DenseI64ArrayAttr>:$rhs_batch_dims,
    OptionalAttr<I64Attr>:$rhs_contracting_dim,
    OptionalAttr<I64Attr>:$rhs_n_dim,
    OptionalAttr<DenseI64ArrayAttr>:$result_batch_dims,
    OptionalAttr<I64Attr>:$result_m_dim,
    OptionalAttr<I64Attr>:$result_n_dim,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$lhs `,` $rhs `into` $dest attr-dict `:` type($lhs) `,` type($rhs) `into` type($dest)";
}

def Wafer_InstrConvOp : Wafer_NCCInstrOp<"instr.conv", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level NE convolution";
  let arguments = (ins
    Wafer_InstrConvKindAttr:$kind,
    Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemRead]>:$weight,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    DenseI64ArrayAttr:$input_shape,
    DenseI64ArrayAttr:$weight_shape,
    DenseI64ArrayAttr:$output_shape,
    DenseI64ArrayAttr:$pads,
    DenseI64ArrayAttr:$unpads,
    DenseI64ArrayAttr:$kernel_strides,
    DenseI64ArrayAttr:$dilations,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input `,` $weight `into` $dest attr-dict `:` type($input) `,` type($weight) `into` type($dest)";
}

def Wafer_InstrPoolOp : Wafer_NCCInstrOp<"instr.pool", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level CT pool";
  let arguments = (ins
    Wafer_InstrPoolKindAttr:$kind,
    Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<Variadic<AnyMemRef>, "", [MemWrite]>:$dests,
    DenseI64ArrayAttr:$source_shape,
    DenseI64ArrayAttr:$dest_shape,
    DenseI64ArrayAttr:$pads,
    DenseI64ArrayAttr:$kernel_strides,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input `into` $dests attr-dict `:` type($input) `into` type($dests)";
}

def Wafer_InstrUnpoolOp : Wafer_NCCInstrOp<"instr.unpool", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level CT unpool";
  let arguments = (ins
    Wafer_InstrUnpoolKindAttr:$kind,
    Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<Optional<AnyMemRef>, "", [MemRead]>:$index,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    DenseI64ArrayAttr:$source_shape,
    DenseI64ArrayAttr:$dest_shape,
    DenseI64ArrayAttr:$kernel_strides,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input (`,` $index^)? `into` $dest attr-dict `:` type($input) "
      "(`,` type($index)^)? `into` type($dest)";
}

def Wafer_InstrTDMADataMoveOp : Wafer_NCCInstrOp<"instr.tdma_data_move", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "instruction-level TDMA structured data movement";
  let arguments = (ins
    Wafer_InstrDataMoveKindAttr:$kind,
    Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$dest,
    DenseI64ArrayAttr:$source_shape,
    DenseI64ArrayAttr:$dest_shape,
    OptionalAttr<DenseI64ArrayAttr>:$permutation,
    OptionalAttr<DenseI64ArrayAttr>:$axes,
    OptionalAttr<DenseI64ArrayAttr>:$pads,
    OptionalAttr<DenseI64ArrayAttr>:$kernel_strides,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $source `into` $dest attr-dict `:` type($source) `to` type($dest)";
}

def Wafer_InstrPeripheralOp : Wafer_NCCInstrOp<"instr.peripheral", [
    AttrSizedOperandSegments,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "instruction-level CT peripheral operation";
  let arguments = (ins
    Wafer_InstrPeripheralKindAttr:$kind,
    Arg<Variadic<AnyMemRef>, "", [MemRead]>:$inputs,
    Arg<Variadic<AnyMemRef>, "", [MemWrite]>:$dests,
    I64Attr:$elem_count,
    OptionalAttr<DenseI64ArrayAttr>:$source_shape,
    OptionalAttr<DenseI64ArrayAttr>:$dest_shape,
    OptionalAttr<I64Attr>:$lut_elem_count,
    OptionalAttr<I64Attr>:$scale,
    OptionalAttr<I64Attr>:$probability,
    OptionalAttr<I64Attr>:$rounding_mode,
    DefaultValuedAttr<Wafer_NCCWorkerAttr,
                      "::wafer::NCCWorker::Worker0">:$worker
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $inputs `into` $dests attr-dict `:` type($inputs) `into` type($dests)";
}

#endif // WAFER_IR_INSTR_INSTRUCTIONOPS_TD
