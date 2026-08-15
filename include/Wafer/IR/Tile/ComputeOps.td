//===- ComputeOps.td - Wafer Compute operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMPUTEOPS_TD
#define WAFER_IR_OPS_COMPUTEOPS_TD

def Wafer_ComputeFillOp : Wafer_Op<"tile.fill", [
    WaferTileDataflowOpInterface,
    MemoryEffects<[MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract tile fill compute";
  let description = [{
    `wafer.tile.fill` initializes an existing tile-local destination storage
    with a scalar value. The operation is explicit because the initialized
    buffer participates in SPM lifetime and resource planning; it does not
    assign a physical SPM offset.
  }];

  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemWrite]>:$dest,
    AnyType:$value,
    OptionalAttr<Wafer_FillDomainAttr>:$fill_domain
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$dest `,` $value attr-dict `:` type($dest) `,` type($value)";
}

def Wafer_ComputeConvertOp : Wafer_Op<"tile.compute.convert", [
    WaferTileDataflowOpInterface,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract tile element-type conversion";
  let description = [{
    `wafer.tile.compute.convert` changes the element type of one tile-local SPM
    tensor while preserving its logical shape and tensor layout.  It records
    the numeric cast before instruction lowering selects a target conversion
    route and any route-specific parameter.
  }];

  let arguments = (ins Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source);
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source attr-dict `:` type($source) `to` type($result)";
}

def Wafer_ComputeGemmOp : Wafer_Op<"tile.gemm", [
    WaferTileDataflowOpInterface,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract tile GEMM";
  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$lhs,
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$rhs,
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
    OptionalAttr<I64Attr>:$result_n_dim
  );
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Wafer_ComputeConvOp : Wafer_Op<"tile.conv", [
    WaferTileDataflowOpInterface,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract ordinary 2-D tile convolution";
  let description = [{
    `wafer.tile.conv` is the canonical tile-local ordinary 2-D convolution
    contract. Input and result storage use logical `[N, H, W, C]` order and
    weight storage uses logical `[X, Y, O, I]` order. `strides` and
    `dilations` use logical `[H, W]` order; `pads` and `unpads` use
    `[H-before, H-after, W-before, W-after]` order. The operation remains
    target-abstract: instruction field packing and register bounds belong to
    instruction lowering.
  }];

  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$input,
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$weight,
    DenseI64ArrayAttr:$pads,
    DenseI64ArrayAttr:$unpads,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$dilations
  );
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$input `,` $weight attr-dict `:` functional-type(operands, results)";
}

def Wafer_ComputeElementwiseOp : Wafer_Op<"tile.elementwise", [
    WaferTileDataflowOpInterface,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract tile elementwise compute";
  let description = [{
    `wafer.tile.elementwise` represents tile-local scalar elementwise
    compute over already materialized SPM storage values. The operation keeps
    its operand/result relation explicit: without `indexing_maps`, all operands
    and the result must have the same logical tensor type; with
    `indexing_maps`, operands may use verifier-proven permutation-only
    broadcast from result dimensions. All operands and the result must use SPM
    tensor storage values with matching element types.
  }];

  let arguments = (ins
    Wafer_ComputeElementwiseKindAttr:$kind,
    Arg<Variadic<Wafer_AnySPMMemRef>, "", [MemRead]>:$inputs,
    OptionalAttr<AffineMapArrayAttr>:$indexing_maps
  );
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $inputs attr-dict `:` functional-type(operands, results)";
}

def Wafer_ComputeReduceOp : Wafer_Op<"tile.reduce", [
    WaferTileDataflowOpInterface,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract tile-local reduction compute";
  let description = [{
    `wafer.tile.reduce` represents a tile-local reduction over an SPM tensor
    tile. The result shape is the input shape with `dimensions` removed, and
    exactly one constant scalar initialization source is required. It is
    represented either by a typed `init_value` attribute or by an SSA `init`
    directly defined by `arith.constant`; both forms must match the input
    element type.
  }];

  let arguments = (ins
    Wafer_ComputeReduceKindAttr:$kind,
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$input,
    Optional<AnyType>:$init,
    DenseI64ArrayAttr:$dimensions,
    OptionalAttr<TypedAttrInterface>:$init_value
  );
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input (`,` $init^)? attr-dict `:` functional-type(operands, results)";
}

#endif // WAFER_IR_OPS_COMPUTEOPS_TD
