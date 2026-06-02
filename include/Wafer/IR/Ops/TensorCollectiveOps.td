//===- TensorCollectiveOps.td - Wafer tensor collective ops -*- tablegen -*-===//

#ifndef WAFER_IR_OPS_TENSORCOLLECTIVEOPS_TD
#define WAFER_IR_OPS_TENSORCOLLECTIVEOPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"

class Wafer_TensorCollectiveOp<string mnemonic, list<Trait> traits = []>
    : Wafer_Op<mnemonic, !listconcat([
        AttrSizedOperandSegments,
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<WaferTilingInterface>
      ], traits)> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outs
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutsMutable();
    }
  }];
}

class Wafer_TensorCollectiveRankGroupOp<string mnemonic,
                                        list<Trait> traits = []>
    : Wafer_TensorCollectiveOp<mnemonic, traits> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outs,
    I64Attr:$axis,
    DenseI64ArrayAttr:$rank_group,
    OptionalAttr<I64Attr>:$channel_id,
    OptionalAttr<BoolAttr>:$use_global_device_ids
  );
}

def Wafer_TensorCollectiveAllGatherOp
    : Wafer_TensorCollectiveRankGroupOp<"tensor_collective.all_gather"> {
  let summary = "tensor-level post-SPMD all-gather handoff";
  let description = [{
    Carries StableHLO all_gather semantics after SPMD partitioning as a
    destination-style tensor op. It preserves logical rank group and gather
    axis facts for group/tiling, but does not own physical placement, SPM
    buffers, DTE tokens, byte schedules, or runtime handles.
  }];
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict `->` type($results)
  }];
}

def Wafer_TensorCollectiveReduceScatterOp
    : Wafer_TensorCollectiveRankGroupOp<"tensor_collective.reduce_scatter",
        [SingleBlockImplicitTerminator<"TensorCollectiveYieldOp">]> {
  let summary = "tensor-level post-SPMD reduce-scatter handoff";
  let regions = (region SizedRegion<1>:$combiner);
  let hasRegionVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    $combiner attr-dict `->` type($results)
  }];
}

def Wafer_TensorCollectiveAllReduceOp
    : Wafer_TensorCollectiveOp<"tensor_collective.all_reduce",
        [SingleBlockImplicitTerminator<"TensorCollectiveYieldOp">]> {
  let summary = "tensor-level post-SPMD all-reduce handoff";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outs,
    DenseI64ArrayAttr:$rank_group,
    OptionalAttr<I64Attr>:$channel_id,
    OptionalAttr<BoolAttr>:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$combiner);
  let hasRegionVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    $combiner attr-dict `->` type($results)
  }];
}

def Wafer_TensorCollectiveAllToAllOp
    : Wafer_TensorCollectiveOp<"tensor_collective.all_to_all"> {
  let summary = "tensor-level post-SPMD all-to-all handoff";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outs,
    I64Attr:$split_axis,
    I64Attr:$concat_axis,
    I64Attr:$split_count,
    DenseI64ArrayAttr:$rank_group,
    OptionalAttr<I64Attr>:$channel_id,
    OptionalAttr<BoolAttr>:$use_global_device_ids
  );
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict `->` type($results)
  }];
}

def Wafer_TensorCollectiveCollectivePermuteOp
    : Wafer_TensorCollectiveOp<"tensor_collective.collective_permute"> {
  let summary = "tensor-level post-SPMD collective-permute handoff";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outs,
    DenseI64ArrayAttr:$source_target_pairs,
    OptionalAttr<I64Attr>:$channel_id
  );
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict `->` type($results)
  }];
}

def Wafer_TensorCollectiveYieldOp : Wafer_Op<"tensor_collective.yield", [
    ParentOneOf<["TensorCollectiveAllReduceOp",
                 "TensorCollectiveReduceScatterOp"]>,
    Pure,
    ReturnLike,
    Terminator
  ]> {
  let summary = "yield scalar combiner values from a tensor collective body";
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{ attr-dict ($values^ `:` type($values))? }];
}

#endif // WAFER_IR_OPS_TENSORCOLLECTIVEOPS_TD
