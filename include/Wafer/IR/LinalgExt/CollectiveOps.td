//===- CollectiveOps.td - Wafer linalg extension collectives -*- tablegen -*-===//

#ifndef WAFER_IR_LINALGEXT_COLLECTIVEOPS_TD
#define WAFER_IR_LINALGEXT_COLLECTIVEOPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"

class Wafer_LinalgExtCollectiveOp<string mnemonic, list<Trait> traits = []>
    : Wafer_Op<mnemonic, !listconcat([
        AttrSizedOperandSegments,
        DestinationStyleOpInterface,
        TilingInterface,
        DeclareOpInterfaceMethods<WaferLinalgExtCollectiveOpInterface>
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
    ::llvm::SmallVector<::mlir::utils::IteratorType>
    getLoopIteratorTypes();
    ::llvm::SmallVector<::mlir::Range>
    getIterationDomain(::mlir::OpBuilder &builder);
    ::mlir::FailureOr<::mlir::TilingResult>
    getTiledImplementation(::mlir::OpBuilder &builder,
                           ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
                           ::llvm::ArrayRef<::mlir::OpFoldResult> sizes);
    ::mlir::LogicalResult
    getResultTilePosition(::mlir::OpBuilder &builder, unsigned resultNumber,
                          ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
                          ::llvm::ArrayRef<::mlir::OpFoldResult> sizes,
                          ::llvm::SmallVector<::mlir::OpFoldResult> &resultOffsets,
                          ::llvm::SmallVector<::mlir::OpFoldResult> &resultSizes);
  }];
}

class Wafer_LinalgExtCollectivePartitionGroupOp<string mnemonic,
                                                list<Trait> traits = []>
    : Wafer_LinalgExtCollectiveOp<mnemonic, traits> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outs,
    I64Attr:$axis,
    OptionalAttr<DenseI64ArrayAttr>:$partition_group,
    OptionalAttr<I64ElementsAttr>:$partition_groups,
    OptionalAttr<I64Attr>:$channel_id,
    OptionalAttr<BoolAttr>:$use_global_device_ids
  );
}

def Wafer_LinalgExtCollectiveAllGatherOp
    : Wafer_LinalgExtCollectivePartitionGroupOp<"linalg_ext.collective.all_gather"> {
  let summary = "post-SPMD all-gather with linalg semantics";
  let description = [{
    Carries StableHLO all_gather semantics after SPMD partitioning as a
    destination-style tensor op. It preserves logical partition group and gather
    axis facts for group/tiling. `partition_group` entries are logical card
    partition IDs in the current `wafer.execution.mesh`; if the enclosing
    module has an execution mesh, the verifier checks those IDs against the
    mesh partition domain. The op does not own physical endpoint mapping, SPM buffers, DTE
    tokens, byte schedules, or runtime handles.
  }];
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict `->` type($results)
  }];
}

def Wafer_LinalgExtCollectiveReduceScatterOp
    : Wafer_LinalgExtCollectivePartitionGroupOp<"linalg_ext.collective.reduce_scatter",
        [SingleBlockImplicitTerminator<"LinalgExtCollectiveYieldOp">]> {
  let summary = "post-SPMD reduce-scatter with linalg semantics";
  let description = [{
    Carries linalg-extension reduce-scatter semantics after SPMD partitioning.
    `partition_group` entries are logical card partition IDs in the current
    `wafer.execution.mesh`; physical endpoint selection is deferred to
    topology/execution-mesh consumers.
  }];
  let regions = (region SizedRegion<1>:$combiner);
  let hasRegionVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    $combiner attr-dict `->` type($results)
  }];
}

def Wafer_LinalgExtCollectiveAllReduceOp
    : Wafer_LinalgExtCollectiveOp<"linalg_ext.collective.all_reduce",
        [SingleBlockImplicitTerminator<"LinalgExtCollectiveYieldOp">]> {
  let summary = "post-SPMD all-reduce with linalg semantics";
  let description = [{
    Carries linalg-extension all-reduce semantics after SPMD partitioning.
    `partition_group` entries are logical card partition IDs in the current
    `wafer.execution.mesh`; the verifier checks the mesh partition domain when an
    execution mesh is present. Physical routes and endpoints are not encoded in
    this op.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outs,
    OptionalAttr<DenseI64ArrayAttr>:$partition_group,
    OptionalAttr<I64ElementsAttr>:$partition_groups,
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

def Wafer_LinalgExtCollectiveAllToAllOp
    : Wafer_LinalgExtCollectiveOp<"linalg_ext.collective.all_to_all"> {
  let summary = "post-SPMD all-to-all with linalg semantics";
  let description = [{
    Carries linalg-extension all-to-all semantics after SPMD partitioning.
    `partition_group` entries are logical card partition IDs in the current
    `wafer.execution.mesh`; physical exchange endpoints are derived later from
    the execution mesh and target topology.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outs,
    I64Attr:$split_axis,
    I64Attr:$concat_axis,
    I64Attr:$split_count,
    OptionalAttr<DenseI64ArrayAttr>:$partition_group,
    OptionalAttr<I64ElementsAttr>:$partition_groups,
    OptionalAttr<I64Attr>:$channel_id,
    OptionalAttr<BoolAttr>:$use_global_device_ids
  );
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict `->` type($results)
  }];
}

def Wafer_LinalgExtCollectiveCollectivePermuteOp
    : Wafer_LinalgExtCollectiveOp<"linalg_ext.collective.collective_permute"> {
  let summary = "post-SPMD collective-permute with linalg semantics";
  let description = [{
    Carries linalg-extension collective-permute semantics after SPMD partitioning.
    `source_target_pairs` contains source/target logical card partition pairs,
    not physical endpoint encodings. If the enclosing module has an execution
    mesh, the verifier checks every pair entry against the mesh partition domain.
  }];
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

def Wafer_LinalgExtCollectiveYieldOp : Wafer_Op<"linalg_ext.collective.yield", [
    ParentOneOf<["LinalgExtCollectiveAllReduceOp",
                 "LinalgExtCollectiveReduceScatterOp"]>,
    Pure,
    ReturnLike,
    Terminator
  ]> {
  let summary = "yield scalar combiner values from a linalg-ext collective body";
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{ attr-dict ($values^ `:` type($values))? }];
}

#endif // WAFER_IR_LINALGEXT_COLLECTIVEOPS_TD
