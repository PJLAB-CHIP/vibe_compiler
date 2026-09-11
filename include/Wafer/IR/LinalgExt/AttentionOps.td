//===- AttentionOps.td - Wafer structured attention op ----*- tablegen -*-===//

#ifndef WAFER_IR_LINALGEXT_ATTENTIONOPS_TD
#define WAFER_IR_LINALGEXT_ATTENTIONOPS_TD

def Wafer_AttentionShaped
    : AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

def Wafer_LinalgExtAttentionYieldOp : Wafer_Op<"linalg_ext.attention.yield", [
    ParentOneOf<["LinalgExtAttentionOp", "LinalgExtOnlineAttentionOp"]>,
    Pure, ReturnLike, Terminator
  ]> {
  let summary = "yield the adjusted scalar attention score";
  let arguments = (ins AnyFloat:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def Wafer_LinalgExtAttentionOp
    : Wafer_Op<"linalg_ext.attention", [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
        TilingInterface,
        DeclareOpInterfaceMethods<WaferCoupledReductionOpInterface>
      ]> {
  let summary = "normalized scaled dot-product attention";
  let description = [{
    Represents a complete forward scaled dot-product attention root over
    query, key, value, an optional additive mask, and one destination. The
    indexing maps define the batch, query, Q/K contraction, K/V normalization,
    and value/output coordinates. `algorithm` is a graph fact established by
    normalization; block sizes, physical partitions, placement, storage, and
    schedules are not part of this operation.

    The operation has one final result in tensor form and no result in buffer
    form. Its three online reduction components are exposed only through
    WaferCoupledReductionOpInterface and are materialized only after the
    Spatial/Region choice, not as extra graph-level results.
  }];

  let arguments = (ins
    Wafer_AttentionShaped:$query,
    Wafer_AttentionShaped:$key,
    Wafer_AttentionShaped:$value,
    AnyFloat:$scale,
    Optional<Wafer_AttentionShaped>:$mask,
    Wafer_AttentionShaped:$output,
    Wafer_AttentionAlgorithmAttr:$algorithm,
    AffineMapArrayAttr:$indexing_maps
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$score_region);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $query `,` $key `,` $value `,` $scale (`,` $mask^)? `:`
      type($query) `,` type($key) `,` type($value) `,` type($scale)
      (`,` type($mask)^)? `)`
    `outs` `(` $output `:` type($output) `)`
    `algorithm` `(` $algorithm `)`
    `indexing_maps` `=` $indexing_maps
    `score` $score_region
    attr-dict
    (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }

    ::llvm::SmallVector<::mlir::AffineMap, 6> getIndexingMapsArray();
    ::llvm::SmallVector<int64_t> getStaticLoopRanges();
    ::mlir::Type getScoreType();
    ::mlir::FailureOr<::wafer::AttentionIterationRoles>
    getIterationRoles();

    ::llvm::SmallVector<::mlir::utils::IteratorType>
    getLoopIteratorTypes();
    ::llvm::SmallVector<::mlir::Range>
    getIterationDomain(::mlir::OpBuilder &builder);
    ::mlir::FailureOr<::mlir::TilingResult>
    getTiledImplementation(
        ::mlir::OpBuilder &builder,
        ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
        ::llvm::ArrayRef<::mlir::OpFoldResult> sizes);
    ::mlir::LogicalResult getResultTilePosition(
        ::mlir::OpBuilder &builder, unsigned resultNumber,
        ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
        ::llvm::ArrayRef<::mlir::OpFoldResult> sizes,
        ::llvm::SmallVector<::mlir::OpFoldResult> &resultOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult> &resultSizes);
    ::mlir::FailureOr<::mlir::TilingResult> generateResultTileValue(
        ::mlir::OpBuilder &builder, unsigned resultNumber,
        ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
        ::llvm::ArrayRef<::mlir::OpFoldResult> sizes);

    ::mlir::AffineMap getQueryMap() { return getIndexingMapsArray()[0]; }
    ::mlir::AffineMap getKeyMap() { return getIndexingMapsArray()[1]; }
    ::mlir::AffineMap getValueMap() { return getIndexingMapsArray()[2]; }
    ::mlir::AffineMap getScaleMap() { return getIndexingMapsArray()[3]; }
    ::std::optional<::mlir::AffineMap> getMaskMap() {
      if (!getMask())
        return ::std::nullopt;
      return getIndexingMapsArray()[4];
    }
    ::mlir::AffineMap getOutputMap() {
      return getIndexingMapsArray()[getMask() ? 5 : 4];
    }
    int64_t getIterationDomainRank() {
      return getQueryMap().getNumDims();
    }
  }];
}

def Wafer_LinalgExtOnlineAttentionOp
    : Wafer_Op<"linalg_ext.online_attention", [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
        TilingInterface,
        DeclareOpInterfaceMethods<WaferCoupledReductionOpInterface>
      ]> {
  let summary = "stateful online attention update";
  let description = [{
    Represents one current online-attention state update over query, key, and
    value tiles. The three destination-style results are the accumulator,
    maximum, and sum states. Unlike the graph-level attention operation, this
    operation may be tiled along the K/V reduction dimensions because each
    tile consumes the previous states and returns the updated states.

    This operation exists only in candidate-owned structural tensor IR. It
    carries no algorithm, Tile, layout, movement, resource, or completion
    choice.
  }];

  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    AnyFloat:$scale,
    Optional<AnyRankedTensor>:$mask,
    AnyRankedTensor:$accumulator,
    AnyRankedTensor:$maximum,
    AnyRankedTensor:$sum,
    AffineMapArrayAttr:$indexing_maps
  );
  let results = (outs
    AnyRankedTensor:$updated_accumulator,
    AnyRankedTensor:$updated_maximum,
    AnyRankedTensor:$updated_sum
  );
  let regions = (region AnyRegion:$score_region);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $query `,` $key `,` $value `,` $scale (`,` $mask^)? `:`
      type($query) `,` type($key) `,` type($value) `,` type($scale)
      (`,` type($mask)^)? `)`
    `outs` `(` $accumulator `,` $maximum `,` $sum `:`
      type($accumulator) `,` type($maximum) `,` type($sum) `)`
    `indexing_maps` `=` $indexing_maps
    `score` $score_region
    attr-dict
    `->` `(` type($updated_accumulator) `,` type($updated_maximum) `,`
             type($updated_sum) `)`
  }];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable();

    ::llvm::SmallVector<::mlir::AffineMap, 8> getIndexingMapsArray();
    ::llvm::SmallVector<int64_t> getStaticLoopRanges();
    ::mlir::Type getScoreType();
    ::mlir::FailureOr<::wafer::AttentionIterationRoles>
    getIterationRoles();

    ::llvm::SmallVector<::mlir::utils::IteratorType>
    getLoopIteratorTypes();
    ::llvm::SmallVector<::mlir::Range>
    getIterationDomain(::mlir::OpBuilder &builder);
    ::mlir::FailureOr<::mlir::TilingResult>
    getTiledImplementation(
        ::mlir::OpBuilder &builder,
        ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
        ::llvm::ArrayRef<::mlir::OpFoldResult> sizes);
    ::mlir::LogicalResult getResultTilePosition(
        ::mlir::OpBuilder &builder, unsigned resultNumber,
        ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
        ::llvm::ArrayRef<::mlir::OpFoldResult> sizes,
        ::llvm::SmallVector<::mlir::OpFoldResult> &resultOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult> &resultSizes);

    ::mlir::AffineMap getQueryMap() { return getIndexingMapsArray()[0]; }
    ::mlir::AffineMap getKeyMap() { return getIndexingMapsArray()[1]; }
    ::mlir::AffineMap getValueMap() { return getIndexingMapsArray()[2]; }
    ::mlir::AffineMap getScaleMap() { return getIndexingMapsArray()[3]; }
    ::std::optional<::mlir::AffineMap> getMaskMap() {
      if (!getMask())
        return ::std::nullopt;
      return getIndexingMapsArray()[4];
    }
    unsigned getAccumulatorMapIndex() { return getMask() ? 5 : 4; }
    ::mlir::AffineMap getAccumulatorMap() {
      return getIndexingMapsArray()[getAccumulatorMapIndex()];
    }
    ::mlir::AffineMap getMaximumMap() {
      return getIndexingMapsArray()[getAccumulatorMapIndex() + 1];
    }
    ::mlir::AffineMap getSumMap() {
      return getIndexingMapsArray()[getAccumulatorMapIndex() + 2];
    }
    int64_t getIterationDomainRank() {
      return getQueryMap().getNumDims();
    }
  }];
}

#endif // WAFER_IR_LINALGEXT_ATTENTIONOPS_TD
