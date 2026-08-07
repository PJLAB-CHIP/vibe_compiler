//===- CompleteTraversal.cpp - Complete candidate traversal -----------===//

#include "Internal.h"

#include "mlir/IR/Verifier.h"

#include <algorithm>
#include <iterator>
#include <limits>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

static mlir::Value getCandidateViewSource(mlir::Operation *operation) {
  if (auto slice =
          mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(operation))
    return slice.getSource();
  if (auto expand =
          mlir::dyn_cast_or_null<mlir::tensor::ExpandShapeOp>(operation))
    return expand.getSrc();
  if (auto collapse =
          mlir::dyn_cast_or_null<mlir::tensor::CollapseShapeOp>(operation))
    return collapse.getSrc();
  if (auto cast = mlir::dyn_cast_or_null<mlir::tensor::CastOp>(operation))
    return cast.getSource();
  return {};
}

static mlir::OpResult traceCandidateStructuredProducer(mlir::Value value,
                                                       mlir::Block &body) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return nullptr;
    mlir::Operation *owner = result.getOwner();
    if (owner->getBlock() != &body)
      return {};
    if (classifyCandidateTraversalRoot(owner) !=
        CandidateTraversalRootCapability::Unsupported)
      return result;
    value = getCandidateViewSource(owner);
  }
  return {};
}

/// Maps an observable tensor destination back through the exact shape-only
/// view chain between a returned value and its structured producer.  Joint
/// traversal tiles are expressed in the producer result domain; reversing the
/// views keeps their insert_slice coordinates in that same domain while the
/// original forward views continue to form the public result afterwards.
static mlir::FailureOr<mlir::Value> materializeJointOutputDestination(
    mlir::OpBuilder &builder, mlir::Value returned,
    mlir::Value structuredResult, mlir::Value publicDestination,
    std::string *failureReason) {
  mlir::Value currentReturned = returned;
  mlir::Value currentDestination = publicDestination;
  llvm::DenseSet<mlir::Value> visited;
  while (currentReturned != structuredResult &&
         visited.insert(currentReturned).second) {
    auto result = mlir::dyn_cast<mlir::OpResult>(currentReturned);
    if (!result) {
      setFailureReason(
          failureReason,
          "joint traversal output view left the current SSA graph");
      return mlir::failure();
    }
    if (auto expand =
            mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(result.getOwner())) {
      currentDestination =
          builder
              .create<mlir::tensor::CollapseShapeOp>(
                  expand.getLoc(), expand.getSrcType(), currentDestination,
                  expand.getReassociationIndices())
              .getResult();
      currentReturned = expand.getSrc();
      continue;
    }
    if (auto collapse =
            mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(result.getOwner())) {
      currentDestination =
          builder
              .create<mlir::tensor::ExpandShapeOp>(
                  collapse.getLoc(), collapse.getSrcType(), currentDestination,
                  collapse.getReassociationIndices())
              .getResult();
      currentReturned = collapse.getSrc();
      continue;
    }
    setFailureReason(
        failureReason,
        "joint traversal output view is not an exact static reshape");
    return mlir::failure();
  }
  if (currentReturned != structuredResult ||
      currentDestination.getType() != structuredResult.getType()) {
    setFailureReason(
        failureReason,
        "joint traversal output view does not map to the producer domain");
    return mlir::failure();
  }
  return currentDestination;
}

mlir::FailureOr<llvm::SmallVector<CandidateTraversalConnection, 16>>
collectCandidateTraversalConnections(mlir::func::FuncOp function,
                                     std::string *failureReason) {
  if (!function || function.isExternal() || !function.getBody().hasOneBlock()) {
    setFailureReason(failureReason,
                     "connection traversal requires one defined block");
    return mlir::failure();
  }
  mlir::Block &body = function.getBody().front();
  llvm::SmallVector<CandidateTraversalConnection, 16> connections;
  for (mlir::Operation &consumer : body.without_terminator()) {
    if (classifyCandidateTraversalRoot(&consumer) ==
        CandidateTraversalRootCapability::Unsupported)
      continue;
    auto appendOperandConnection = [&](mlir::OpOperand &operand) {
      mlir::OpResult producerResult =
          traceCandidateStructuredProducer(operand.get(), body);
      mlir::Operation *producer =
          producerResult ? producerResult.getOwner() : nullptr;
      if (!producer || producer == &consumer ||
          !producer->isBeforeInBlock(&consumer))
        return;
      auto duplicate = llvm::find_if(
          connections, [&](const CandidateTraversalConnection &connection) {
            return connection.producerResult == producerResult &&
                   connection.consumerOperand == &operand;
          });
      if (duplicate == connections.end())
        connections.push_back({producerResult, &operand});
    };
    // DPS init values are destination/initialization state, not producer data
    // connections. In particular, a zero fill feeding matmul must remain the
    // typed overwrite initialization proof rather than becoming a fusion or
    // residency action.
    if (auto dps =
            mlir::dyn_cast<mlir::DestinationStyleOpInterface>(&consumer)) {
      for (mlir::OpOperand *input : dps.getDpsInputOperands())
        appendOperandConnection(*input);
    } else {
      for (mlir::OpOperand &operand : consumer.getOpOperands())
        appendOperandConnection(operand);
    }
  }
  return connections;
}

bool CandidateConnectionFusionPolicy::shouldFuse(
    mlir::OpResult producerResult, mlir::OpOperand &consumerOperand) const {
  if (!isValid() || !producerResult)
    return false;
  for (auto [connection, choice] : llvm::zip_equal(connections, choices))
    if (connection.producerResult == producerResult &&
        connection.consumerOperand == &consumerOperand)
      return choice.action ==
             CandidateTraversalConnectionAction::CoupledResident;
  return false;
}

static mlir::FailureOr<uint64_t>
getCandidateOutputTileCount(mlir::RankedTensorType resultType,
                            llvm::ArrayRef<int64_t> candidateTileSizes,
                            std::string *failureReason) {
  if (candidateTileSizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "candidate tile rank does not match program result rank");
    return mlir::failure();
  }

  for (auto [bound, tileSize] :
       llvm::zip(resultType.getShape(), candidateTileSizes)) {
    if (mlir::ShapedType::isDynamic(bound)) {
      setFailureReason(failureReason,
                       "complete candidate traversal requires static result "
                       "shape");
      return mlir::failure();
    }
    if (bound <= 0 || tileSize <= 0 || tileSize > bound) {
      setFailureReason(
          failureReason,
          "complete candidate traversal tile size is outside result bounds");
      return mlir::failure();
    }
  }

  uint64_t tileCount = 0;
  switch (wafer::detail::checkedStaticTileProduct(
      resultType.getShape(), candidateTileSizes, tileCount)) {
  case wafer::detail::CheckedStaticTileProductStatus::Success:
    return tileCount;
  case wafer::detail::CheckedStaticTileProductStatus::Overflow:
    setFailureReason(
        failureReason,
        "complete candidate traversal output tile count is not representable");
    return mlir::failure();
  case wafer::detail::CheckedStaticTileProductStatus::InvalidInput:
    setFailureReason(
        failureReason,
        "complete candidate traversal has invalid static output ranges");
    return mlir::failure();
  }
  llvm_unreachable("unknown checked tile product status");
}

static bool haveEquivalentExternalSlices(mlir::tensor::ExtractSliceOp lhs,
                                         mlir::tensor::ExtractSliceOp rhs,
                                         TensorProgramScope scope) {
  auto lhsSource = mlir::dyn_cast<mlir::BlockArgument>(lhs.getSource());
  auto rhsSource = mlir::dyn_cast<mlir::BlockArgument>(rhs.getSource());
  if (!lhsSource || !rhsSource || lhsSource != rhsSource ||
      lhsSource.getOwner() != &scope.getBody())
    return false;
  return lhs.getType() == rhs.getType() &&
         llvm::equal(lhs.getMixedOffsets(), rhs.getMixedOffsets()) &&
         llvm::equal(lhs.getMixedSizes(), rhs.getMixedSizes()) &&
         llvm::equal(lhs.getMixedStrides(), rhs.getMixedStrides());
}

static void deduplicateExternalSlicesInBlock(mlir::Block &block,
                                             TensorProgramScope scope) {
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> canonicalSlices;
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> duplicateSlices;
  for (mlir::Operation &op : block.without_terminator()) {
    auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op);
    if (!slice)
      continue;
    auto existing = llvm::find_if(canonicalSlices, [&](auto candidate) {
      return haveEquivalentExternalSlices(candidate, slice, scope);
    });
    if (existing == canonicalSlices.end()) {
      canonicalSlices.push_back(slice);
      continue;
    }
    slice.getResult().replaceAllUsesWith(existing->getResult());
    duplicateSlices.push_back(slice);
  }
  for (mlir::tensor::ExtractSliceOp duplicate : duplicateSlices)
    duplicate->erase();
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
materializeLoopedRootsTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    llvm::ArrayRef<mlir::Operation *> roots,
    llvm::ArrayRef<unsigned> outputIndices, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> tileSizes,
    llvm::ArrayRef<int64_t> reductionTileSizes,
    unsigned dim, mlir::ValueRange outputs,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
    llvm::SmallVectorImpl<int64_t> &sizes,
    llvm::SmallVectorImpl<mlir::LoopLikeOpInterface> &loops,
    std::string *failureReason, bool elideSingleIteration = false) {
  if (roots.size() != outputs.size() || roots.size() != outputIndices.size()) {
    setFailureReason(failureReason,
                     "shared traversal root/output count mismatch");
    return mlir::failure();
  }
  if (dim == shape.size()) {
    llvm::SmallVector<mlir::Value, 4> nextOutputs;
    nextOutputs.reserve(roots.size());
    for (auto [index, root, output] : llvm::enumerate(roots, outputs)) {
      mlir::FailureOr<mlir::Value> tile =
          !reductionTileSizes.empty()
              ? materializeCandidatePartialReductionRootTileValue(
                    builder, scope, root, offsets, sizes, reductionTileSizes,
                    loops, failureReason)
              : materializeCandidateRootTileValue(
                    builder, scope, root, outputIndices[index], offsets, sizes,
                    reductionTileSizes, loops, failureReason);
      if (mlir::failed(tile))
        return mlir::failure();
      nextOutputs.push_back(insertCandidateRootTile(
          builder, root->getLoc(), *tile, output, offsets, sizes));
    }
    deduplicateExternalSlicesInBlock(*builder.getInsertionBlock(), scope);
    return nextOutputs;
  }

  mlir::Location loc = roots.front()->getLoc();
  int64_t tailSize = shape[dim] % tileSizes[dim];
  int64_t mainEnd = shape[dim] - tailSize;

  llvm::SmallVector<mlir::Value, 4> currentOutputs(outputs.begin(),
                                                   outputs.end());

  if (elideSingleIteration && tailSize == 0 && tileSizes[dim] == shape[dim]) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(tileSizes[dim]);
    auto next = materializeLoopedRootsTraversal(
        builder, scope, roots, outputIndices, shape, tileSizes,
        reductionTileSizes, dim + 1, currentOutputs, offsets, sizes, loops,
        failureReason, elideSingleIteration);
    sizes.pop_back();
    offsets.pop_back();
    return next;
  }

  if (mainEnd > 0) {
    auto lower = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = builder.create<mlir::arith::ConstantIndexOp>(loc, mainEnd);
    auto step =
        builder.create<mlir::arith::ConstantIndexOp>(loc, tileSizes[dim]);
    auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step,
                                                 currentOutputs);

    offsets.push_back(loop.getInductionVar());
    sizes.push_back(tileSizes[dim]);
    loops.push_back(mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> next =
        materializeLoopedRootsTraversal(
            bodyBuilder, scope, roots, outputIndices, shape, tileSizes,
            reductionTileSizes, dim + 1, loop.getRegionIterArgs(), offsets,
            sizes, loops, failureReason, elideSingleIteration);
    loops.pop_back();
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(next))
      return mlir::failure();

    bodyBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange(*next));
    builder.setInsertionPointAfter(loop);
    currentOutputs.assign(loop.getResults().begin(), loop.getResults().end());
  }

  if (tailSize > 0) {
    offsets.push_back(builder.getIndexAttr(mainEnd));
    sizes.push_back(tailSize);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> tail =
        materializeLoopedRootsTraversal(
            builder, scope, roots, outputIndices, shape, tileSizes,
            reductionTileSizes, dim + 1, currentOutputs, offsets, sizes, loops,
            failureReason, elideSingleIteration);
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(tail))
      return mlir::failure();
    currentOutputs = std::move(*tail);
  }
  return currentOutputs;
}

struct OperandDrivenTraversalSeed {
  mlir::Operation *root = nullptr;
  unsigned operandNumber = 0;
  mlir::RankedTensorType type;
};

static bool isScalarInitializedTargetCompute(mlir::Operation *operation) {
  if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
                mlir::linalg::MatmulTransposeBOp, mlir::linalg::BatchMatmulOp,
                mlir::linalg::BatchMatmulTransposeAOp,
                mlir::linalg::BatchMatmulTransposeBOp>(operation))
    return true;
  auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(operation);
  return generic && !getReductionLoopDims(generic).empty();
}

static mlir::Value getConservativeTraversalDestination(mlir::Operation *root,
                                                       mlir::Value dpsInit) {
  if (!isScalarInitializedTargetCompute(root))
    return dpsInit;
  auto fill = dpsInit.getDefiningOp<mlir::linalg::FillOp>();
  if (!fill || fill.getDpsInits().size() != 1)
    return dpsInit;

  // Tile materialization still consumes the fill result to recover the exact
  // scalar initialization required by target GEMM/reduce.  The complete
  // traversal, however, writes every output tile and must carry the fill's
  // destination object rather than a full-shape value produced by the fill.
  return fill.getDpsInits().front();
}

static mlir::FailureOr<OperandDrivenTraversalSeed>
findOperandDrivenTraversalSeed(TensorProgramScope scope,
                               llvm::ArrayRef<mlir::Operation *> roots,
                               std::string *failureReason) {
  if (roots.size() != 1) {
    setFailureReason(
        failureReason,
        "operand-driven complete traversal requires one yielded consumer");
    return mlir::failure();
  }
  mlir::Operation *root = roots.front();
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(root);
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!tiling || !resultType || !resultType.hasStaticShape()) {
    setFailureReason(
        failureReason,
        "operand-driven complete traversal requires a static tiled consumer");
    return mlir::failure();
  }

  mlir::OpBuilder builder(root);
  for (auto [operandNumber, operand] : llvm::enumerate(root->getOperands())) {
    auto boundary = mlir::dyn_cast<mlir::BlockArgument>(operand);
    auto operandType =
        mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (!boundary || boundary.getOwner() != &scope.getBody() ||
        boundary.getArgNumber() >= scope.getInputCount() || !operandType ||
        !operandType.hasStaticShape() || operandType != resultType)
      continue;

    llvm::SmallVector<mlir::OpFoldResult, 4> offsets(operandType.getRank(),
                                                     builder.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    for (int64_t size : operandType.getShape())
      sizes.push_back(builder.getIndexAttr(size));

    std::string relationFailure;
    mlir::FailureOr<OperandTileIterationDomain> iteration =
        mapOperandTileToIterationDomain(root, builder,
                                        static_cast<unsigned>(operandNumber),
                                        offsets, sizes, &relationFailure);
    if (mlir::failed(iteration))
      continue;
    llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
    llvm::SmallVector<mlir::OpFoldResult> resultSizes;
    if (mlir::failed(tiling.getResultTilePosition(
            builder, /*resultNumber=*/0, iteration->offsets, iteration->sizes,
            resultOffsets, resultSizes)))
      continue;
    if (!llvm::equal(resultOffsets, offsets) ||
        !llvm::equal(resultSizes, sizes))
      continue;
    return OperandDrivenTraversalSeed{
        root, static_cast<unsigned>(operandNumber), operandType};
  }

  setFailureReason(
      failureReason,
      "operand-driven complete traversal found no exact boundary tile seed");
  return mlir::failure();
}

static mlir::FailureOr<mlir::Value> materializeLoopedOperandTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    const OperandDrivenTraversalSeed &seed, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> tileSizes, unsigned dim, mlir::Value output,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
    llvm::SmallVectorImpl<int64_t> &sizes,
    llvm::SmallVectorImpl<mlir::LoopLikeOpInterface> &loops,
    std::string *failureReason) {
  if (dim == shape.size()) {
    llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
    for (int64_t size : sizes)
      mixedSizes.push_back(builder.getIndexAttr(size));
    mlir::FailureOr<mlir::Value> tile =
        materializeCandidateOperandConsumerTileValue(
            builder, scope, seed.root, seed.operandNumber, offsets, mixedSizes,
            loops, failureReason);
    if (mlir::failed(tile))
      return mlir::failure();
    mlir::Value next = insertCandidateRootTile(builder, seed.root->getLoc(),
                                               *tile, output, offsets, sizes);
    deduplicateExternalSlicesInBlock(*builder.getInsertionBlock(), scope);
    return next;
  }

  mlir::Location loc = seed.root->getLoc();
  int64_t tailSize = shape[dim] % tileSizes[dim];
  int64_t mainEnd = shape[dim] - tailSize;
  mlir::Value currentOutput = output;

  if (mainEnd > 0) {
    auto lower = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = builder.create<mlir::arith::ConstantIndexOp>(loc, mainEnd);
    auto step =
        builder.create<mlir::arith::ConstantIndexOp>(loc, tileSizes[dim]);
    auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step,
                                                 currentOutput);
    offsets.push_back(loop.getInductionVar());
    sizes.push_back(tileSizes[dim]);
    loops.push_back(mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
    mlir::FailureOr<mlir::Value> next = materializeLoopedOperandTraversal(
        bodyBuilder, scope, seed, shape, tileSizes, dim + 1,
        loop.getRegionIterArgs().front(), offsets, sizes, loops, failureReason);
    loops.pop_back();
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(next))
      return mlir::failure();
    bodyBuilder.create<mlir::scf::YieldOp>(loc, *next);
    builder.setInsertionPointAfter(loop);
    currentOutput = loop.getResult(0);
  }

  if (tailSize > 0) {
    offsets.push_back(builder.getIndexAttr(mainEnd));
    sizes.push_back(tailSize);
    mlir::FailureOr<mlir::Value> tail = materializeLoopedOperandTraversal(
        builder, scope, seed, shape, tileSizes, dim + 1, currentOutput, offsets,
        sizes, loops, failureReason);
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(tail))
      return mlir::failure();
    currentOutput = *tail;
  }
  return currentOutput;
}

static mlir::LogicalResult materializeCompleteOperandDrivenTraversal(
    TensorProgramScope scope, llvm::ArrayRef<mlir::Operation *> roots,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  if (!candidateReductionTileSizes.empty()) {
    setFailureReason(failureReason,
                     "operand-driven traversal does not invent a reduction "
                     "partition");
    return mlir::failure();
  }
  mlir::FailureOr<OperandDrivenTraversalSeed> seed =
      findOperandDrivenTraversalSeed(scope, roots, failureReason);
  if (mlir::failed(seed))
    return mlir::failure();
  if (mlir::failed(getCandidateOutputTileCount(seed->type, candidateTileSizes,
                                               failureReason)))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> outputBoundary =
      getCandidateOutputBoundary(scope, /*outputIndex=*/0, failureReason);
  if (mlir::failed(outputBoundary))
    return mlir::failure();
  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  mlir::OpBuilder builder(seed->root);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
  mlir::FailureOr<mlir::Value> completeOutput =
      materializeLoopedOperandTraversal(
          builder, scope, *seed, seed->type.getShape(), candidateTileSizes,
          /*dim=*/0, *outputBoundary, offsets, sizes, loops, failureReason);
  if (mlir::failed(completeOutput))
    return mlir::failure();
  returnOp->setOperand(0, *completeOutput);
  seed->root->erase();
  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

mlir::LogicalResult materializeCompleteCandidateTraversal(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    CandidateTileTraversalKind traversalKind, std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> roots =
      collectCandidateRoots(scope, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();
  if (traversalKind == CandidateTileTraversalKind::OperandDriven)
    return materializeCompleteOperandDrivenTraversal(
        scope, *roots, candidateTileSizes, candidateReductionTileSizes,
        failureReason);

  struct RootTraversalGroup {
    llvm::SmallVector<int64_t, 4> shape;
    llvm::SmallVector<mlir::Operation *, 4> roots;
    llvm::SmallVector<unsigned, 4> outputIndices;
    llvm::SmallVector<mlir::Value, 4> outputBoundaries;
  };
  llvm::SmallVector<RootTraversalGroup, 4> groups;

  bool hasReductionRoot = false;
  for (auto [outputIndex, root] : llvm::enumerate(*roots)) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape()) {
      setFailureReason(failureReason,
                       "complete candidate traversal result is not a static "
                       "ranked tensor");
      return mlir::failure();
    }
    // Compact loops avoid eager instance expansion, but every traversal
    // domain must still be representable before subview construction.
    if (mlir::failed(getCandidateOutputTileCount(resultType, candidateTileSizes,
                                                 failureReason)))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(outputIndex), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();
    auto group = llvm::find_if(groups, [&](const RootTraversalGroup &current) {
      return llvm::equal(current.shape, resultType.getShape());
    });
    if (group == groups.end()) {
      RootTraversalGroup next;
      next.shape.assign(resultType.getShape().begin(),
                        resultType.getShape().end());
      groups.push_back(std::move(next));
      group = std::prev(groups.end());
    }
    group->roots.push_back(root);
    group->outputIndices.push_back(static_cast<unsigned>(outputIndex));
    group->outputBoundaries.push_back(*outputBoundary);

    if (!candidateReductionTileSizes.empty()) {
      mlir::FailureOr<mlir::linalg::LinalgOp> computeRoot =
          getCandidatePartialReductionComputeRoot(root, failureReason);
      if (mlir::failed(computeRoot))
        return mlir::failure();
      hasReductionRoot |= !getReductionLoopDims(*computeRoot).empty();
      if (!mlir::isa<mlir::PartialReductionOpInterface>(
              computeRoot->getOperation()) ||
          mlir::failed(verifyCandidateReductionSplitNumericLegality(
              *computeRoot, failureReason)))
        return mlir::failure();
      continue;
    }
    if (auto linalgRoot = mlir::dyn_cast<mlir::linalg::LinalgOp>(root))
      hasReductionRoot |= !getReductionLoopDims(linalgRoot).empty();
  }
  if (!candidateReductionTileSizes.empty() && !hasReductionRoot) {
    setFailureReason(failureReason,
                     "candidate reduction split requires a reduction root");
    return mlir::failure();
  }
  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  // Equal-shape roots share one traversal. Different result domains receive
  // independent traversals in the same actual residency program; recursively
  // tiled shared producers may therefore form bounded consumer-compatible
  // versions instead of forcing a full-shape spill or rejecting multi-root
  // programs outright.
  for (RootTraversalGroup &group : groups) {
    mlir::OpBuilder builder(group.roots.front());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<int64_t, 4> sizes;
    llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> completeOutputs =
        materializeLoopedRootsTraversal(
            builder, scope, group.roots, group.outputIndices, group.shape,
            candidateTileSizes, candidateReductionTileSizes, /*dim=*/0,
            group.outputBoundaries, offsets, sizes, loops, failureReason);
    if (mlir::failed(completeOutputs))
      return mlir::failure();
    for (auto [index, output] :
         llvm::zip(group.outputIndices, *completeOutputs))
      returnOp->setOperand(index, output);
  }
  for (mlir::Operation *root : *roots)
    root->erase();

  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

static bool isDDRConnectionAction(CandidateTraversalConnectionAction action) {
  return action == CandidateTraversalConnectionAction::SeparatedDDR ||
         action == CandidateTraversalConnectionAction::CrossRegion;
}

static bool
isSeparatedConnectionAction(CandidateTraversalConnectionAction action) {
  return action != CandidateTraversalConnectionAction::CoupledResident;
}

static mlir::FailureOr<mlir::Value>
createJointTraversalDestination(TensorProgramScope scope, mlir::Operation *root,
                                CandidateTraversalConnectionAction action,
                                std::string *failureReason) {
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(root);
  auto resultType =
      root && root->getNumResults() == 1
          ? mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType())
          : mlir::RankedTensorType{};
  if (!dps || dps.getNumDpsInits() != 1 || !resultType ||
      !resultType.hasStaticShape()) {
    setFailureReason(
        failureReason,
        "joint traversal root requires one static ranked DPS result");
    return mlir::failure();
  }
  if (!isDDRConnectionAction(action))
    return getConservativeTraversalDestination(root, dps.getDpsInits().front());

  auto ddrType = mlir::MemRefType::get(
      resultType.getShape(), resultType.getElementType(),
      mlir::MemRefLayoutAttrInterface{},
      wafer::MemoryAttr::get(scope.getContext(), wafer::MemorySpace::DDR,
                             wafer::MemLayout::Tensor));
  mlir::OpBuilder builder(root);
  auto allocation =
      builder.create<mlir::memref::AllocOp>(root->getLoc(), ddrType);
  auto tensor = builder.create<mlir::bufferization::ToTensorOp>(
      root->getLoc(), allocation.getResult(), /*restrict=*/true,
      /*writable=*/true);
  return tensor.getResult();
}

mlir::LogicalResult materializeJointCompleteRankTraversals(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<CandidateTraversalConnection> connections,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> choices,
    std::string *failureReason) {
  if (connections.empty() || connections.size() != choices.size()) {
    setFailureReason(failureReason,
                     "joint traversal requires one action per connection");
    return mlir::failure();
  }

  CandidateConnectionFusionPolicy policy(connections, choices);
  if (!policy.isValid())
    return mlir::failure();
  for (auto [connection, choice] : llvm::zip_equal(connections, choices)) {
    mlir::Operation *producer = connection.getProducer();
    mlir::Operation *consumer = connection.getConsumer();
    auto producerType = connection.producerResult
                            ? mlir::dyn_cast<mlir::RankedTensorType>(
                                  connection.producerResult.getType())
                            : mlir::RankedTensorType{};
    auto consumerOperandType =
        connection.consumerOperand
            ? mlir::dyn_cast<mlir::RankedTensorType>(
                  connection.consumerOperand->get().getType())
            : mlir::RankedTensorType{};
    auto consumerType =
        consumer && consumer->getNumResults() == 1
            ? mlir::dyn_cast<mlir::RankedTensorType>(
                  consumer->getResult(0).getType())
            : mlir::RankedTensorType{};
    if (!connection.isValid() || !producer || !consumer || !producerType ||
        !consumerOperandType || !consumerType ||
        !producerType.hasStaticShape() ||
        !consumerOperandType.hasStaticShape() ||
        !consumerType.hasStaticShape()) {
      setFailureReason(
          failureReason,
          "joint traversal choices require exact static connection domains");
      return mlir::failure();
    }
    if (choice.action ==
            CandidateTraversalConnectionAction::CoupledResident &&
        !choice.producerTileSizes.empty()) {
      setFailureReason(
          failureReason,
          "coupled connection derives producer demand from the consumer "
          "tile and cannot select a producer tile independently");
      return mlir::failure();
    }
    if ((!choice.producerTileSizes.empty() &&
         mlir::failed(getCandidateOutputTileCount(
             producerType, choice.producerTileSizes, failureReason))) ||
        (!choice.consumerTileSizes.empty() &&
         mlir::failed(getCandidateOutputTileCount(
             consumerType, choice.consumerTileSizes, failureReason))))
      return mlir::failure();
  }
  TensorProgramScope policyScope(scope.getFunction(), &policy);
  mlir::Block &body = policyScope.getBody();
  auto returnOp = policyScope.getReturn();

  llvm::SmallVector<mlir::Operation *, 32> nodes;
  llvm::DenseMap<mlir::Operation *, unsigned> nodeIndices;
  for (mlir::Operation &operation : body.without_terminator()) {
    if (classifyCandidateTraversalRoot(&operation) ==
        CandidateTraversalRootCapability::Unsupported)
      continue;
    nodeIndices[&operation] = nodes.size();
    nodes.push_back(&operation);
  }
  if (nodes.empty()) {
    setFailureReason(failureReason,
                     "joint traversal found no structured operations");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 32> parents(nodes.size());
  for (unsigned index = 0; index < parents.size(); ++index)
    parents[index] = index;
  auto findRoot = [&](unsigned index) {
    while (parents[index] != index) {
      parents[index] = parents[parents[index]];
      index = parents[index];
    }
    return index;
  };
  auto unite = [&](unsigned left, unsigned right) {
    unsigned leftRoot = findRoot(left);
    unsigned rightRoot = findRoot(right);
    if (leftRoot != rightRoot)
      parents[std::max(leftRoot, rightRoot)] = std::min(leftRoot, rightRoot);
  };
  for (auto [connection, choice] : llvm::zip_equal(connections, choices)) {
    auto producer = nodeIndices.find(connection.getProducer());
    auto consumer = nodeIndices.find(connection.getConsumer());
    if (producer == nodeIndices.end() || consumer == nodeIndices.end()) {
      setFailureReason(failureReason,
                       "joint traversal connection left the current IR");
      return mlir::failure();
    }
    if (choice.action == CandidateTraversalConnectionAction::CoupledResident)
      unite(producer->second, consumer->second);
  }

  struct RootEntry {
    mlir::Operation *operation = nullptr;
    unsigned component = 0;
    unsigned blockOrdinal = 0;
    std::optional<unsigned> outputIndex;
    mlir::Value returnedValue;
    CandidateTraversalConnectionAction boundaryAction =
        CandidateTraversalConnectionAction::SeparatedResident;
    llvm::SmallVector<int64_t, 4> tileSizes;
  };
  llvm::SmallVector<RootEntry, 16> roots;
  llvm::DenseSet<mlir::Operation *> seenRoots;
  auto appendRoot = [&](mlir::Operation *operation,
                        std::optional<unsigned> outputIndex,
                        mlir::Value returnedValue,
                        CandidateTraversalConnectionAction boundaryAction,
                        llvm::ArrayRef<int64_t> tileSizes) {
    if (!operation || !seenRoots.insert(operation).second)
      return;
    auto node = nodeIndices.find(operation);
    if (node == nodeIndices.end())
      return;
    // A materialized internal boundary is its own traversal root even when a
    // second fanout branch couples the same producer into an observable root.
    // Grouping those roots would recreate the forbidden hybrid traversal and
    // can also make the original producer fail SSA dominance.
    unsigned component =
        outputIndex ? findRoot(node->second) : nodes.size() + node->second;
    RootEntry entry{operation,   component,     node->second,
                    outputIndex, returnedValue, boundaryAction};
    entry.tileSizes.assign(tileSizes.begin(), tileSizes.end());
    roots.push_back(std::move(entry));
  };

  for (auto [outputIndex, returned] : llvm::enumerate(returnOp.getOperands())) {
    mlir::OpResult rootResult =
        traceCandidateStructuredProducer(returned, body);
    mlir::Operation *root = rootResult ? rootResult.getOwner() : nullptr;
    llvm::SmallVector<int64_t, 4> rootTileSizes;
    for (auto [connection, choice] : llvm::zip_equal(connections, choices)) {
      if (connection.getConsumer() != root ||
          choice.consumerTileSizes.empty())
        continue;
      if (!rootTileSizes.empty() &&
          !llvm::equal(rootTileSizes, choice.consumerTileSizes)) {
        setFailureReason(
            failureReason,
            "one coupled consumer requires incompatible tile vectors");
        return mlir::failure();
      }
      rootTileSizes.assign(choice.consumerTileSizes.begin(),
                           choice.consumerTileSizes.end());
    }
    if (rootTileSizes.empty())
      rootTileSizes.assign(candidateTileSizes.begin(),
                           candidateTileSizes.end());
    appendRoot(root, static_cast<unsigned>(outputIndex), returned,
               CandidateTraversalConnectionAction::SeparatedResident,
               rootTileSizes);
  }

  for (auto [connectionIndex, connection] : llvm::enumerate(connections)) {
    const CandidateTraversalConnectionChoice &choice = choices[connectionIndex];
    CandidateTraversalConnectionAction action = choice.action;
    if (!isSeparatedConnectionAction(action))
      continue;
    CandidateTraversalConnectionAction rootAction = action;
    llvm::SmallVector<int64_t, 4> producerTileSizes(
        choice.producerTileSizes.begin(), choice.producerTileSizes.end());
    for (auto [otherIndex, other] : llvm::enumerate(connections)) {
      if (otherIndex == connectionIndex ||
          other.producerResult != connection.producerResult ||
          !isSeparatedConnectionAction(choices[otherIndex].action))
        continue;
      bool leftDDR = isDDRConnectionAction(rootAction);
      bool rightDDR = isDDRConnectionAction(choices[otherIndex].action);
      bool leftSpill =
          rootAction == CandidateTraversalConnectionAction::SelectiveSpill;
      bool rightSpill = choices[otherIndex].action ==
                        CandidateTraversalConnectionAction::SelectiveSpill;
      bool incompatibleTileVersion =
          !producerTileSizes.empty() &&
          !choices[otherIndex].producerTileSizes.empty() &&
          !llvm::equal(producerTileSizes,
                       choices[otherIndex].producerTileSizes);
      if (leftDDR != rightDDR || leftSpill != rightSpill ||
          incompatibleTileVersion) {
        setFailureReason(
            failureReason,
            "one producer fanout requires incompatible shared versions");
        return mlir::failure();
      }
      if (producerTileSizes.empty())
        producerTileSizes.assign(choices[otherIndex].producerTileSizes.begin(),
                                 choices[otherIndex].producerTileSizes.end());
      if (choices[otherIndex].action ==
          CandidateTraversalConnectionAction::CrossRegion)
        rootAction = CandidateTraversalConnectionAction::CrossRegion;
    }
    if (producerTileSizes.empty())
      producerTileSizes.assign(candidateTileSizes.begin(),
                               candidateTileSizes.end());
    appendRoot(connection.getProducer(), std::nullopt, mlir::Value{},
               rootAction, producerTileSizes);
  }
  if (roots.empty()) {
    setFailureReason(failureReason,
                     "joint traversal has no observable or cut roots");
    return mlir::failure();
  }

  llvm::sort(roots, [](const RootEntry &left, const RootEntry &right) {
    return left.blockOrdinal > right.blockOrdinal;
  });
  struct RootTraversalGroup {
    unsigned component = 0;
    llvm::SmallVector<int64_t, 4> shape;
    llvm::SmallVector<int64_t, 4> tileSizes;
    llvm::SmallVector<RootEntry *, 4> roots;
  };
  llvm::SmallVector<RootTraversalGroup, 16> groups;
  for (RootEntry &root : roots) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        root.operation->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape()) {
      setFailureReason(failureReason,
                       "joint traversal root has no static result domain");
      return mlir::failure();
    }
    llvm::SmallVector<int64_t, 4> rootTileSizes = root.tileSizes;
    if (rootTileSizes.empty())
      rootTileSizes.assign(resultType.getShape().begin(),
                           resultType.getShape().end());
    auto group = llvm::find_if(groups, [&](const RootTraversalGroup &current) {
      return current.component == root.component &&
             llvm::equal(current.shape, resultType.getShape()) &&
             llvm::equal(current.tileSizes, rootTileSizes);
    });
    if (group == groups.end()) {
      RootTraversalGroup next;
      next.component = root.component;
      next.shape.assign(resultType.getShape().begin(),
                        resultType.getShape().end());
      next.tileSizes = std::move(rootTileSizes);
      groups.push_back(std::move(next));
      group = std::prev(groups.end());
    }
    group->roots.push_back(&root);
  }

  for (RootTraversalGroup &group : groups) {
    llvm::SmallVector<mlir::Operation *, 4> groupRoots;
    llvm::SmallVector<unsigned, 4> outputIndices;
    llvm::SmallVector<mlir::Value, 4> destinations;
    llvm::SmallVector<mlir::Value, 4> originalResults;
    for (RootEntry *root : group.roots) {
      if (!root->operation || root->operation->getBlock() != &body) {
        setFailureReason(failureReason,
                         "joint traversal root was consumed out of order");
        return mlir::failure();
      }
      groupRoots.push_back(root->operation);
      outputIndices.push_back(root->outputIndex.value_or(0));
      originalResults.push_back(root->operation->getResult(0));
      if (root->outputIndex) {
        mlir::FailureOr<mlir::Value> boundary = getCandidateOutputBoundary(
            policyScope, *root->outputIndex, failureReason);
        if (mlir::failed(boundary))
          return mlir::failure();
        mlir::OpBuilder destinationBuilder(root->operation);
        mlir::FailureOr<mlir::Value> mappedDestination =
            materializeJointOutputDestination(
                destinationBuilder, root->returnedValue,
                root->operation->getResult(0), *boundary, failureReason);
        if (mlir::failed(mappedDestination))
          return mlir::failure();
        destinations.push_back(*mappedDestination);
      } else {
        mlir::FailureOr<mlir::Value> destination =
            createJointTraversalDestination(policyScope, root->operation,
                                            root->boundaryAction,
                                            failureReason);
        if (mlir::failed(destination))
          return mlir::failure();
        destinations.push_back(*destination);
      }
    }

    llvm::ArrayRef<int64_t> groupTileSizes(group.tileSizes);
    if (mlir::failed(getCandidateOutputTileCount(
            mlir::cast<mlir::RankedTensorType>(
                groupRoots.front()->getResult(0).getType()),
            groupTileSizes, failureReason)))
      return mlir::failure();
    mlir::OpBuilder builder(groupRoots.front());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<int64_t, 4> sizes;
    llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> completeOutputs =
        materializeLoopedRootsTraversal(
            builder, policyScope, groupRoots, outputIndices, group.shape,
            groupTileSizes, /*reductionTileSizes=*/{}, /*dim=*/0,
            destinations, offsets, sizes, loops, failureReason,
            /*elideSingleIteration=*/true);
    if (mlir::failed(completeOutputs) ||
        completeOutputs->size() != groupRoots.size())
      return mlir::failure();
    for (auto [root, originalResult, completeOutput] :
         llvm::zip_equal(groupRoots, originalResults, *completeOutputs)) {
      originalResult.replaceAllUsesWith(completeOutput);
      root->erase();
    }
    eraseDeadCandidateSupportClosure(policyScope);
  }

  if (mlir::failed(mlir::verify(policyScope.getFunction()))) {
    setFailureReason(failureReason,
                     "joint traversal produced invalid structured IR");
    return mlir::failure();
  }
  return mlir::success();
}

static bool
isConservativeCompleteRankTraversalRoot(mlir::Operation &operation) {
  if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(operation)) {
    bool isReturned = llvm::any_of(fill->getUsers(), [](mlir::Operation *user) {
      return mlir::isa<mlir::func::ReturnOp>(user);
    });
    // Keep a non-terminal fill attached to its consumer so reduction and
    // overwrite lowering can prove the exact initialization scalar while
    // materializing that consumer's tile.
    if (!isReturned)
      return false;
  }
  return classifyCandidateTraversalRoot(&operation) !=
         CandidateTraversalRootCapability::Unsupported;
}

static mlir::LogicalResult materializeSeparatedCompleteRankTraversalsImpl(
    TensorProgramScope scope,
    std::optional<llvm::ArrayRef<int64_t>> candidateTileSizes,
    llvm::ArrayRef<mlir::Operation *> coveredTopLevelOperations,
    std::string *failureReason) {
  llvm::DenseSet<mlir::Operation *> covered(
      coveredTopLevelOperations.begin(), coveredTopLevelOperations.end());
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (mlir::Operation &operation : scope.getBody().without_terminator()) {
    if (covered.contains(&operation))
      continue;
    if (!isConservativeCompleteRankTraversalRoot(operation))
      continue;
    if (operation.getNumResults() != 1) {
      setFailureReason(
          failureReason,
          "conservative complete-rank baseline requires single-result "
          "structured roots");
      return mlir::failure();
    }
    roots.push_back(&operation);
  }
  if (roots.empty()) {
    setFailureReason(failureReason,
                     "conservative complete-rank baseline found no "
                     "structured roots");
    return mlir::failure();
  }

  // C1's mandatory baseline is deliberately spill-conservative: every
  // original tensor.empty that is not the public function output becomes an
  // explicit compiler-owned DDR tensor. Each structured root then traverses
  // into that destination before the next root is tiled. This keeps one
  // complete-rank decision clone and one residency region, but prevents an
  // unsupported producer-view relation from silently materializing a
  // full-shape SPM producer.
  llvm::SmallVector<mlir::tensor::EmptyOp, 16> emptyDestinations;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    if (!covered.contains(&operation))
      if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(operation))
      emptyDestinations.push_back(empty);
  for (mlir::tensor::EmptyOp empty : emptyDestinations) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(empty.getType());
    if (!tensorType || !tensorType.hasStaticShape()) {
      setFailureReason(
          failureReason,
          "conservative complete-rank spill requires static tensor.empty "
          "destinations");
      return mlir::failure();
    }
    auto ddrType = mlir::MemRefType::get(
        tensorType.getShape(), tensorType.getElementType(),
        mlir::MemRefLayoutAttrInterface{},
        wafer::MemoryAttr::get(scope.getContext(), wafer::MemorySpace::DDR,
                               wafer::MemLayout::Tensor));
    mlir::OpBuilder builder(empty);
    auto allocation =
        builder.create<mlir::memref::AllocOp>(empty.getLoc(), ddrType);
    auto tensor = builder.create<mlir::bufferization::ToTensorOp>(
        empty.getLoc(), allocation.getResult(), /*restrict=*/true,
        /*writable=*/true);
    empty.getResult().replaceAllUsesWith(tensor.getResult());
    empty.erase();
  }

  for (mlir::Operation *root : roots) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        llvm::any_of(resultType.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      setFailureReason(failureReason,
                       "conservative complete-rank traversal requires "
                       "positive static result shapes");
      return mlir::failure();
    }
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(root);
    if (!dps || dps.getNumDpsInits() != 1) {
      setFailureReason(
          failureReason,
          "conservative complete-rank baseline requires one DPS destination");
      return mlir::failure();
    }

    // A typed collective that is explicitly classified as full-traversal-only
    // remains one complete tensor operation in this conservative clone. Its
    // DPS destination is already an explicit DDR boundary/allocation, and the
    // Tile body emitter materializes the matching typed communication op. Do
    // not invent a tile relation for an operation whose interface does not
    // provide one; coordinated collective tiling is a separate candidate
    // action rather than a baseline legality requirement.
    if (classifyCandidateTraversalRoot(root) ==
        CandidateTraversalRootCapability::FullTraversalOnly)
      continue;

    llvm::SmallVector<int64_t, 4> shape(resultType.getShape().begin(),
                                        resultType.getShape().end());
    llvm::SmallVector<int64_t, 4> tileSizes;
    if (candidateTileSizes)
      tileSizes.assign(candidateTileSizes->begin(), candidateTileSizes->end());
    else
      tileSizes.assign(shape.size(), 1);
    if (mlir::failed(
            getCandidateOutputTileCount(resultType, tileSizes, failureReason)))
      return mlir::failure();

    unsigned outputIndex = 0;
    if (auto argument =
            mlir::dyn_cast<mlir::BlockArgument>(dps.getDpsInits().front());
        argument && argument.getOwner() == &scope.getBody() &&
        argument.getArgNumber() >= scope.getInputCount())
      outputIndex = argument.getArgNumber() - scope.getInputCount();

    mlir::Value originalResult = root->getResult(0);
    mlir::Value outputDestination =
        getConservativeTraversalDestination(root, dps.getDpsInits().front());
    mlir::OpBuilder builder(root);
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<int64_t, 4> sizes;
    llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> completeOutputs =
        materializeLoopedRootsTraversal(
            builder, scope, llvm::ArrayRef<mlir::Operation *>{root},
            llvm::ArrayRef<unsigned>{outputIndex}, shape, tileSizes,
            /*reductionTileSizes=*/{}, /*dim=*/0,
            mlir::ValueRange{outputDestination}, offsets, sizes, loops,
            failureReason);
    if (mlir::failed(completeOutputs))
      return mlir::failure();
    if (completeOutputs->size() != 1) {
      setFailureReason(
          failureReason,
          "conservative complete-rank traversal produced invalid arity");
      return mlir::failure();
    }
    originalResult.replaceAllUsesWith(completeOutputs->front());
    root->erase();
    eraseDeadCandidateSupportClosure(scope);
  }

  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

mlir::LogicalResult
materializeConservativeCompleteRankTraversals(TensorProgramScope scope,
                                              std::string *failureReason) {
  return materializeSeparatedCompleteRankTraversalsImpl(
      scope, std::nullopt, /*coveredTopLevelOperations=*/{}, failureReason);
}

mlir::LogicalResult materializeRemainingConservativeCompleteRankTraversals(
    TensorProgramScope scope,
    llvm::ArrayRef<mlir::Operation *> coveredTopLevelOperations,
    std::string *failureReason) {
  llvm::DenseSet<mlir::Operation *> covered(
      coveredTopLevelOperations.begin(), coveredTopLevelOperations.end());
  bool hasRemainingRoot = llvm::any_of(
      scope.getBody().without_terminator(), [&](mlir::Operation &operation) {
        return !covered.contains(&operation) &&
               isConservativeCompleteRankTraversalRoot(operation);
      });
  if (!hasRemainingRoot)
    return mlir::success();
  return materializeSeparatedCompleteRankTraversalsImpl(
      scope, std::nullopt, coveredTopLevelOperations, failureReason);
}

mlir::LogicalResult materializeSeparatedCompleteRankTraversals(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileSizes,
    std::string *failureReason) {
  return materializeSeparatedCompleteRankTraversalsImpl(
      scope, candidateTileSizes, /*coveredTopLevelOperations=*/{},
      failureReason);
}

} // namespace wafer::tensor_program_to_tile_region
