//===- TemporalTiling.cpp - Apply live-operation temporal choices -----===//

#include "TemporalTiling.h"

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer {
namespace {

template <typename T>
mlir::FailureOr<T> fail(TemporalTilingFailure *failure,
                        TemporalTilingFailureKind kind,
                        llvm::StringRef detail) {
  if (failure) {
    failure->kind = kind;
    failure->detail = detail.str();
  }
  return mlir::failure();
}

mlir::FailureOr<llvm::DenseSet<mlir::Value>>
collectExactFusionEdges(TileRegionOp region, TemporalTilingFailure *failure) {
  llvm::DenseSet<mlir::Value> exact;
  for (mlir::Operation &operation :
       region.getBody().front().without_terminator()) {
    for (mlir::OpResult result : operation.getResults()) {
      for (mlir::OpOperand &use : result.getUses()) {
        compiler::detail::TemporalFusionQueryResult query =
            compiler::detail::queryTemporalProducerFusion(result, use);
        if (query.kind ==
            compiler::detail::TemporalFusionQueryKind::BrokenContract)
          return fail<llvm::DenseSet<mlir::Value>>(
              failure, TemporalTilingFailureKind::BrokenContract, query.detail);
        if (query.kind ==
            compiler::detail::TemporalFusionQueryKind::ExactDerived)
          exact.insert(result);
      }
    }
  }
  return exact;
}

llvm::SmallVector<int64_t, 6>
buildInterchange(const compiler::detail::TemporalScopeDescriptor &descriptor,
                 const compiler::detail::TemporalScopeChoice &choice) {
  llvm::SmallVector<int64_t, 6> interchange;
  interchange.reserve(descriptor.iterationExtents.size());
  for (uint32_t dimension : choice.loopOrder)
    interchange.push_back(dimension);
  for (unsigned dimension = 0; dimension < descriptor.iterationExtents.size();
       ++dimension)
    if (!llvm::is_contained(choice.loopOrder, dimension))
      interchange.push_back(dimension);
  return interchange;
}

mlir::LogicalResult
eraseFusedProducers(mlir::IRRewriter &rewriter,
                    llvm::ArrayRef<mlir::Operation *> fusedProducers) {
  for (mlir::Operation *producer : fusedProducers) {
    bool erasedDeadUser = true;
    while (producer && producer->getBlock() && !producer->use_empty() &&
           erasedDeadUser) {
      erasedDeadUser = false;
      llvm::SmallVector<mlir::Operation *, 4> users;
      for (mlir::Operation *user : producer->getUsers())
        if (!llvm::is_contained(users, user))
          users.push_back(user);
      for (mlir::Operation *user : users)
        if (user->getBlock() && mlir::isOpTriviallyDead(user)) {
          rewriter.eraseOp(user);
          erasedDeadUser = true;
        }
    }
    if (!producer || !producer->getBlock() || !producer->use_empty())
      return mlir::failure();
    rewriter.eraseOp(producer);
  }
  return mlir::success();
}

mlir::LogicalResult specializeRaggedTails(
    mlir::IRRewriter &rewriter,
    const compiler::detail::TemporalScopeDescriptor &descriptor,
    const compiler::detail::TemporalScopeChoice &choice,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    TemporalTilingStatistics &statistics) {
  if (loops.size() != choice.loopOrder.size())
    return mlir::failure();
  for (size_t reverse = 0; reverse < loops.size(); ++reverse) {
    const size_t position = loops.size() - reverse - 1;
    const unsigned dimension = choice.loopOrder[position];
    const int64_t extent = descriptor.iterationExtents[dimension];
    const int64_t tileSize = choice.iteratorTileSizes[dimension];
    if (extent % tileSize == 0)
      continue;
    auto loop =
        mlir::dyn_cast<mlir::scf::ForOp>(loops[position].getOperation());
    if (!loop)
      return mlir::failure();
    mlir::scf::ForOp partialIteration;
    if (mlir::failed(mlir::scf::peelForLoopAndSimplifyBounds(
            rewriter, loop, partialIteration)) ||
        !partialIteration ||
        mlir::failed(partialIteration.promoteIfSingleIteration(rewriter)))
      return mlir::failure();
    ++statistics.specializedTails;
  }
  return mlir::success();
}

void eraseDeadOperations(mlir::IRRewriter &rewriter, TileRegionOp region) {
  llvm::SmallVector<mlir::Operation *, 32> operations;
  region.walk<mlir::WalkOrder::PostOrder>([&](mlir::Operation *operation) {
    if (operation != region.getOperation() &&
        !operation->hasTrait<mlir::OpTrait::IsTerminator>())
      operations.push_back(operation);
  });
  for (mlir::Operation *operation : operations)
    if (operation->getBlock() && mlir::isOpTriviallyDead(operation))
      rewriter.eraseOp(operation);
}

mlir::LogicalResult
canonicalizeTiledRegion(TileRegionOp region,
                        mlir::RewriterBase::Listener *listener) {
  mlir::RewritePatternSet patterns =
      mlir::linalg::getLinalgTilingCanonicalizationPatterns(
          region.getContext());
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.maxIterations = 10;
  config.scope = &region.getBody();
  config.listener = listener;
  return mlir::applyPatternsAndFoldGreedily(
      region.getBody(), mlir::FrozenRewritePatternSet(std::move(patterns)),
      config);
}

mlir::LogicalResult refineOnlineAttentionStaticTypes(mlir::IRRewriter &rewriter,
                                                     TileRegionOp region) {
  llvm::SmallVector<LinalgExtOnlineAttentionOp, 8> operations;
  region.walk([&](LinalgExtOnlineAttentionOp operation) {
    operations.push_back(operation);
  });
  for (LinalgExtOnlineAttentionOp operation : operations) {
    llvm::SmallVector<mlir::Value, 8> operands(operation->getOperands());
    bool changed = false;
    for (mlir::Value &operand : operands) {
      auto cast = operand.getDefiningOp<mlir::tensor::CastOp>();
      auto sourceType = cast ? mlir::dyn_cast<mlir::RankedTensorType>(
                                   cast.getSource().getType())
                             : mlir::RankedTensorType{};
      auto resultType =
          cast ? mlir::dyn_cast<mlir::RankedTensorType>(cast.getType())
               : mlir::RankedTensorType{};
      if (!cast || !sourceType || !resultType || !sourceType.hasStaticShape() ||
          resultType.hasStaticShape())
        continue;
      operand = cast.getSource();
      changed = true;
    }
    if (!changed)
      continue;
    const unsigned initStart = operation.getMask() ? 5 : 4;
    if (operands.size() != initStart + 3)
      return mlir::failure();
    llvm::SmallVector<mlir::Type, 3> resultTypes{
        operands[initStart].getType(), operands[initStart + 1].getType(),
        operands[initStart + 2].getType()};
    rewriter.setInsertionPoint(operation);
    mlir::Operation *replacement =
        mlir::clone(rewriter, operation.getOperation(), resultTypes, operands);
    rewriter.replaceOp(operation, replacement->getResults());
  }
  return mlir::success();
}

mlir::LogicalResult refineLinalgStaticTypes(mlir::IRRewriter &rewriter,
                                            TileRegionOp region) {
  llvm::SmallVector<mlir::linalg::LinalgOp, 16> operations;
  region.walk([&](mlir::linalg::LinalgOp operation) {
    operations.push_back(operation);
  });
  for (mlir::linalg::LinalgOp operation : operations) {
    if (!operation.hasPureTensorSemantics())
      continue;
    llvm::SmallVector<mlir::Value, 8> operands(operation->getOperands());
    bool changed = false;
    for (mlir::Value &operand : operands) {
      auto cast = operand.getDefiningOp<mlir::tensor::CastOp>();
      auto sourceType = cast ? mlir::dyn_cast<mlir::RankedTensorType>(
                                   cast.getSource().getType())
                             : mlir::RankedTensorType{};
      auto resultType =
          cast ? mlir::dyn_cast<mlir::RankedTensorType>(cast.getType())
               : mlir::RankedTensorType{};
      if (!cast || !sourceType || !resultType || !sourceType.hasStaticShape() ||
          resultType.hasStaticShape())
        continue;
      operand = cast.getSource();
      changed = true;
    }
    if (!changed)
      continue;
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    for (unsigned index = 0; index < operation.getNumDpsInits(); ++index) {
      mlir::OpOperand *init = operation.getDpsInitOperand(index);
      resultTypes.push_back(operands[init->getOperandNumber()].getType());
    }
    rewriter.setInsertionPoint(operation);
    mlir::Operation *replacement =
        mlir::clone(rewriter, operation.getOperation(), resultTypes, operands);
    rewriter.replaceOp(operation, replacement->getResults());
  }
  return mlir::success();
}

} // namespace

mlir::FailureOr<TemporalTilingStatistics>
applyTemporalTiling(const compiler::detail::TemporalDomain &domain,
                    const compiler::detail::TemporalChoice &choice,
                    StructuredMaterializationRelations &relations,
                    TemporalTilingFailure *failure) {
  if (failure)
    *failure = {};
  TileRegionOp region = domain.getRegion();
  if (!region || !domain.contains(choice) || mlir::failed(mlir::verify(region)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::BrokenContract,
        "temporal apply requires one live choice from an unchanged TileRegion");
  if (mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          region->getParentOfType<mlir::ModuleOp>(), relations)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::BrokenContract,
        "temporal apply received stale structural endpoint relations");

  llvm::DenseMap<mlir::Operation *,
                 const compiler::detail::TemporalScopeDescriptor *>
      descriptors;
  for (const auto &descriptor : domain.getScopeDescriptors())
    descriptors.try_emplace(descriptor.operation, &descriptor);
  for (const auto &scope : choice.scopes) {
    auto descriptor = descriptors.find(scope.operation);
    if (!scope.operation || descriptor == descriptors.end() ||
        scope.operation->getParentOfType<TileRegionOp>() != region ||
        scope.operation->getBlock() != &region.getBody().front())
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::BrokenContract,
          "temporal choice contains a stale traversal operation");
  }

  bool hasAnyActiveDimension = false;
  for (const auto &scope : choice.scopes) {
    const auto &descriptor = *descriptors.find(scope.operation)->second;
    hasAnyActiveDimension |= llvm::any_of(
        llvm::zip_equal(descriptor.iterationExtents, scope.iteratorTileSizes),
        [](auto values) { return std::get<1>(values) < std::get<0>(values); });
  }
  if (!hasAnyActiveDimension)
    return TemporalTilingStatistics{};

  compiler::detail::StructuredBufferReplacementListener listener(relations);
  mlir::IRRewriter rewriter(region.getContext(), &listener);
  TemporalTilingStatistics statistics;
  for (const auto &scope : llvm::reverse(choice.scopes)) {
    const auto &descriptor = *descriptors.find(scope.operation)->second;
    llvm::SmallVector<mlir::OpFoldResult, 6> tileSizes;
    bool hasActiveDimension = false;
    for (auto [extent, size] : llvm::zip_equal(descriptor.iterationExtents,
                                               scope.iteratorTileSizes)) {
      const bool active = size < extent;
      hasActiveDimension |= active;
      tileSizes.push_back(rewriter.getIndexAttr(active ? size : 0));
    }
    if (!hasActiveDimension)
      continue;

    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(scope.operation);
    if (!tiling)
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::BrokenContract,
          "temporal traversal lost TilingInterface before apply");
    mlir::FailureOr<llvm::DenseSet<mlir::Value>> exactFusionEdges =
        collectExactFusionEdges(region, failure);
    if (mlir::failed(exactFusionEdges))
      return mlir::failure();
    mlir::scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setInterchange(buildInterchange(descriptor, scope));
    mlir::scf::SCFTileAndFuseOptions options;
    options.setTilingOptions(std::move(tilingOptions));
    options.setFusionControlFn(
        [&](mlir::tensor::ExtractSliceOp, mlir::OpResult producer,
            bool isDestinationOperand)
            -> std::optional<
                mlir::scf::SCFTileAndFuseOptions::ControlFnResult> {
          if (isDestinationOperand || !exactFusionEdges->contains(producer))
            return std::nullopt;
          return mlir::scf::SCFTileAndFuseOptions::ControlFnResult{
              /*yieldProducerReplacement=*/false};
        });

    rewriter.setInsertionPoint(scope.operation);
    mlir::FailureOr<mlir::scf::SCFTileAndFuseResult> tiled =
        mlir::scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tiling,
                                                        options);
    if (mlir::failed(tiled))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "pinned SCF tile-and-fuse failed after temporal preflight");
    if (tiled->loops.size() != scope.loopOrder.size())
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "pinned SCF tiler returned an unexpected loop nest");
    llvm::SmallVector<mlir::Value, 4> replacements;
    for (mlir::Value result : scope.operation->getResults()) {
      auto replacement = tiled->replacements.find(result);
      if (replacement == tiled->replacements.end())
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "pinned SCF tiler omitted a traversal result replacement");
      replacements.push_back(replacement->second);
    }
    llvm::SmallVector<mlir::Operation *, 8> fusedProducers(
        tiled->fusedProducers.begin(), tiled->fusedProducers.end());
    rewriter.replaceOp(scope.operation, replacements);
    if (mlir::failed(eraseFusedProducers(rewriter, fusedProducers)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "exact-derived producer remained live after fusion");
    statistics.tiledTraversals++;
    statistics.loops += tiled->loops.size();
    statistics.fusedProducers += fusedProducers.size();
    if (mlir::failed(specializeRaggedTails(rewriter, descriptor, scope,
                                           tiled->loops, statistics)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "ragged temporal loop could not form one static tail");
  }

  if (mlir::failed(canonicalizeTiledRegion(region, &listener)) ||
      mlir::failed(refineLinalgStaticTypes(rewriter, region)) ||
      mlir::failed(refineOnlineAttentionStaticTypes(rewriter, region)) ||
      mlir::failed(canonicalizeTiledRegion(region, &listener)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "bounded temporal canonicalization did not converge");
  mlir::DominanceInfo dominance(region);
  mlir::eliminateCommonSubExpressions(rewriter, dominance, region);
  eraseDeadOperations(rewriter, region);
  if (!listener.finalizeAfterRewrite() || mlir::failed(mlir::verify(region)) ||
      mlir::failed(verifyStructuralTileRegions(
          region->getParentOfType<mlir::ModuleOp>())) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          region->getParentOfType<mlir::ModuleOp>(), relations)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "temporal tiling produced invalid current structural IR");
  return statistics;
}

} // namespace wafer
