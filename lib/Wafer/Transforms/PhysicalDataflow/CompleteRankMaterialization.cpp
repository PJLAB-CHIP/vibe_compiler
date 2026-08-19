//===- CompleteRankMaterialization.cpp - Complete-rank Tile IR ---------===//

#include "Wafer/Transforms/CompleteRankMaterialization.h"

#include "CompleteRankMaterializationInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>
#include <tuple>
#include <utility>

namespace wafer::complete_rank_materialization {

mlir::memref::GlobalOp findOrCreateDenseConstantGlobal(
    mlir::ModuleOp module, mlir::arith::ConstantOp constant,
    mlir::MemRefType memrefType, mlir::ElementsAttr elements) {
  mlir::memref::GlobalOp global;
  for (mlir::memref::GlobalOp candidate :
       module.getOps<mlir::memref::GlobalOp>()) {
    if (candidate.getType() == memrefType && candidate.getConstant() &&
        candidate.getConstantInitValue() == elements) {
      global = candidate;
      break;
    }
  }

  if (!global) {
    mlir::OpBuilder globalBuilder(module.getContext());
    mlir::SymbolTable symbolTable(module);
    global = globalBuilder.create<mlir::memref::GlobalOp>(
        constant.getLoc(), "__wafer_constant",
        globalBuilder.getStringAttr("private"), memrefType, elements,
        /*constant=*/true, /*alignment=*/mlir::IntegerAttr{});
    symbolTable.insert(global);
    global->moveBefore(&module.front());
  }

  return global;
}

void outlineRankDenseTensorConstants(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::arith::ConstantOp, 8> constants;
  module.walk([&](mlir::arith::ConstantOp constant) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
    auto elements = mlir::dyn_cast<mlir::ElementsAttr>(constant.getValue());
    if (tensorType && tensorType.hasStaticShape() && elements &&
        !elements.isSplat())
      constants.push_back(constant);
  });

  for (mlir::arith::ConstantOp constant : constants) {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(constant.getType());
    auto elements = mlir::cast<mlir::ElementsAttr>(constant.getValue());
    mlir::MemRefType memrefType = mlir::MemRefType::get(
        tensorType.getShape(), tensorType.getElementType(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(module.getContext(), MemorySpace::DDR,
                        MemLayout::Tensor));
    mlir::memref::GlobalOp global =
        findOrCreateDenseConstantGlobal(module, constant, memrefType, elements);
    mlir::OpBuilder builder(constant);
    builder.setInsertionPointAfter(constant);
    auto getGlobal = builder.create<mlir::memref::GetGlobalOp>(
        constant.getLoc(), memrefType, global.getSymName());
    auto tensor = builder.create<mlir::bufferization::ToTensorOp>(
        constant.getLoc(), getGlobal.getResult(), /*restrict=*/false,
        /*writable=*/false);
    constant.getResult().replaceAllUsesWith(tensor.getResult());
    constant.erase();
  }
}

} // namespace wafer::complete_rank_materialization

namespace wafer {
using namespace tensor_program_scheduling;

namespace {

static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
splitCompleteRankTileModuleAtDDRBoundary(
    mlir::OwningOpRef<mlir::ModuleOp> candidate, std::string *failureReason) {
  llvm::SmallVector<TileRegionOp, 8> regions;
  candidate->walk([&](TileRegionOp region) {
    if (!region->getParentOfType<TileRegionOp>())
      regions.push_back(region);
  });
  if (regions.empty()) {
    if (failureReason)
      *failureReason =
          "complete-rank residency split requires actual Tile regions";
    return mlir::failure();
  }
  auto isResidencyWork = [](mlir::Operation *operation) {
    return wafer::containsTileDataflowOperations(operation);
  };
  for (auto [regionOrdinal, original] : llvm::enumerate(regions)) {
    if (original.getBody().empty())
      continue;
    llvm::SmallVector<unsigned, 16> candidateIndices;
    llvm::SmallVector<mlir::Operation *, 16> originalOperations;
    for (mlir::Operation &operation :
         original.getBody().front().without_terminator())
      originalOperations.push_back(&operation);
    auto isMeaningfulCut = [&](unsigned index) {
      llvm::ArrayRef<mlir::Operation *> operations(originalOperations);
      return llvm::any_of(operations.take_front(index), isResidencyWork) &&
             llvm::any_of(operations.drop_front(index), isResidencyWork);
    };
    // A split is one bounded representative, not a search over every actual
    // clone. Rank legal DDR-clean cuts first by the static physical SPM bytes
    // owned on either side. Tile operation count is the deterministic fallback
    // and tie-breaker when the current IR has no complete byte estimate. This
    // is a generic current-IR structural prior only; exact completion, SPM and
    // final cost remain the common downstream gates.
    bool footprintOverflow = false;
    llvm::SmallVector<uint64_t, 16> prefixSPMBytes(
        originalOperations.size() + 1, 0);
    llvm::SmallVector<uint64_t, 16> prefixWork(originalOperations.size() + 1,
                                               0);
    for (auto [index, operation] : llvm::enumerate(originalOperations)) {
      uint64_t operationSPMBytes = 0;
      operation->walk([&](mlir::memref::AllocOp allocation) {
        if (!isWaferSPMMemRefType(allocation.getType()))
          return;
        std::optional<WaferPhysicalTensorInfo> info =
            computeWaferPhysicalTensorInfo(allocation.getType());
        if (!info || info->physicalBytes <= 0)
          return;
        uint64_t bytes = static_cast<uint64_t>(info->physicalBytes);
        if (bytes > std::numeric_limits<uint64_t>::max() - operationSPMBytes) {
          footprintOverflow = true;
          return;
        }
        operationSPMBytes += bytes;
      });
      uint64_t operationWork = 0;
      operation->walk([&](mlir::Operation *nested) {
        if (nested->getNumRegions() == 0 &&
            wafer::containsTileDataflowOperations(nested))
          ++operationWork;
      });
      if (operationSPMBytes >
          std::numeric_limits<uint64_t>::max() - prefixSPMBytes[index]) {
        footprintOverflow = true;
      } else {
        prefixSPMBytes[index + 1] = prefixSPMBytes[index] + operationSPMBytes;
      }
      prefixWork[index + 1] = prefixWork[index] + operationWork;
    }
    const bool hasSPMFootprintPrior =
        !footprintOverflow && prefixSPMBytes.back() != 0;
    for (unsigned index = 1; index < originalOperations.size(); ++index)
      if (isMeaningfulCut(index))
        candidateIndices.push_back(index);
    llvm::sort(candidateIndices, [&](unsigned left, unsigned right) {
      auto key = [&](unsigned index) {
        const uint64_t headBytes =
            hasSPMFootprintPrior ? prefixSPMBytes[index] : 0;
        const uint64_t tailBytes =
            hasSPMFootprintPrior ? prefixSPMBytes.back() - headBytes : 0;
        const uint64_t byteImbalance = headBytes > tailBytes
                                           ? headBytes - tailBytes
                                           : tailBytes - headBytes;
        const uint64_t head = prefixWork[index];
        const uint64_t tail = prefixWork.back() - head;
        const uint64_t imbalance = head > tail ? head - tail : tail - head;
        const bool traversalBoundary =
            mlir::isa<mlir::scf::ForOp>(originalOperations[index]);
        return std::make_tuple(std::max(headBytes, tailBytes), byteImbalance,
                               std::max(head, tail), imbalance,
                               !traversalBoundary, index);
      };
      return key(left) < key(right);
    });

    for (unsigned index : candidateIndices) {
      mlir::OwningOpRef<mlir::ModuleOp> trial =
          mlir::cast<mlir::ModuleOp>((*candidate)->clone());
      llvm::SmallVector<TileRegionOp, 8> trialRegions;
      trial->walk([&](TileRegionOp region) {
        if (!region->getParentOfType<TileRegionOp>())
          trialRegions.push_back(region);
      });
      if (regionOrdinal >= trialRegions.size())
        return mlir::failure();
      TileRegionOp trialRegion = trialRegions[regionOrdinal];
      llvm::SmallVector<mlir::Operation *, 16> trialOperations;
      for (mlir::Operation &operation :
           trialRegion.getBody().front().without_terminator())
        trialOperations.push_back(&operation);
      if (index >= trialOperations.size())
        return mlir::failure();
      if (mlir::succeeded(
              tensor_program_scheduling::partitionTileRegionAtDDRBoundary(
                  trialRegion, trialOperations[index])) &&
          mlir::succeeded(mlir::verify(*trial)))
        return std::move(trial);
    }
  }
  if (failureReason)
    *failureReason =
        "actual Tile clone has no explicit DDR-clean residency cut";
  return mlir::failure();
}

static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankSelectiveSpill(
    mlir::OwningOpRef<mlir::ModuleOp> candidate, std::string *failureReason) {
  llvm::SmallVector<TileRegionOp, 8> regions;
  candidate->walk([&](TileRegionOp region) {
    if (!region->getParentOfType<TileRegionOp>())
      regions.push_back(region);
  });
  if (regions.empty()) {
    if (failureReason)
      *failureReason =
          "complete-rank selective spill requires actual Tile regions";
    return mlir::failure();
  }

  struct SpillProposal {
    mlir::Value root;
    mlir::Operation *storeAfter = nullptr;
    mlir::Operation *reloadBefore = nullptr;
    uint64_t rootBytes = 0;
    uint64_t deadOperationCount = 0;
    unsigned blockOrdinal = 0;
    unsigned definitionIndex = 0;
    unsigned resultIndex = 0;
    unsigned storeIndex = 0;
    unsigned reloadIndex = 0;
    TileRegionOp region;
  };
  llvm::SmallVector<SpillProposal, 32> proposals;
  unsigned nextBlockOrdinal = 0;
  for (TileRegionOp region : regions) {
    if (region.getBody().empty())
      continue;
    llvm::SmallVector<mlir::Block *, 16> blocks;
    llvm::DenseSet<mlir::Block *> seenBlocks;
    blocks.push_back(&region.getBody().front());
    seenBlocks.insert(blocks.front());
    region.walk([&](mlir::Operation *operation) {
      mlir::Block *block = operation->getBlock();
      if (block && seenBlocks.insert(block).second)
        blocks.push_back(block);
    });
    for (mlir::Block *block : blocks) {
      unsigned blockOrdinal = nextBlockOrdinal++;
      llvm::SmallVector<mlir::Operation *, 32> operations;
      for (mlir::Operation &operation : block->without_terminator())
        operations.push_back(&operation);
      for (auto [definitionIndex, definition] : llvm::enumerate(operations)) {
        for (auto [resultIndex, root] :
             llvm::enumerate(definition->getResults())) {
          auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
          MemoryAttr memory =
              rootType ? getWaferMemoryAttr(rootType) : MemoryAttr{};
          if (!rootType || !memory || !isWaferSPMMemRefType(rootType))
            continue;
          mlir::FailureOr<int64_t> footprint =
              memory.getPhysicalFootprintBytes(rootType);
          if (mlir::failed(footprint) || *footprint <= 0)
            continue;
          for (unsigned storeIndex = definitionIndex;
               storeIndex + 1 < operations.size(); ++storeIndex) {
            for (unsigned reloadIndex = storeIndex + 2;
                 reloadIndex < operations.size(); ++reloadIndex) {
              if (!tensor_program_scheduling::canMaterializeSelectiveTileSpill(
                      region, root, operations[storeIndex],
                      operations[reloadIndex]))
                continue;
              proposals.push_back(
                  {root, operations[storeIndex], operations[reloadIndex],
                   static_cast<uint64_t>(*footprint),
                   static_cast<uint64_t>(reloadIndex - storeIndex - 1),
                   static_cast<unsigned>(blockOrdinal),
                   static_cast<unsigned>(definitionIndex),
                   static_cast<unsigned>(resultIndex), storeIndex, reloadIndex,
                   region});
            }
          }
        }
      }
    }
  }
  llvm::sort(
      proposals, [](const SpillProposal &left, const SpillProposal &right) {
        unsigned __int128 leftRelief =
            static_cast<unsigned __int128>(left.rootBytes) *
            left.deadOperationCount;
        unsigned __int128 rightRelief =
            static_cast<unsigned __int128>(right.rootBytes) *
            right.deadOperationCount;
        if (leftRelief != rightRelief)
          return leftRelief > rightRelief;
        if (left.rootBytes != right.rootBytes)
          return left.rootBytes > right.rootBytes;
        if (left.deadOperationCount != right.deadOperationCount)
          return left.deadOperationCount > right.deadOperationCount;
        return std::tie(left.blockOrdinal, left.definitionIndex,
                        left.resultIndex, left.storeIndex, left.reloadIndex) <
               std::tie(right.blockOrdinal, right.definitionIndex,
                        right.resultIndex, right.storeIndex, right.reloadIndex);
      });
  for (const SpillProposal &proposal : proposals)
    if (mlir::succeeded(
            tensor_program_scheduling::materializeSelectiveTileSpill(
                proposal.region, proposal.root, proposal.storeAfter,
                proposal.reloadBefore))) {
      if (mlir::failed(mlir::verify(*candidate)))
        return mlir::failure();
      return std::move(candidate);
    }
  if (failureReason)
    *failureReason =
        "actual Tile clone has no bounded selective-spill interval";
  return mlir::failure();
}

static mlir::Value traceTileBoundaryBase(mlir::Value value) {
  while (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>())
    value = subview.getSource();
  return value;
}

static bool isReadOnlyFunctionalBoundarySource(mlir::Value value) {
  auto argument =
      mlir::dyn_cast<mlir::BlockArgument>(traceTileBoundaryBase(value));
  if (!argument)
    return false;
  auto region =
      mlir::dyn_cast_or_null<TileRegionOp>(argument.getOwner()->getParentOp());
  if (!region || argument.getArgNumber() >= region.getInputs().size())
    return false;
  auto toMemref = region.getInputs()[argument.getArgNumber()]
                      .getDefiningOp<mlir::bufferization::ToMemrefOp>();
  if (!toMemref)
    return false;
  auto functionArgument =
      mlir::dyn_cast<mlir::BlockArgument>(toMemref.getTensor());
  return functionArgument && mlir::isa_and_nonnull<mlir::func::FuncOp>(
                                 functionArgument.getOwner()->getParentOp());
}

static bool hasOnlyReadUses(mlir::Value value) {
  if (value.use_empty())
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    auto effects =
        mlir::dyn_cast<mlir::MemoryEffectOpInterface>(use.getOwner());
    if (!effects)
      return false;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> onValue;
    effects.getEffectsOnValue(value, onValue);
    if (onValue.empty() || llvm::any_of(onValue, [](const auto &effect) {
          return !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        }))
      return false;
  }
  return true;
}

static StorageLoadOp
getReadOnlyBoundaryLoadForAlloc(mlir::memref::AllocOp alloc) {
  if (!alloc || !isWaferSPMMemRefType(alloc.getType()) ||
      !alloc.getDynamicSizes().empty())
    return {};
  StorageLoadOp load;
  for (mlir::OpOperand &use : alloc.getResult().getUses()) {
    if (auto candidate = mlir::dyn_cast<StorageLoadOp>(use.getOwner());
        candidate && candidate.getDest() == alloc.getResult()) {
      if (load || !isReadOnlyFunctionalBoundarySource(candidate.getSource()))
        return {};
      load = candidate;
      continue;
    }
    auto effects =
        mlir::dyn_cast<mlir::MemoryEffectOpInterface>(use.getOwner());
    if (!effects ||
        effects.getEffectOnValue<mlir::MemoryEffects::Write>(alloc.getResult()))
      return {};
  }
  return load;
}

static bool tracesReadOnlyBoundaryLoad(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (visited.insert(value).second) {
    if (auto layout = value.getDefiningOp<LayoutMaterializeOp>()) {
      value = layout.getSource();
      continue;
    }
    auto alloc = value.getDefiningOp<mlir::memref::AllocOp>();
    return alloc && getReadOnlyBoundaryLoadForAlloc(alloc);
  }
  return false;
}

static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
hoistInvariantReadOnlyTileMovement(mlir::OwningOpRef<mlir::ModuleOp> candidate,
                                   std::string *failureReason) {
  llvm::SmallVector<mlir::scf::ForOp, 8> loops;
  candidate->walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::scf::ForOp loop) { loops.push_back(loop); });
  unsigned movedMovement = 0;
  for (mlir::scf::ForOp loop : loops) {
    auto loopLike = mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation());
    mlir::moveLoopInvariantCode(
        {&loop.getRegion()},
        [&](mlir::Value value, mlir::Region *) {
          return loopLike.isDefinedOutsideOfLoop(value);
        },
        [&](mlir::Operation *operation, mlir::Region *) {
          if (mlir::isMemoryEffectFree(operation) &&
              mlir::isSpeculatable(operation))
            return true;
          if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(operation))
            return static_cast<bool>(getReadOnlyBoundaryLoadForAlloc(alloc));
          if (auto load = mlir::dyn_cast<StorageLoadOp>(operation))
            return load.getDest().getDefiningOp<mlir::memref::AllocOp>() &&
                   isReadOnlyFunctionalBoundarySource(load.getSource());
          if (auto layout = mlir::dyn_cast<LayoutMaterializeOp>(operation))
            return tracesReadOnlyBoundaryLoad(layout.getSource()) &&
                   hasOnlyReadUses(layout.getResult());
          return false;
        },
        [&](mlir::Operation *operation, mlir::Region *) {
          movedMovement +=
              mlir::isa<StorageLoadOp, LayoutMaterializeOp>(operation);
          loopLike.moveOutOfLoop(operation);
        });
  }
  if (movedMovement == 0) {
    if (failureReason)
      *failureReason =
          "actual Tile clone has no invariant read-only boundary movement";
    return mlir::failure();
  }
  if (mlir::failed(mlir::verify(*candidate))) {
    if (failureReason)
      *failureReason =
          "invariant read-only Tile movement produced invalid actual IR";
    return mlir::failure();
  }
  return std::move(candidate);
}

} // namespace

bool isCompleteRankTensorProgramRankInvariant(mlir::ModuleOp sourceModule) {
  if (!sourceModule)
    return false;
  bool rankDependent = false;
  sourceModule.walk([&](mlir::Operation *operation) {
    rankDependent |=
        mlir::isa<LinalgExtCollectiveAllGatherOp,
                  LinalgExtCollectiveReduceScatterOp,
                  LinalgExtCollectiveAllReduceOp, LinalgExtCollectiveAllToAllOp,
                  LinalgExtCollectiveCollectivePermuteOp>(operation);
    return rankDependent ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
  });
  return !rankDependent;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeConservativeCompleteRankBaseline(mlir::ModuleOp sourceModule,
                                            int64_t logicalRank) {
  if (!sourceModule || logicalRank < 0) {
    if (sourceModule)
      sourceModule.emitError(
          "conservative complete-rank baseline requires a non-negative "
          "logical rank");
    return mlir::failure();
  }

  mlir::OwningOpRef<mlir::ModuleOp> stagedModule =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone());
  complete_rank_materialization::outlineRankDenseTensorConstants(*stagedModule);

  bool hasStructuredRoot = false;
  stagedModule->walk([&](mlir::Operation *operation) {
    hasStructuredRoot |= classifyCandidateTraversalRoot(operation) !=
                         CandidateTraversalRootCapability::Unsupported;
    return hasStructuredRoot ? mlir::WalkResult::interrupt()
                             : mlir::WalkResult::advance();
  });

  mlir::OwningOpRef<mlir::ModuleOp> baselineModule;
  if (hasStructuredRoot) {
    unsigned regionCount = 0;
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> materialized =
        materializeCompleteRankTileModule(*stagedModule, logicalRank,
                                          &regionCount);
    if (mlir::failed(materialized) || regionCount != 1)
      return mlir::failure();
    baselineModule = std::move(*materialized);
  } else {
    baselineModule = std::move(stagedModule);
    clearRankCandidatePhysicalFacts(*baselineModule);
  }

  if (mlir::failed(mlir::verify(*baselineModule))) {
    baselineModule->emitError(
        "conservative complete-rank baseline failed verification");
    return mlir::failure();
  }
  return std::move(baselineModule);
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankCandidateTileModule(
    mlir::ModuleOp sourceModule, int64_t logicalRank,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    CandidateTileTraversalKind traversalKind,
    CompleteRankTraversalComposition composition,
    CandidateTileResidencyAction residencyAction,
    CandidateBoundaryMovementAction boundaryMovementAction,
    CandidateLoopMovementAction loopMovementAction, std::string *failureReason,
    std::optional<TargetImplementationKind> selectedImplementation) {
  if (!sourceModule || logicalRank < 0 || candidateTileSizes.empty())
    return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> stagedModule =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone());
  complete_rank_materialization::outlineRankDenseTensorConstants(*stagedModule);
  mlir::OwningOpRef<mlir::ModuleOp> candidate;
  if (mlir::failed(lowerCompleteRankCandidateTensorProgramToTileRegionModule(
          *stagedModule, candidateTileSizes, candidateReductionTileSizes,
          traversalKind, composition, candidate, failureReason, logicalRank,
          selectedImplementation,
          boundaryMovementAction ==
              CandidateBoundaryMovementAction::ExactDirectMapped)))
    return mlir::failure();
  if (loopMovementAction ==
      CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary) {
    auto hoisted =
        hoistInvariantReadOnlyTileMovement(std::move(candidate), failureReason);
    if (mlir::failed(hoisted))
      return mlir::failure();
    candidate = std::move(*hoisted);
  }
  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> realized =
      std::move(candidate);
  switch (residencyAction) {
  case CandidateTileResidencyAction::KeepSingleRegion:
    break;
  case CandidateTileResidencyAction::SplitAtExplicitDDRBoundary:
    realized = splitCompleteRankTileModuleAtDDRBoundary(std::move(*realized),
                                                        failureReason);
    break;
  case CandidateTileResidencyAction::SelectiveSpill:
    realized = materializeCompleteRankSelectiveSpill(std::move(*realized),
                                                     failureReason);
    break;
  }
  if (mlir::failed(realized))
    return mlir::failure();
  return std::move(*realized);
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankConnectionTileModule(
    mlir::ModuleOp sourceModule, int64_t logicalRank,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<CandidateTraversalConnectionAction> connectionActions,
    CandidateBoundaryMovementAction boundaryMovementAction,
    CandidateLoopMovementAction loopMovementAction, std::string *failureReason,
    std::optional<TargetImplementationKind> selectedImplementation) {
  if (!sourceModule || logicalRank < 0 || connectionActions.empty())
    return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> stagedModule =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone());
  complete_rank_materialization::outlineRankDenseTensorConstants(*stagedModule);
  mlir::OwningOpRef<mlir::ModuleOp> candidate;
  if (mlir::failed(lowerCompleteRankConnectionTensorProgramToTileRegionModule(
          *stagedModule, candidateTileSizes, connectionActions, candidate,
          failureReason, logicalRank, selectedImplementation,
          boundaryMovementAction ==
              CandidateBoundaryMovementAction::ExactDirectMapped)))
    return mlir::failure();

  if (loopMovementAction ==
      CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary) {
    auto hoisted =
        hoistInvariantReadOnlyTileMovement(std::move(candidate), failureReason);
    if (mlir::failed(hoisted))
      return mlir::failure();
    candidate = std::move(*hoisted);
  }

  unsigned crossRegionCount = llvm::count(
      connectionActions, CandidateTraversalConnectionAction::CrossRegion);
  unsigned selectiveSpillCount = llvm::count(
      connectionActions, CandidateTraversalConnectionAction::SelectiveSpill);
  for (unsigned index = 0; index < selectiveSpillCount; ++index) {
    auto spilled = materializeCompleteRankSelectiveSpill(std::move(candidate),
                                                         failureReason);
    if (mlir::failed(spilled))
      return mlir::failure();
    candidate = std::move(*spilled);
  }
  for (unsigned index = 0; index < crossRegionCount; ++index) {
    auto split = splitCompleteRankTileModuleAtDDRBoundary(std::move(candidate),
                                                          failureReason);
    if (mlir::failed(split))
      return mlir::failure();
    candidate = std::move(*split);
  }
  if (mlir::failed(mlir::verify(*candidate))) {
    if (failureReason)
      *failureReason =
          "connection actions produced invalid complete-rank Tile IR";
    return mlir::failure();
  }
  return candidate;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankConnectionChoicesTileModule(
    mlir::ModuleOp sourceModule, int64_t logicalRank,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> connectionChoices,
    CandidateBoundaryMovementAction boundaryMovementAction,
    CandidateLoopMovementAction loopMovementAction, std::string *failureReason,
    std::optional<TargetImplementationKind> selectedImplementation) {
  if (!sourceModule || logicalRank < 0 || connectionChoices.empty())
    return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> stagedModule =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone());
  complete_rank_materialization::outlineRankDenseTensorConstants(*stagedModule);
  mlir::OwningOpRef<mlir::ModuleOp> candidate;
  if (mlir::failed(
          lowerCompleteRankConnectionChoicesTensorProgramToTileRegionModule(
              *stagedModule, connectionChoices, candidate, failureReason,
              logicalRank, selectedImplementation,
              boundaryMovementAction ==
                  CandidateBoundaryMovementAction::ExactDirectMapped)))
    return mlir::failure();

  if (loopMovementAction ==
      CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary) {
    auto hoisted =
        hoistInvariantReadOnlyTileMovement(std::move(candidate), failureReason);
    if (mlir::failed(hoisted))
      return mlir::failure();
    candidate = std::move(*hoisted);
  }

  unsigned crossRegionCount = llvm::count_if(
      connectionChoices, [](const CandidateTraversalConnectionChoice &choice) {
        return choice.action == CandidateTraversalConnectionAction::CrossRegion;
      });
  unsigned selectiveSpillCount = llvm::count_if(
      connectionChoices, [](const CandidateTraversalConnectionChoice &choice) {
        return choice.action ==
               CandidateTraversalConnectionAction::SelectiveSpill;
      });
  for (unsigned index = 0; index < selectiveSpillCount; ++index) {
    auto spilled = materializeCompleteRankSelectiveSpill(std::move(candidate),
                                                         failureReason);
    if (mlir::failed(spilled))
      return mlir::failure();
    candidate = std::move(*spilled);
  }
  for (unsigned index = 0; index < crossRegionCount; ++index) {
    auto split = splitCompleteRankTileModuleAtDDRBoundary(std::move(candidate),
                                                          failureReason);
    if (mlir::failed(split))
      return mlir::failure();
    candidate = std::move(*split);
  }
  if (mlir::failed(mlir::verify(*candidate))) {
    if (failureReason)
      *failureReason =
          "connection choices produced invalid complete-rank Tile IR";
    return mlir::failure();
  }
  return candidate;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankTileResidencySibling(
    mlir::ModuleOp tileParent, CandidateTileResidencyAction residencyAction,
    std::string *failureReason) {
  if (!tileParent ||
      residencyAction == CandidateTileResidencyAction::KeepSingleRegion) {
    if (failureReason)
      *failureReason =
          "Tile residency sibling requires an actual non-identity action";
    return mlir::failure();
  }
  bool hasTile = false;
  bool hasInstruction = false;
  tileParent.walk([&](mlir::Operation *operation) {
    hasTile |= mlir::isa<TileRegionOp>(operation);
    hasInstruction |=
        mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(operation);
  });
  if (!hasTile || hasInstruction) {
    if (failureReason)
      *failureReason =
          "residency repair requires unplaced Tile IR without Instr state";
    return mlir::failure();
  }

  mlir::OwningOpRef<mlir::ModuleOp> sibling =
      mlir::cast<mlir::ModuleOp>(tileParent->clone());
  switch (residencyAction) {
  case CandidateTileResidencyAction::KeepSingleRegion:
    llvm_unreachable("identity residency action rejected above");
  case CandidateTileResidencyAction::SplitAtExplicitDDRBoundary:
    return splitCompleteRankTileModuleAtDDRBoundary(std::move(sibling),
                                                    failureReason);
  case CandidateTileResidencyAction::SelectiveSpill:
    return materializeCompleteRankSelectiveSpill(std::move(sibling),
                                                 failureReason);
  }
  llvm_unreachable("unknown complete-rank Tile residency action");
}

} // namespace wafer
