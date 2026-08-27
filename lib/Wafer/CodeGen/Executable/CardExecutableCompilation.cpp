//===- CardExecutableCompilation.cpp - Policy-free executable seam ------===//

#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"
#include "Wafer/Analysis/Structured/StructuredNodeUseIndex.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/Target/TargetTopology.h"

#include "Wafer/CodeGen/Executable/BoundedTileExecutor.h"

#include "Scheduling/RedundantTransferElimination.h"
#include "Wafer/Conversion/WaferCardModuleToTileModules/WaferCardModuleToTileModules.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Planning/PhysicalDataflow/CurrentIRLayoutOptimization.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/MemoryPlanningPipelines.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

static std::optional<int64_t> getPeerTileId(mlir::Operation *operation) {
  if (auto peerSend = mlir::dyn_cast<CommPeerSendOp>(operation))
    return peerSend.getPeerAttr().getInt();
  if (auto peerRecv = mlir::dyn_cast<CommPeerRecvOp>(operation))
    return peerRecv.getPeerAttr().getInt();
  if (auto dteSend = mlir::dyn_cast<InstrDTESendOp>(operation))
    return dteSend.getPeerAttr().getInt();
  if (auto dteRecv = mlir::dyn_cast<InstrDTERecvOp>(operation))
    return dteRecv.getPeerAttr().getInt();
  return std::nullopt;
}

static mlir::LogicalResult
verifyCardModuleStage(mlir::ModuleOp module, CardId expectedCardId,
                      llvm::ArrayRef<TileId> expectedTileIds,
                      std::string &detail) {
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(module, &detail);
  if (mlir::failed(topology))
    return mlir::failure();
  if (!topology->getCardCoordinate(expectedCardId)) {
    detail = "selected card_id is outside the target topology";
    return mlir::failure();
  }
  std::optional<llvm::ArrayRef<TileId>> topologyTiles =
      topology->getAvailableTileIds(expectedCardId);
  if (!topologyTiles || topologyTiles->size() != expectedTileIds.size() ||
      !std::equal(topologyTiles->begin(), topologyTiles->end(),
                  expectedTileIds.begin(), expectedTileIds.end())) {
    detail = "selected Tile domain does not match target topology";
    return mlir::failure();
  }

  llvm::SmallVector<CardModuleOp, 2> cards(module.getOps<CardModuleOp>());
  if (cards.size() != 1) {
    detail = "expected exactly one direct wafer.card.module";
    return mlir::failure();
  }
  CardModuleOp card = cards.front();
  if (card.getCardIdAttr().getInt() != expectedCardId.getValue()) {
    detail = "CardModule card_id does not match the selected card";
    return mlir::failure();
  }

  llvm::DenseSet<int64_t> expectedTiles;
  for (TileId tile : expectedTileIds)
    if (tile.getValue() < 0 || !expectedTiles.insert(tile.getValue()).second) {
      detail = "selected Tile domain must be non-negative and unique";
      return mlir::failure();
    }
  llvm::DenseSet<int64_t> actualTiles;
  for (TileModuleOp tile : card.getBody().front().getOps<TileModuleOp>()) {
    const int64_t tileId = tile.getTileIdAttr().getInt();
    if (!expectedTiles.contains(tileId) || !actualTiles.insert(tileId).second) {
      detail = "CardModule has an unexpected or duplicate tile_id";
      return mlir::failure();
    }
    mlir::WalkResult peerResult = tile.walk([&](mlir::Operation *operation) {
      std::optional<int64_t> peer = getPeerTileId(operation);
      if (!peer || expectedTiles.contains(*peer))
        return mlir::WalkResult::advance();
      operation->emitOpError()
          << "peer tile_id " << *peer
          << " is outside the selected available Tile domain";
      return mlir::WalkResult::interrupt();
    });
    if (peerResult.wasInterrupted()) {
      detail = "CardModule has a peer outside the selected Tile domain";
      return mlir::failure();
    }
  }
  if (actualTiles.size() != expectedTiles.size()) {
    detail = "CardModule does not cover all selected available Tiles";
    return mlir::failure();
  }

  llvm::DenseSet<int64_t> cardDDRResources;
  for (mlir::memref::GlobalOp global :
       card.getBody().front().getOps<mlir::memref::GlobalOp>())
    if (auto resource = global->getAttrOfType<CardDDRResourceAttr>(
            kWaferCardDDRResourceAttrName))
      cardDDRResources.insert(resource.getResourceId());
  for (TileModuleOp tile : card.getBody().front().getOps<TileModuleOp>()) {
    llvm::DenseMap<int64_t, unsigned> bindings;
    tile.walk([&](mlir::func::FuncOp function) {
      for (unsigned argument = 0; argument < function.getNumArguments();
           ++argument)
        if (auto binding = function.getArgAttrOfType<CardDDRBindingAttr>(
                argument, kWaferCardDDRBindingAttrName))
          ++bindings[binding.getResourceId()];
    });
    for (int64_t resource : cardDDRResources)
      if (bindings.lookup(resource) != 1) {
        detail = "every selected Tile must bind each card DDR resource "
                 "exactly once";
        return mlir::failure();
      }
  }
  return mlir::success();
}

struct TileLoweringResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
  std::vector<CandidateInstructionIR::RegionNodeRelation> regionNodes;
  wafer::support::CompileIRInventory instructionInventory;
  std::string detail;
  TileMemoryPlanningFailure memoryPlanning;
  bool conversionFailed = false;
  bool memoryPlanningFailed = false;
  uint64_t redundantTransfersEliminated = 0;
};

static std::string captureTileIR(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

static void
retargetStructuredRelationValue(StructuredMaterializationRelations &relations,
                                mlir::Value oldValue, mlir::Value newValue) {
  if (!oldValue || !newValue || oldValue.getType() != newValue.getType())
    return;
  auto retarget = [&](auto &entries) {
    for (auto &entry : entries)
      if (entry.buffer == oldValue)
        entry.buffer = newValue;
  };
  retarget(relations.operationResultBuffers);
  retarget(relations.operandBuffers);
  retarget(relations.scratchBuffers);
  retarget(relations.outputBuffers);
  retarget(relations.cardDDRBuffers);
  retarget(relations.partialReductionContributions);
  retarget(relations.partialReductionMergeInputs);
}

static mlir::FailureOr<unsigned> cleanupCanonicalInstructionTransfersImpl(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations) {
  if (!module)
    return mlir::failure();
  const unsigned eliminated =
      wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
          module, [&](mlir::Value oldValue, mlir::Value newValue) {
            retargetStructuredRelationValue(relations, oldValue, newValue);
          });
  retainCurrentStructuredBufferRelations(module.getOperation(), relations);
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations)) ||
      mlir::failed(mlir::verify(module)))
    return mlir::failure();
  return eliminated;
}

static mlir::LogicalResult bufferizeTileDataflowFunctionBoundaries(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    wafer::support::CompileIRInventory *bufferizedInventory,
    std::string &detail) {
  if (mlir::failed(
          closeCurrentTileDataflowOwnerRelations(module, relations, &detail)))
    return mlir::failure();
  // One-Shot function-boundary bufferization may rebuild structured loop
  // bodies. Rebase attribution to the unique current allocation/view root
  // before that mutation so the caller-owned relation remains attached to an
  // actual storage object instead of an operation-result pointer that the pass
  // is allowed to replace.
  if (mlir::failed(rebaseStructuredBufferRelationsToStorageRoots(relations))) {
    detail = "function-boundary bufferization has a non-unique current "
             "storage owner";
    return mlir::failure();
  }
  if (mlir::failed(
          runPassPipeline(module, "instr-function-boundary-bufferization",
                          wafer::buildBufferizeInstrFunctionsPipeline))) {
    detail = "Instr function-boundary bufferization failed";
    return mlir::failure();
  }
  mlir::memref::CopyOp unownedCopy;
  module.walk([&](mlir::memref::CopyOp copy) {
    if (!unownedCopy && !copy->getParentOfType<TileRegionOp>())
      unownedCopy = copy;
  });
  if (unownedCopy) {
    llvm::raw_string_ostream diagnostic(detail);
    diagnostic << "function-boundary bufferization left memref.copy outside "
                  "a TileRegion owner: ";
    unownedCopy.print(diagnostic, mlir::OpPrintingFlags().skipRegions());
    return mlir::failure();
  }
  retainCurrentStructuredBufferRelations(module.getOperation(), relations);
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "Instr function-boundary bufferization changed a selected "
             "Tile buffer relation";
    return mlir::failure();
  }
  if (mlir::failed(
          closeCurrentTileDataflowOwnerRelations(module, relations, &detail))) {
    if (detail.empty())
      detail = "function-boundary bufferization lost a current layout owner";
    return mlir::failure();
  }
  if (bufferizedInventory)
    bufferizedInventory->record(module.getOperation());
  return mlir::success();
}

static mlir::LogicalResult convertTileRegionsToInstructionIR(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    std::vector<CandidateInstructionIR::RegionNodeRelation> &regionNodes,
    std::string &detail) {
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "CardModule splitting produced buffer relations outside the "
             "current IR";
    return mlir::failure();
  }
  if (mlir::failed(rebaseStructuredBufferRelationsToStorageRoots(relations))) {
    detail = "Instr function-boundary bufferization cannot preserve a unique "
             "storage root for every selected relation";
    return mlir::failure();
  }
  // Function-boundary bufferization has completed before this stage. Capture
  // current TileRegion ownership immediately before the listener-tracked
  // Tile-to-Instr conversion.
  llvm::SmallVector<TileRegionOp, 4> regions;
  module.walk([&](TileRegionOp region) { regions.push_back(region); });
  std::map<mlir::Operation *, std::set<uint32_t>> nodesByRegion;
  for (const StructuredOperationEmissionRelation &relation :
       relations.operationEmissions) {
    TileRegionOp region =
        relation.operation ? relation.operation->getParentOfType<TileRegionOp>()
                           : TileRegionOp{};
    if (region)
      nodesByRegion[region].insert(relation.structuredNodeId);
  }
  for (const auto &[region, nodes] : nodesByRegion)
    regionNodes.push_back(
        {region, std::vector<uint32_t>(nodes.begin(), nodes.end())});
  {
    StructuredBufferReplacementListener replacementListener(relations);
    TileRegionToInstrLoweringSession loweringSession(*module.getContext(),
                                                     &replacementListener);
    for (TileRegionOp region : regions)
      if (mlir::failed(convertTileRegionToInstr(region, loweringSession,
                                                &replacementListener))) {
        detail = "TileRegion-to-Instr conversion failed";
        return mlir::failure();
      }
    if (mlir::failed(convertBufferizationCopiesToInstr(module, loweringSession,
                                                       &replacementListener))) {
      detail = "function-boundary memref copy lowering failed";
      return mlir::failure();
    }

    if (!replacementListener.finalizeAfterRewrite() ||
        mlir::failed(checkStructuredBufferRelationsCurrent(
            module.getOperation(), relations))) {
      detail = replacementListener.getFailureReason().empty()
                   ? "TileRegion-to-Instr lowering did not preserve every "
                     "structured buffer relation in the current Tile IR"
                   : replacementListener.getFailureReason().str();
      return mlir::failure();
    }
  }

  return mlir::success();
}

} // namespace

bool isProvenExactTileMemoryPlanningFailure(
    const TileMemoryPlanningFailure &failure) {
  return failure.kind == TileMemoryPlanningFailureKind::SPMAllocation &&
         failure.spmPlanningFailureKind ==
             SPMMemoryPlanningFailureKind::CapacityOverflow;
}

mlir::FailureOr<unsigned> cleanupCanonicalInstructionTransfers(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations) {
  return cleanupCanonicalInstructionTransfersImpl(module, relations);
}

namespace {

static CardExecutableCompilationResult
fail(CardExecutableCompilationStatus status, llvm::StringRef gate,
     llvm::StringRef detail,
     llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures = {}) {
  CardExecutableCompilationResult result;
  result.status = status;
  result.gate = gate.str();
  result.detail = detail.str();
  result.tileFailures = std::move(tileFailures);
  return result;
}

static CardExecutableCompilationStatus
classifyTileMemoryFailureImpl(const TileMemoryPlanningFailure &failure) {
  switch (failure.kind) {
  case TileMemoryPlanningFailureKind::None:
  case TileMemoryPlanningFailureKind::Contract:
  case TileMemoryPlanningFailureKind::PreexistingPlacementFacts:
  case TileMemoryPlanningFailureKind::Verification:
    return CardExecutableCompilationStatus::CompilerFailure;
  case TileMemoryPlanningFailureKind::SPMAllocation:
    switch (failure.spmPlanningFailureKind) {
    case SPMMemoryPlanningFailureKind::CapacityOverflow:
      return CardExecutableCompilationStatus::ProvenExactRejection;
    case SPMMemoryPlanningFailureKind::ResourceExhausted:
      return CardExecutableCompilationStatus::IndeterminateFailure;
    case SPMMemoryPlanningFailureKind::UnsupportedLifetime:
      return CardExecutableCompilationStatus::UnsupportedFailure;
    case SPMMemoryPlanningFailureKind::MissingCompletion:
    case SPMMemoryPlanningFailureKind::Other:
    case SPMMemoryPlanningFailureKind::None:
      return CardExecutableCompilationStatus::CompilerFailure;
    }
  }
  return CardExecutableCompilationStatus::CompilerFailure;
}

static CardExecutableCompilationStatus
combineTileMemoryFailureStatus(CardExecutableCompilationStatus current,
                               CardExecutableCompilationStatus next) {
  auto priority = [](CardExecutableCompilationStatus status) {
    switch (status) {
    case CardExecutableCompilationStatus::CompilerFailure:
      return 4;
    case CardExecutableCompilationStatus::IndeterminateFailure:
      return 3;
    case CardExecutableCompilationStatus::UnsupportedFailure:
      return 2;
    case CardExecutableCompilationStatus::ProvenExactRejection:
      return 1;
    case CardExecutableCompilationStatus::Accepted:
      return 0;
    }
    return 4;
  };
  return priority(next) > priority(current) ? next : current;
}

} // namespace

CardExecutableCompilationStatus
classifyTileMemoryPlanningFailure(const TileMemoryPlanningFailure &failure) {
  return classifyTileMemoryFailureImpl(failure);
}

CardExecutableCompilationResult compileCanonicalInstructionTilesToExecutable(
    std::vector<CanonicalInstructionTile> tiles, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    CardExecutableLoweringStatistics *statistics,
    unsigned tilePipelineParallelism) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "stage", "actual-memory-target-gate", "canonical-instr-to-executable");
  if (statistics)
    ++statistics->actualMemoryTargetGateInvocations;
  auto failLeaf = [&](CardExecutableCompilationResult result) {
    totalTiming.markFailed();
    return result;
  };

  if (tiles.empty() || tiles.size() != expectedTileIds.size())
    return failLeaf(fail(CardExecutableCompilationStatus::CompilerFailure,
                         "actual-memory-target-input",
                         "canonical Instr Tile domain is incomplete"));

  mlir::MLIRContext *context = nullptr;
  std::vector<TileLoweringResult> loweringResults(tiles.size());
  for (auto [index, tile] : llvm::enumerate(tiles)) {
    if (!tile.module || tile.card != expectedCardId ||
        tile.tile != expectedTileIds[index])
      return failLeaf(fail(CardExecutableCompilationStatus::CompilerFailure,
                           "actual-memory-target-input",
                           "canonical Instr Tile identity is inconsistent"));
    if (!context)
      context = tile.module->getContext();
    if (tile.module->getContext() != context ||
        containsTileDataflowOperations(tile.module->getOperation()) ||
        mlir::failed(mlir::verify(*tile.module)) ||
        mlir::failed(checkStructuredBufferRelationsCurrent(
            tile.module->getOperation(), tile.relations)))
      return failLeaf(fail(
          CardExecutableCompilationStatus::CompilerFailure,
          "actual-memory-target-input",
          "actual leaf requires verifier-legal canonical Instr and current "
          "buffer owner relations"));
    loweringResults[index].module = std::move(tile.module);
    loweringResults[index].materializationRelations = std::move(tile.relations);
  }

  auto planMemory = [&](size_t tileIndex) {
    TileLoweringResult &result = loweringResults[tileIndex];
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> memoryPlanned =
        planTileMemory(std::move(result.module), &result.memoryPlanning,
                       &result.materializationRelations,
                       /*emitSPMCapacityDiagnostics=*/false);
    if (mlir::failed(memoryPlanned)) {
      result.memoryPlanningFailed = true;
      result.detail = "Tile memory planning failed";
      return;
    }
    result.module = std::move(*memoryPlanned);
  };

  const unsigned requestedWorkers = tilePipelineParallelism == 0
                                        ? kMaximumBoundedTilePipelineWorkers
                                        : tilePipelineParallelism;
  unsigned workers = runBoundedTilePipelines(context, loweringResults.size(),
                                             planMemory, requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);

  llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> instructionModules;
  std::vector<StructuredMaterializationRelations> tileRelations;
  instructionModules.reserve(loweringResults.size());
  tileRelations.reserve(loweringResults.size());
  CardExecutableCompilationStatus failureStatus =
      CardExecutableCompilationStatus::ProvenExactRejection;
  for (auto [tileIndex, result] : llvm::enumerate(loweringResults)) {
    if (result.memoryPlanningFailed || !result.module) {
      CardExecutableTileFailure failure;
      failure.tileId = expectedTileIds[tileIndex];
      failure.gate = result.memoryPlanning.kind ==
                             TileMemoryPlanningFailureKind::SPMAllocation
                         ? "spm-allocation"
                         : "tile-memory-planning";
      failure.detail = std::move(result.detail);
      failure.memoryPlanning = std::move(result.memoryPlanning);
      failureStatus = combineTileMemoryFailureStatus(
          failureStatus,
          classifyTileMemoryPlanningFailure(failure.memoryPlanning));
      tileFailures.push_back(std::move(failure));
      continue;
    }
    instructionModules.push_back(std::move(result.module));
    tileRelations.push_back(std::move(result.materializationRelations));
  }
  if (!tileFailures.empty()) {
    const std::string primaryGate = tileFailures.front().gate;
    const std::string primaryDetail = tileFailures.front().detail;
    return failLeaf(fail(failureStatus, primaryGate, primaryDetail,
                         std::move(tileFailures)));
  }

  CardExecutableLoweringFailure loweringFailure;
  mlir::FailureOr<CardExecutableLoweringResult> executable =
      lowerTileModulesToCardExecutable(
          std::move(instructionModules), program, executionConfig, diagnostics,
          loweringFailure, programData, statistics, tilePipelineParallelism);
  if (mlir::failed(executable))
    return failLeaf(
        fail(loweringFailure.isProvenExactRejection()
                 ? CardExecutableCompilationStatus::ProvenExactRejection
                 : CardExecutableCompilationStatus::IndeterminateFailure,
             loweringFailure.getDiagnosticLabel(),
             loweringFailure.detail.empty() ? "Tile module lowering failed"
                                            : loweringFailure.detail));

  if (executable->tiles.size() != expectedTileIds.size())
    return failLeaf(fail(CardExecutableCompilationStatus::CompilerFailure,
                         "card-executable-domain",
                         "Tile executable domain is incomplete"));
  for (auto [index, tile] : llvm::enumerate(executable->tiles))
    if (tile.getCardId() != expectedCardId ||
        tile.getTileId() != expectedTileIds[index])
      return failLeaf(
          fail(CardExecutableCompilationStatus::CompilerFailure,
               "card-executable-domain",
               "executable lowering changed the selected Tile identity"));

  CardExecutableCompilationResult result;
  result.status = CardExecutableCompilationStatus::Accepted;
  result.executable.emplace(std::move(*executable));
  for (auto [tileIndex, tile] : llvm::enumerate(result.executable->tiles)) {
    const StructuredMaterializationRelations &relations =
        tileRelations[tileIndex];
    StructuredNodeUseIndex nodeUses(relations);
    tile.getModule().walk([&](mlir::Operation *operation) {
      for (uint32_t node : nodeUses.collectNodesUsedBy(operation))
        result.operationNodeRelations.push_back({operation, node});
    });
  }
  return result;
}

CardExecutableCompilationResult compileCardModuleToExecutable(
    mlir::OwningOpRef<mlir::ModuleOp> cardModule, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    const StructuredMaterializationRelations &materializationRelations,
    CardExecutablePreparation &preparation,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    CardExecutableLoweringStatistics *statistics,
    unsigned tilePipelineParallelism, bool captureTileIRTrace) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "stage", "card-module-to-executable", "card-executable-compilation");
  if (statistics)
    ++statistics->cardModuleCompilationInvocations;

  auto reportFailure = [&](CardExecutableCompilationResult result) {
    totalTiming.markFailed();
    llvm::StringRef outcome = "indeterminate";
    if (result.isProvenExactRejection())
      outcome = "exact-rejection";
    else if (result.status ==
             CardExecutableCompilationStatus::UnsupportedFailure)
      outcome = "unsupported";
    else if (result.status == CardExecutableCompilationStatus::CompilerFailure)
      outcome = "compiler-failure";
    diagnostics << "wafer-compile: card-executable-compilation outcome="
                << outcome << " gate=" << result.gate
                << " detail=" << result.detail << '\n';
    return result;
  };
  auto preparationStatus = [](CardExecutablePreparationFailureKind kind) {
    switch (kind) {
    case CardExecutablePreparationFailureKind::Unsupported:
      return CardExecutableCompilationStatus::UnsupportedFailure;
    case CardExecutablePreparationFailureKind::ExactRejection:
      return CardExecutableCompilationStatus::ProvenExactRejection;
    case CardExecutablePreparationFailureKind::Indeterminate:
      return CardExecutableCompilationStatus::IndeterminateFailure;
    case CardExecutablePreparationFailureKind::CompilerBug:
      return CardExecutableCompilationStatus::CompilerFailure;
    }
    return CardExecutableCompilationStatus::CompilerFailure;
  };

  if (!cardModule || expectedTileIds.empty())
    return reportFailure(fail(
        CardExecutableCompilationStatus::IndeterminateFailure,
        "compilation-contract", "CardModule or Tile domain is incomplete"));

  mlir::MLIRContext *context = cardModule->getContext();
  const unsigned requestedWorkers = tilePipelineParallelism == 0
                                        ? kMaximumBoundedTilePipelineWorkers
                                        : tilePipelineParallelism;
  std::string detail;
  if (mlir::failed(verifyCardModuleStage(*cardModule, expectedCardId,
                                         expectedTileIds, detail)))
    return reportFailure(fail(CardExecutableCompilationStatus::CompilerFailure,
                              "card-module-stage", detail));
  mlir::FailureOr<llvm::SmallVector<TileModule, 16>> projectedModules;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "conversion", "card-module-to-executable",
        "card-module-to-tile-modules");
    projectedModules = splitCardModuleIntoTileModules(
        std::move(cardModule), &detail, &materializationRelations);
  }
  if (mlir::failed(projectedModules))
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "card-module-to-tile-modules", detail));
  if (projectedModules->size() != expectedTileIds.size())
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "card-module-to-tile-modules",
             "CardModule does not contain the expected Tile domain"));

  for (auto [index, tile] : llvm::enumerate(*projectedModules))
    if (tile.cardId != expectedCardId || tile.tileId != expectedTileIds[index])
      return reportFailure(
          fail(CardExecutableCompilationStatus::CompilerFailure,
               "card-module-to-tile-modules",
               "per-Tile modules changed the selected Tile domain"));

  std::vector<CurrentIRLayoutModule> layoutModules;
  layoutModules.reserve(projectedModules->size());
  for (TileModule &tile : *projectedModules)
    layoutModules.push_back({*tile.module, &tile.materializationRelations});
  CurrentIRLayoutOptimizationResult layoutOptimization =
      optimizeCurrentIRLayouts(layoutModules);
  if (statistics) {
    statistics->currentIRLayoutOptimizationInvocations +=
        layoutOptimization.statistics.invocations;
    statistics->currentIRLayoutPBQPWork +=
        layoutOptimization.statistics.solverWork;
    statistics->currentIRLayoutMaterializationsBefore +=
        layoutOptimization.statistics.layoutMaterializationsBefore;
    statistics->currentIRLayoutMaterializationsAfter +=
        layoutOptimization.statistics.layoutMaterializationsAfter;
    statistics->currentIRLayoutMaterializationsErased +=
        layoutOptimization.statistics.unusedMaterializationsErased;
    statistics->currentIRLayoutMaterializationsReused +=
        layoutOptimization.statistics.sharedMaterializationsReused;
    statistics->currentIRLayoutHardOnlyInvocations +=
        layoutOptimization.statistics.hardOnlyInvocations;
  }
  if (!layoutOptimization.succeeded()) {
    CardExecutableCompilationStatus status =
        layoutOptimization.status == RepresentationPBQPStatus::NoSolution
            ? CardExecutableCompilationStatus::UnsupportedFailure
        : layoutOptimization.status == RepresentationPBQPStatus::Indeterminate
            ? CardExecutableCompilationStatus::IndeterminateFailure
            : CardExecutableCompilationStatus::CompilerFailure;
    return reportFailure(fail(status, "current-ir-layout",
                              layoutOptimization.detail.empty()
                                  ? "current-IR layout assignment failed"
                                  : layoutOptimization.detail));
  }
  if (wafer::support::getActiveCompileTimingSession())
    diagnostics << "wafer-compile: current-ir-layout"
                << " materializations_before="
                << layoutOptimization.statistics.layoutMaterializationsBefore
                << " materializations_after="
                << layoutOptimization.statistics.layoutMaterializationsAfter
                << " reused="
                << layoutOptimization.statistics.sharedMaterializationsReused
                << " erased="
                << layoutOptimization.statistics.unusedMaterializationsErased
                << " pbqp_work=" << layoutOptimization.statistics.solverWork
                << " soft_terms=disabled\n";

  std::vector<uint8_t> bufferizationFailures(projectedModules->size());
  std::vector<std::string> bufferizationDetails(projectedModules->size());
  std::vector<wafer::support::CompileIRInventory> bufferizedInventories(
      projectedModules->size());
  unsigned workers = runBoundedTilePipelines(
      context, projectedModules->size(),
      [&](size_t tileIndex) {
        TileModule &tile = (*projectedModules)[tileIndex];
        bufferizationFailures[tileIndex] =
            mlir::failed(bufferizeTileDataflowFunctionBoundaries(
                *tile.module, tile.materializationRelations,
                wafer::support::getActiveCompileTimingSession()
                    ? &bufferizedInventories[tileIndex]
                    : nullptr,
                bufferizationDetails[tileIndex]));
      },
      requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);
  if (llvm::is_contained(bufferizationFailures, uint8_t{1})) {
    auto failed = llvm::find(bufferizationFailures, uint8_t{1});
    const size_t index = static_cast<size_t>(
        std::distance(bufferizationFailures.begin(), failed));
    return reportFailure(
        fail(CardExecutableCompilationStatus::CompilerFailure,
             "tile-dataflow-bufferization",
             bufferizationDetails[index].empty()
                 ? "Tile dataflow function-boundary bufferization failed"
                 : bufferizationDetails[index]));
  }
  if (wafer::support::getActiveCompileTimingSession()) {
    wafer::support::CompileIRInventory aggregate;
    for (auto [tileIndex, inventory] : llvm::enumerate(bufferizedInventories)) {
      diagnostics << "wafer-compile: ir-inventory-tile "
                  << "stage=bufferized-tile-dataflow tile="
                  << expectedTileIds[tileIndex].getValue()
                  << " total_operations=" << inventory.getTotalOperations()
                  << '\n';
      aggregate.merge(inventory);
    }
    aggregate.print("bufferized-tile-dataflow", diagnostics);
  }

  std::vector<CandidateTileDataflowIR> tileDataflowViews;
  tileDataflowViews.reserve(projectedModules->size());
  for (TileModule &tile : *projectedModules)
    tileDataflowViews.push_back({tile.cardId, tile.tileId, &tile.module,
                                 &tile.materializationRelations});
  CardExecutablePreparationFailure preparationFailure;
  if (mlir::failed(preparation.prepareTileDataflow(tileDataflowViews,
                                                   preparationFailure)))
    return reportFailure(fail(preparationStatus(preparationFailure.kind),
                              "selected-tile-dataflow-preparation",
                              preparationFailure.detail.empty()
                                  ? "selected Tile dataflow preparation failed"
                                  : preparationFailure.detail));
  for (CandidateTileDataflowIR &tile : tileDataflowViews)
    if (!tile.getModule() || !tile.relations ||
        mlir::failed(checkStructuredBufferRelationsCurrent(
            tile.getModule().getOperation(), *tile.relations)) ||
        mlir::failed(mlir::verify(tile.getModule())))
      return reportFailure(
          fail(CardExecutableCompilationStatus::CompilerFailure,
               "selected-tile-dataflow-preparation",
               "selected Tile dataflow preparation produced stale relations "
               "or invalid IR"));
  if (wafer::support::getActiveCompileTimingSession()) {
    wafer::support::CompileIRInventory inventory;
    for (CandidateTileDataflowIR &tile : tileDataflowViews) {
      wafer::support::CompileIRInventory tileInventory;
      tileInventory.record(tile.getModule().getOperation());
      diagnostics << "wafer-compile: ir-inventory-tile "
                  << "stage=selected-tile-dataflow tile="
                  << tile.tile.getValue()
                  << " total_operations=" << tileInventory.getTotalOperations()
                  << '\n';
      uint64_t regionCount = 0;
      uint64_t regionBodyOperations = 0;
      uint64_t minimumRegionBodyOperations =
          std::numeric_limits<uint64_t>::max();
      uint64_t maximumRegionBodyOperations = 0;
      tile.getModule().walk([&](TileRegionOp region) {
        ++regionCount;
        uint64_t bodyOperations = 0;
        region.getBody().walk([&](mlir::Operation *) { ++bodyOperations; });
        regionBodyOperations += bodyOperations;
        minimumRegionBodyOperations =
            std::min(minimumRegionBodyOperations, bodyOperations);
        maximumRegionBodyOperations =
            std::max(maximumRegionBodyOperations, bodyOperations);
      });
      diagnostics << "wafer-compile: ir-region-inventory tile="
                  << tile.tile.getValue() << " regions=" << regionCount
                  << " body_operations=" << regionBodyOperations
                  << " min_body_operations="
                  << (regionCount == 0 ? 0 : minimumRegionBodyOperations)
                  << " max_body_operations=" << maximumRegionBodyOperations
                  << '\n';
      diagnostics
          << "wafer-compile: ir-provenance-tile tile=" << tile.tile.getValue()
          << " compute_emissions=" << tile.relations->operationEmissions.size()
          << " operand_buffers=" << tile.relations->operandBuffers.size()
          << " result_buffers=" << tile.relations->operationResultBuffers.size()
          << " scratch_buffers=" << tile.relations->scratchBuffers.size()
          << '\n';
      std::map<uint32_t, uint64_t> emissionsByNode;
      std::map<mlir::Operation *, std::set<uint32_t>> ownersByEmission;
      for (const StructuredOperationEmissionRelation &relation :
           tile.relations->operationEmissions) {
        ++emissionsByNode[relation.structuredNodeId];
        if (relation.operation)
          ownersByEmission[relation.operation].insert(
              relation.structuredNodeId);
      }
      std::map<uint64_t, uint64_t> emissionMultiplicity;
      for (const auto &[node, count] : emissionsByNode) {
        (void)node;
        ++emissionMultiplicity[count];
      }
      for (const auto &[emissions, nodes] : emissionMultiplicity)
        diagnostics << "wafer-compile: ir-provenance-multiplicity tile="
                    << tile.tile.getValue()
                    << " emissions_per_node=" << emissions
                    << " node_count=" << nodes << '\n';
      for (const auto &[operation, owners] : ownersByEmission)
        if (owners.size() > 1) {
          diagnostics << "wafer-compile: ir-provenance-shared-emission tile="
                      << tile.tile.getValue() << " op=" << operation->getName()
                      << " nodes=[";
          llvm::interleaveComma(owners, diagnostics);
          diagnostics << "]\n";
        }
      if (tile.tile == expectedTileIds.front()) {
        std::map<uint32_t, std::set<mlir::Operation *>> regionsByNode;
        std::map<uint32_t, std::map<std::string, uint64_t>> opsByNode;
        for (const StructuredOperationEmissionRelation &relation :
             tile.relations->operationEmissions) {
          if (!relation.operation)
            continue;
          TileRegionOp region =
              relation.operation->getParentOfType<TileRegionOp>();
          if (region)
            regionsByNode[relation.structuredNodeId].insert(
                region.getOperation());
          ++opsByNode[relation.structuredNodeId]
                     [relation.operation->getName().getStringRef().str()];
        }
        for (const auto &[node, operationCounts] : opsByNode) {
          diagnostics << "wafer-compile: ir-provenance-node tile="
                      << tile.tile.getValue() << " node=" << node
                      << " regions=" << regionsByNode[node].size()
                      << " emissions=" << emissionsByNode[node];
          for (const auto &[name, count] : operationCounts)
            diagnostics << " op." << name << '=' << count;
          diagnostics << '\n';
        }
      }
      inventory.merge(tileInventory);
    }
    inventory.print("selected-tile-dataflow", diagnostics);
  }

  // Tile IR inspection is an explicit request: production compilation,
  // candidate evaluation and the deterministic baseline never pay the
  // printer unless the caller asks for the inspection snapshot.
  std::vector<std::string> tileDataflowIRTrace;
  if (captureTileIRTrace) {
    tileDataflowIRTrace.reserve(projectedModules->size());
    for (const TileModule &tile : *projectedModules)
      tileDataflowIRTrace.push_back(captureTileIR(*tile.module));
  }

  std::vector<TileLoweringResult> loweringResults(projectedModules->size());
  auto convertTile = [&](size_t tileIndex) {
    TileModule &tile = (*projectedModules)[tileIndex];
    TileLoweringResult &result = loweringResults[tileIndex];
    result.module = std::move(tile.module);
    result.materializationRelations = std::move(tile.materializationRelations);
    if (mlir::failed(convertTileRegionsToInstructionIR(
            *result.module, result.materializationRelations, result.regionNodes,
            result.detail))) {
      result.conversionFailed = true;
      return;
    }
    mlir::FailureOr<unsigned> eliminated = cleanupCanonicalInstructionTransfers(
        *result.module, result.materializationRelations);
    if (mlir::failed(eliminated) ||
        containsTileDataflowOperations(result.module->getOperation()) ||
        mlir::failed(mlir::verify(*result.module))) {
      result.conversionFailed = true;
      if (result.detail.empty())
        result.detail =
            "TileRegion-to-Instr result is not canonical verifier-legal Instr "
            "IR";
      return;
    }
    result.redundantTransfersEliminated = *eliminated;
  };

  workers = runBoundedTilePipelines(context, projectedModules->size(),
                                    convertTile, requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);
  if (statistics)
    for (const TileLoweringResult &result : loweringResults)
      statistics->redundantFullBufferTransfersEliminated +=
          result.redundantTransfersEliminated;

  llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures;
  for (auto [tileIndex, result] : llvm::enumerate(loweringResults))
    if (result.conversionFailed || !result.module) {
      CardExecutableTileFailure failure;
      failure.tileId = expectedTileIds[tileIndex];
      failure.gate = "tile-region-to-instr";
      failure.detail = std::move(result.detail);
      failure.memoryPlanning = std::move(result.memoryPlanning);
      tileFailures.push_back(std::move(failure));
    }
  if (!tileFailures.empty()) {
    const std::string primaryDetail = tileFailures.front().detail;
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "tile-region-to-instr", primaryDetail, std::move(tileFailures)));
  }
  std::vector<CandidateInstructionIR> instructionViews;
  instructionViews.reserve(loweringResults.size());
  for (auto [index, result] : llvm::enumerate(loweringResults))
    instructionViews.push_back(
        {expectedCardId, expectedTileIds[index], &result.module,
         &result.materializationRelations, &result.regionNodes});
  preparationFailure = {};
  if (mlir::failed(preparation.prepareInstructionIR(instructionViews,
                                                    preparationFailure)))
    return reportFailure(fail(preparationStatus(preparationFailure.kind),
                              "selected-instruction-preparation",
                              preparationFailure.detail.empty()
                                  ? "selected instruction preparation failed"
                                  : preparationFailure.detail));
  for (CandidateInstructionIR &tile : instructionViews)
    tile.getModule().walk([](InstrGatherScatterOp operation) {
      if (operation.getCardDdrResourceAttr())
        operation.setCardDdrResourceAttr(CardDDRResourceAttr{});
    });

  if (!preparation.ownsInstructionCompletion()) {
    std::vector<uint8_t> completionFailures(instructionViews.size());
    workers = runBoundedTilePipelines(
        context, instructionViews.size(),
        [&](size_t tileIndex) {
          CandidateInstructionIR &tile = instructionViews[tileIndex];
          completionFailures[tileIndex] =
              !tile.getModule() || !tile.relations ||
              mlir::failed(wafer::rebuildRequiredNCCJoins(tile.getModule())) ||
              mlir::failed(checkStructuredBufferRelationsCurrent(
                  tile.getModule().getOperation(), *tile.relations)) ||
              mlir::failed(mlir::verify(tile.getModule()));
        },
        requestedWorkers);
    if (statistics)
      statistics->maximumTilePipelineWorkers =
          std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);
    if (llvm::is_contained(completionFailures, uint8_t{1}))
      return reportFailure(
          fail(CardExecutableCompilationStatus::CompilerFailure,
               "current-instr-completion",
               "fresh required NCC completion construction failed"));
  }

  for (CandidateInstructionIR &tile : instructionViews)
    if (!tile.getModule() || !tile.relations ||
        mlir::failed(checkStructuredBufferRelationsCurrent(
            tile.getModule().getOperation(), *tile.relations)) ||
        mlir::failed(mlir::verify(tile.getModule())))
      return reportFailure(
          fail(CardExecutableCompilationStatus::CompilerFailure,
               "selected-instruction-preparation",
               "selected instruction preparation produced stale relations "
               "or invalid IR"));
  if (wafer::support::getActiveCompileTimingSession()) {
    wafer::support::CompileIRInventory instructionInventory;
    for (auto [tileIndex, lowering] : llvm::enumerate(loweringResults)) {
      lowering.instructionInventory.record(lowering.module->getOperation());
      diagnostics << "wafer-compile: ir-inventory-tile stage=instruction tile="
                  << expectedTileIds[tileIndex].getValue()
                  << " total_operations="
                  << lowering.instructionInventory.getTotalOperations() << '\n';
      instructionInventory.merge(lowering.instructionInventory);
    }
    instructionInventory.print("instruction", diagnostics);
  }
  std::vector<CanonicalInstructionTile> canonicalTiles;
  canonicalTiles.reserve(loweringResults.size());
  for (auto [index, lowering] : llvm::enumerate(loweringResults))
    canonicalTiles.push_back({expectedCardId, expectedTileIds[index],
                              std::move(lowering.module),
                              std::move(lowering.materializationRelations)});
  CardExecutableCompilationResult result =
      compileCanonicalInstructionTilesToExecutable(
          std::move(canonicalTiles), expectedCardId, expectedTileIds, program,
          executionConfig, diagnostics, programData, statistics,
          tilePipelineParallelism);
  if (!result.isAccepted())
    return reportFailure(std::move(result));
  result.tileDataflowIRTrace = std::move(tileDataflowIRTrace);
  diagnostics << "wafer-compile: card-executable-compilation outcome=accepted"
              << " physical_tiles=" << result.executable->tiles.size() << '\n';
  return result;
}

} // namespace wafer::compiler::detail
