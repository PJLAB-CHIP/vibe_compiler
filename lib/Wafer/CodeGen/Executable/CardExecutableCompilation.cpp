//===- CardExecutableCompilation.cpp - Policy-free executable seam ------===//

#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"
#include "Wafer/Analysis/Structured/StructuredNodeUseIndex.h"
#include "Wafer/Driver/CompilationInternal.h"

#include "Wafer/CodeGen/Executable/BoundedTileExecutor.h"

#include "Wafer/Conversion/WaferCardModuleToTileModules/WaferCardModuleToTileModules.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/MemoryPlanningPipelines.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

struct TileLoweringResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
  std::vector<CandidateInstructionIR::RegionNodeRelation> regionNodes;
  wafer::support::CompileIRInventory bufferizedInventory;
  wafer::support::CompileIRInventory instructionInventory;
  std::string detail;
  TileMemoryPlanningFailure memoryPlanning;
  bool conversionFailed = false;
  bool memoryPlanningFailed = false;
};

static std::string captureTileIR(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

static mlir::LogicalResult lowerTileRegionsToInstructionIR(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    std::vector<CandidateInstructionIR::RegionNodeRelation> &regionNodes,
    bool placeCanonicalCompletion,
    wafer::support::CompileIRInventory *bufferizedInventory,
    wafer::support::CompileIRInventory *instructionInventory,
    std::string &detail) {
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "CardModule splitting produced buffer relations outside the "
             "current IR";
    return mlir::failure();
  }
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
  if (mlir::failed(rebaseStructuredBufferRelationsToStorageRoots(relations))) {
    detail = "Instr function-boundary bufferization cannot preserve a unique "
             "storage root for every selected relation";
    return mlir::failure();
  }
  if (mlir::failed(
          runPassPipeline(module, "instr-function-boundary-bufferization",
                          wafer::buildBufferizeInstrFunctionsPipeline))) {
    detail = "Instr function-boundary bufferization failed";
    return mlir::failure();
  }
  retainCurrentStructuredBufferRelations(module.getOperation(), relations);
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "Instr function-boundary bufferization changed a selected "
             "Tile buffer relation";
    return mlir::failure();
  }
  if (bufferizedInventory)
    bufferizedInventory->record(module.getOperation());
  StructuredBufferReplacementListener replacementListener(relations);
  TileRegionToInstrLoweringSession loweringSession(*module.getContext(),
                                                   &replacementListener);
  for (TileRegionOp region : regions)
    if (mlir::failed(convertTileRegionToInstr(region, loweringSession,
                                              &replacementListener))) {
      detail = "TileRegion-to-Instr conversion failed";
      return mlir::failure();
    }

  if (!replacementListener.finalizeAfterRewrite() ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = replacementListener.getFailureReason().empty()
                 ? "TileRegion-to-Instr lowering did not preserve every "
                   "structured buffer relation in the current Tile IR"
                 : replacementListener.getFailureReason().str();
    return mlir::failure();
  }

  if (placeCanonicalCompletion) {
    if (mlir::failed(
            runPassPipeline(module, "required-ncc-join-placement",
                            [](mlir::OpPassManager &manager) {
                              manager.nest<mlir::func::FuncOp>().addPass(
                                  wafer::createPlaceRequiredNCCJoinsPass());
                            }))) {
      detail = "function-level required NCC join placement failed";
      return mlir::failure();
    }
    if (mlir::failed(checkStructuredBufferRelationsCurrent(
            module.getOperation(), relations))) {
      detail = "required NCC join placement changed the Tile buffer "
               "relation domain";
      return mlir::failure();
    }
  }
  if (instructionInventory)
    instructionInventory->record(module.getOperation());
  return mlir::success();
}

} // namespace

bool isProvenExactTileMemoryPlanningFailure(
    const TileMemoryPlanningFailure &failure) {
  return failure.kind == TileMemoryPlanningFailureKind::SPMAllocation &&
         failure.spmPlanningFailureKind ==
             SPMMemoryPlanningFailureKind::CapacityOverflow;
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

} // namespace

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
  std::string detail;
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
                  << tile.tile.getValue() << " total_operations="
                  << tileInventory.getTotalOperations() << '\n';
      diagnostics << "wafer-compile: ir-provenance-tile tile="
                  << tile.tile.getValue() << " compute_emissions="
                  << tile.relations->operationEmissions.size()
                  << " operand_buffers=" << tile.relations->operandBuffers.size()
                  << " result_buffers="
                  << tile.relations->operationResultBuffers.size()
                  << " scratch_buffers="
                  << tile.relations->scratchBuffers.size() << '\n';
      std::map<uint32_t, uint64_t> emissionsByNode;
      for (const StructuredOperationEmissionRelation &relation :
           tile.relations->operationEmissions)
        ++emissionsByNode[relation.structuredNodeId];
      std::map<uint64_t, uint64_t> emissionMultiplicity;
      for (const auto &[node, count] : emissionsByNode) {
        (void)node;
        ++emissionMultiplicity[count];
      }
      for (const auto &[emissions, nodes] : emissionMultiplicity)
        diagnostics << "wafer-compile: ir-provenance-multiplicity tile="
                    << tile.tile.getValue() << " emissions_per_node="
                    << emissions << " node_count=" << nodes << '\n';
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
    if (mlir::failed(lowerTileRegionsToInstructionIR(
            *result.module, result.materializationRelations, result.regionNodes,
            /*placeCanonicalCompletion=*/
            !preparation.ownsInstructionCompletion(),
            wafer::support::getActiveCompileTimingSession()
                ? &result.bufferizedInventory
                : nullptr,
            wafer::support::getActiveCompileTimingSession()
                ? &result.instructionInventory
                : nullptr,
            result.detail)) ||
        containsTileDataflowOperations(result.module->getOperation()) ||
        mlir::failed(mlir::verify(*result.module))) {
      result.conversionFailed = true;
      if (result.detail.empty())
        result.detail =
            "TileRegion-to-Instr result is not canonical verifier-legal Instr "
            "IR";
    }
  };

  const unsigned requestedWorkers = tilePipelineParallelism == 0
                                        ? kMaximumBoundedTilePipelineWorkers
                                        : tilePipelineParallelism;
  unsigned workers = runBoundedTilePipelines(context, projectedModules->size(),
                                             convertTile, requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);

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
  if (wafer::support::getActiveCompileTimingSession()) {
    wafer::support::CompileIRInventory bufferizedInventory;
    wafer::support::CompileIRInventory instructionInventory;
    for (auto [tileIndex, result] : llvm::enumerate(loweringResults)) {
      diagnostics << "wafer-compile: ir-inventory-tile "
                  << "stage=bufferized-tile-dataflow tile="
                  << expectedTileIds[tileIndex].getValue()
                  << " total_operations="
                  << result.bufferizedInventory.getTotalOperations() << '\n';
      diagnostics << "wafer-compile: ir-inventory-tile stage=instruction tile="
                  << expectedTileIds[tileIndex].getValue()
                  << " total_operations="
                  << result.instructionInventory.getTotalOperations() << '\n';
      bufferizedInventory.merge(result.bufferizedInventory);
      instructionInventory.merge(result.instructionInventory);
    }
    bufferizedInventory.print("bufferized-tile-dataflow", diagnostics);
    instructionInventory.print("instruction", diagnostics);
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
  workers = runBoundedTilePipelines(context, loweringResults.size(), planMemory,
                                    requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> instructionModules;
  std::vector<StructuredMaterializationRelations> tileRelations;
  instructionModules.reserve(loweringResults.size());
  tileRelations.reserve(loweringResults.size());
  bool allFailuresAreExact = true;
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
      allFailuresAreExact &=
          isProvenExactTileMemoryPlanningFailure(failure.memoryPlanning);
      tileFailures.push_back(std::move(failure));
      continue;
    }
    instructionModules.push_back(std::move(result.module));
    tileRelations.push_back(std::move(result.materializationRelations));
  }
  if (!tileFailures.empty()) {
    const std::string primaryGate = tileFailures.front().gate;
    const std::string primaryDetail = tileFailures.front().detail;
    return reportFailure(
        fail(allFailuresAreExact
                 ? CardExecutableCompilationStatus::ProvenExactRejection
                 : CardExecutableCompilationStatus::IndeterminateFailure,
             primaryGate, primaryDetail, std::move(tileFailures)));
  }

  CardExecutableLoweringFailure loweringFailure;
  mlir::FailureOr<CardExecutableLoweringResult> executable =
      lowerTileModulesToCardExecutable(
          std::move(instructionModules), program, executionConfig, diagnostics,
          loweringFailure, programData, statistics, tilePipelineParallelism);
  if (mlir::failed(executable))
    return reportFailure(
        fail(loweringFailure.isProvenExactRejection()
                 ? CardExecutableCompilationStatus::ProvenExactRejection
                 : CardExecutableCompilationStatus::IndeterminateFailure,
             loweringFailure.getDiagnosticLabel(),
             loweringFailure.detail.empty() ? "Tile module lowering failed"
                                            : loweringFailure.detail));

  if (executable->tiles.size() != expectedTileIds.size())
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "card-executable-domain", "Tile executable domain is incomplete"));
  for (auto [index, tile] : llvm::enumerate(executable->tiles))
    if (tile.getCardId() != expectedCardId ||
        tile.getTileId() != expectedTileIds[index])
      return reportFailure(
          fail(CardExecutableCompilationStatus::IndeterminateFailure,
               "card-executable-domain",
               "executable lowering changed the Tile identity"));

  CardExecutableCompilationResult result;
  result.status = CardExecutableCompilationStatus::Accepted;
  result.executable.emplace(std::move(*executable));
  result.tileDataflowIRTrace = std::move(tileDataflowIRTrace);
  for (auto [tileIndex, tile] : llvm::enumerate(result.executable->tiles)) {
    const StructuredMaterializationRelations &relations =
        tileRelations[tileIndex];
    StructuredNodeUseIndex nodeUses(relations);
    tile.getModule().walk([&](mlir::Operation *operation) {
      for (uint32_t node : nodeUses.collectNodesUsedBy(operation))
        result.operationNodeRelations.push_back({operation, node});
    });
  }
  diagnostics << "wafer-compile: card-executable-compilation outcome=accepted"
              << " physical_tiles=" << result.executable->tiles.size() << '\n';
  return result;
}

} // namespace wafer::compiler::detail
