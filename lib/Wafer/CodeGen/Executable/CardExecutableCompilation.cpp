//===- CardExecutableCompilation.cpp - Policy-free executable seam ------===//

#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"
#include "Wafer/Analysis/Structured/StructuredNodeUseIndex.h"
#include "Wafer/Driver/CompilationInternal.h"

#include "Wafer/CodeGen/Executable/BoundedTileExecutor.h"

#include "Wafer/Conversion/WaferCardModuleToTileModules/WaferCardModuleToTileModules.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace wafer::compiler::detail {
namespace {

struct TileLoweringResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
  std::string detail;
  TileMemoryPlanningFailure memoryPlanning;
  SelectedBufferMaterializationFailure selectedBuffer;
  unsigned rotatingSlotAllocationCount = 0;
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

static mlir::LogicalResult
lowerTileRegionsToInstructionIR(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations,
                                std::string &detail) {
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "CardModule splitting produced buffer relations outside the "
             "current IR";
    return mlir::failure();
  }

  llvm::SmallVector<TileRegionOp, 4> regions;
  module.walk([&](TileRegionOp region) { regions.push_back(region); });
  TileRegionToInstrLoweringSession loweringSession(*module.getContext());
  StructuredBufferReplacementListener replacementListener(relations);
  for (TileRegionOp region : regions)
    if (mlir::failed(convertTileRegionToInstr(region, loweringSession,
                                              &replacementListener))) {
      detail = "TileRegion-to-Instr conversion failed";
      return mlir::failure();
    }

  if (!replacementListener.finalizeAfterRewrite() ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "TileRegion-to-Instr lowering did not preserve every structured "
             "buffer relation in the current Tile IR";
    return mlir::failure();
  }

  if (mlir::failed(
          runPassPipeline(module, "required-ncc-join-placement",
                          [](mlir::OpPassManager &manager) {
                            manager.nest<mlir::func::FuncOp>().addPass(
                                wafer::createPlaceRequiredNCCJoinsPass());
                          }))) {
    detail = "function-level required NCC join placement failed";
    return mlir::failure();
  }
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "required NCC join placement changed the Tile buffer "
             "relation domain";
    return mlir::failure();
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

namespace {

static CardExecutableCompilationResult
fail(CardExecutableCompilationStatus status, llvm::StringRef gate,
     llvm::StringRef detail,
     llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures = {},
     uint64_t rotatingSlotAllocationsMaterialized = 0) {
  CardExecutableCompilationResult result;
  result.status = status;
  result.gate = gate.str();
  result.detail = detail.str();
  result.tileFailures = std::move(tileFailures);
  result.rotatingSlotAllocationsMaterialized =
      rotatingSlotAllocationsMaterialized;
  return result;
}

} // namespace

CardExecutableCompilationResult compileCardModuleToExecutable(
    mlir::OwningOpRef<mlir::ModuleOp> cardModule, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    llvm::ArrayRef<llvm::SmallVector<SelectedBufferingScope, 4>>
        selectedBufferingScopes,
    const StructuredMaterializationRelations &materializationRelations,
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
    diagnostics << "wafer-compile: card-executable-compilation outcome="
                << (result.isProvenExactRejection() ? "exact-rejection"
                                                    : "indeterminate")
                << " gate=" << result.gate << " detail=" << result.detail
                << '\n';
    return result;
  };

  if (!cardModule || expectedTileIds.empty() ||
      (!selectedBufferingScopes.empty() &&
       selectedBufferingScopes.size() != expectedTileIds.size()))
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "compilation-contract",
             "CardModule, Tile domain or selected-buffer request domain "
             "is incomplete"));

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

  // Tile IR inspection is an explicit request: production compilation,
  // candidate evaluation and the deterministic baseline never pay the
  // printer unless the caller asks for the inspection snapshot.
  std::vector<std::string> tileDataflowIRTrace;
  if (captureTileIRTrace) {
    tileDataflowIRTrace.reserve(projectedModules->size());
    for (auto [index, tile] : llvm::enumerate(*projectedModules)) {
      if (tile.cardId != expectedCardId ||
          tile.tileId != expectedTileIds[index])
        return reportFailure(
            fail(CardExecutableCompilationStatus::IndeterminateFailure,
                 "card-module-to-tile-modules",
                 "per-Tile modules changed the selected Tile domain"));
      tileDataflowIRTrace.push_back(captureTileIR(*tile.module));
    }
  } else {
    for (auto [index, tile] : llvm::enumerate(*projectedModules))
      if (tile.cardId != expectedCardId ||
          tile.tileId != expectedTileIds[index])
        return reportFailure(
            fail(CardExecutableCompilationStatus::IndeterminateFailure,
                 "card-module-to-tile-modules",
                 "per-Tile modules changed the selected Tile domain"));
  }

  std::vector<TileLoweringResult> loweringResults(projectedModules->size());
  auto lowerTile = [&](size_t tileIndex) {
    TileModule &tile = (*projectedModules)[tileIndex];
    TileLoweringResult &result = loweringResults[tileIndex];
    result.module = std::move(tile.module);
    result.materializationRelations = std::move(tile.materializationRelations);
    if (mlir::failed(lowerTileRegionsToInstructionIR(
            *result.module, result.materializationRelations, result.detail)) ||
        containsTileDataflowOperations(result.module->getOperation()) ||
        mlir::failed(mlir::verify(*result.module))) {
      result.conversionFailed = true;
      if (result.detail.empty())
        result.detail =
            "TileRegion-to-Instr result is not canonical verifier-legal Instr "
            "IR";
      return;
    }

    llvm::ArrayRef<SelectedBufferingScope> bufferingScopes;
    if (!selectedBufferingScopes.empty())
      bufferingScopes = selectedBufferingScopes[tileIndex];
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> memoryPlanned =
        planTileMemory(std::move(result.module), &result.memoryPlanning,
                       bufferingScopes, &result.materializationRelations,
                       &result.rotatingSlotAllocationCount,
                       &result.selectedBuffer);
    if (mlir::failed(memoryPlanned)) {
      result.memoryPlanningFailed = true;
      result.detail = result.selectedBuffer.detail;
      if (result.detail.empty())
        result.detail = "Tile memory planning failed";
      return;
    }
    result.module = std::move(*memoryPlanned);
  };

  const unsigned requestedWorkers = tilePipelineParallelism == 0
                                        ? kMaximumBoundedTilePipelineWorkers
                                        : tilePipelineParallelism;
  const unsigned workers = runBoundedTilePipelines(
      context, projectedModules->size(), lowerTile, requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);

  llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> instructionModules;
  std::vector<StructuredMaterializationRelations> tileRelations;
  instructionModules.reserve(loweringResults.size());
  tileRelations.reserve(loweringResults.size());
  uint64_t rotatingSlotAllocationsMaterialized = 0;
  bool allFailuresAreExact = true;
  for (auto [tileIndex, result] : llvm::enumerate(loweringResults)) {
    rotatingSlotAllocationsMaterialized += result.rotatingSlotAllocationCount;
    if (result.conversionFailed || result.memoryPlanningFailed ||
        !result.module) {
      CardExecutableTileFailure failure;
      failure.tileId = expectedTileIds[tileIndex];
      failure.gate =
          result.conversionFailed ? "tile-region-to-instr"
          : result.memoryPlanning.kind ==
                  TileMemoryPlanningFailureKind::SPMAllocation
              ? "spm-allocation"
          : result.memoryPlanning.kind ==
                  TileMemoryPlanningFailureKind::SelectedBufferMaterialization
              ? "selected-buffer-materialization"
              : "tile-memory-planning";
      failure.detail = std::move(result.detail);
      failure.memoryPlanning = std::move(result.memoryPlanning);
      failure.selectedBuffer = std::move(result.selectedBuffer);
      allFailuresAreExact &=
          !result.conversionFailed &&
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
             primaryGate, primaryDetail, std::move(tileFailures),
             rotatingSlotAllocationsMaterialized));
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
                                            : loweringFailure.detail,
             {}, rotatingSlotAllocationsMaterialized));

  if (executable->tiles.size() != expectedTileIds.size())
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "card-executable-domain", "Tile executable domain is incomplete",
             {}, rotatingSlotAllocationsMaterialized));
  for (auto [index, tile] : llvm::enumerate(executable->tiles))
    if (tile.getCardId() != expectedCardId ||
        tile.getTileId() != expectedTileIds[index])
      return reportFailure(
          fail(CardExecutableCompilationStatus::IndeterminateFailure,
               "card-executable-domain",
               "executable lowering changed the Tile identity", {},
               rotatingSlotAllocationsMaterialized));

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
  result.rotatingSlotAllocationsMaterialized =
      rotatingSlotAllocationsMaterialized;
  diagnostics << "wafer-compile: card-executable-compilation outcome=accepted"
              << " physical_tiles=" << result.executable->tiles.size()
              << " rotating_slot_allocations="
              << rotatingSlotAllocationsMaterialized << '\n';
  return result;
}

} // namespace wafer::compiler::detail
