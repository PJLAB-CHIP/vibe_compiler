//===- CardExecutableCompilation.cpp - Policy-free executable seam ------===//

#include "CardExecutableCompilation.h"
#include "CompilationInternal.h"
#include "StructuredBufferRelations.h"

#include "BoundedTileExecutor.h"

#include "Wafer/Conversion/WaferCardProgramToTileModules/WaferCardProgramToTileModules.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace wafer::compiler::detail {
namespace {

struct PhysicalTileLoweringResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
  std::string detail;
  PhysicalTileMemoryPlanningFailure memoryPlanning;
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
    detail = "CardProgram splitting produced buffer relations outside the "
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

  if (!replacementListener.preservedAllRelations() ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "TileRegion-to-Instr lowering did not preserve every structured "
             "buffer relation in the current physical Tile IR";
    return mlir::failure();
  }

  if (mlir::failed(runPassPipeline(module, "required-ncc-join-placement",
                                   [](mlir::OpPassManager &manager) {
                                     wafer::addRequiredNCCJoinPlacementPass(
                                         manager.nest<mlir::func::FuncOp>());
                                   }))) {
    detail = "function-level required NCC join placement failed";
    return mlir::failure();
  }
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module.getOperation(),
                                                         relations))) {
    detail = "required NCC join placement changed the physical Tile buffer "
             "relation domain";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace

bool isProvenExactPhysicalTileMemoryPlanningFailure(
    const PhysicalTileMemoryPlanningFailure &failure) {
  return failure.kind == PhysicalTileMemoryPlanningFailureKind::SPMAllocation &&
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

CardExecutableCompilationResult compileCardProgramToExecutable(
    mlir::OwningOpRef<mlir::ModuleOp> cardProgram,
    PhysicalCardId expectedCardId,
    llvm::ArrayRef<PhysicalTileId> expectedTileIds,
    llvm::ArrayRef<llvm::SmallVector<SelectedBufferRequest, 4>>
        selectedBufferRequests,
    const StructuredMaterializationRelations &materializationRelations,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeCardSynthesisStatistics *statistics,
    unsigned tilePipelineParallelism) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "stage", "card-program-to-executable", "card-executable-compilation");
  if (statistics)
    ++statistics->cardProgramCompilationInvocations;

  auto reportFailure = [&](CardExecutableCompilationResult result) {
    totalTiming.markFailed();
    diagnostics << "wafer-compile: card-executable-compilation outcome="
                << (result.isProvenExactRejection() ? "exact-rejection"
                                                    : "indeterminate")
                << " gate=" << result.gate << " detail=" << result.detail
                << '\n';
    return result;
  };

  if (!cardProgram || expectedTileIds.empty() ||
      (!selectedBufferRequests.empty() &&
       selectedBufferRequests.size() != expectedTileIds.size()))
    return reportFailure(fail(
        CardExecutableCompilationStatus::IndeterminateFailure,
        "compilation-contract",
        "CardProgram, physical Tile domain or selected-buffer request domain "
        "is incomplete"));

  std::string detail;
  mlir::FailureOr<llvm::SmallVector<PhysicalTileModule, 16>> tileModules;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "conversion", "card-program-to-executable",
        "card-program-to-physical-tiles");
    tileModules = splitCardProgramIntoPhysicalTileModules(
        *cardProgram, &detail, &materializationRelations);
  }
  if (mlir::failed(tileModules))
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "card-program-to-tile-modules", detail));
  if (tileModules->size() != expectedTileIds.size())
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "card-program-to-tile-modules",
             "CardProgram does not contain the expected physical Tile domain"));

  std::vector<std::string> tileDataflowIRTrace;
  tileDataflowIRTrace.reserve(tileModules->size());
  for (auto [index, tile] : llvm::enumerate(*tileModules)) {
    if (tile.cardId != expectedCardId || tile.tileId != expectedTileIds[index])
      return reportFailure(
          fail(CardExecutableCompilationStatus::IndeterminateFailure,
               "card-program-to-tile-modules",
               "per-Tile modules changed the selected physical Tile domain"));
    tileDataflowIRTrace.push_back(captureTileIR(*tile.module));
  }

  std::vector<PhysicalTileLoweringResult> loweringResults(tileModules->size());
  auto lowerTile = [&](size_t tileIndex) {
    PhysicalTileModule &tile = (*tileModules)[tileIndex];
    PhysicalTileLoweringResult &result = loweringResults[tileIndex];
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

    llvm::ArrayRef<SelectedBufferRequest> requests;
    if (!selectedBufferRequests.empty())
      requests = selectedBufferRequests[tileIndex];
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> memoryPlanned =
        planPhysicalTileMemory(std::move(result.module), &result.memoryPlanning,
                               requests, &result.materializationRelations,
                               &result.rotatingSlotAllocationCount,
                               &result.selectedBuffer);
    if (mlir::failed(memoryPlanned)) {
      result.memoryPlanningFailed = true;
      result.detail = result.selectedBuffer.detail;
      if (result.detail.empty())
        result.detail = "physical Tile memory planning failed";
      return;
    }
    result.module = std::move(*memoryPlanned);
  };

  const unsigned requestedWorkers = tilePipelineParallelism == 0
                                        ? kMaximumBoundedTilePipelineWorkers
                                        : tilePipelineParallelism;
  const unsigned workers =
      runBoundedTilePipelines(cardProgram->getContext(), tileModules->size(),
                              lowerTile, requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);

  llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> physicalTileModules;
  std::vector<StructuredMaterializationRelations> physicalTileRelations;
  physicalTileModules.reserve(loweringResults.size());
  physicalTileRelations.reserve(loweringResults.size());
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
                  PhysicalTileMemoryPlanningFailureKind::SPMAllocation
              ? "spm-allocation"
          : result.memoryPlanning.kind ==
                  PhysicalTileMemoryPlanningFailureKind::
                      SelectedBufferMaterialization
              ? "selected-buffer-materialization"
              : "physical-tile-memory-planning";
      failure.detail = std::move(result.detail);
      failure.memoryPlanning = std::move(result.memoryPlanning);
      failure.selectedBuffer = std::move(result.selectedBuffer);
      allFailuresAreExact &= !result.conversionFailed &&
                             isProvenExactPhysicalTileMemoryPlanningFailure(
                                 failure.memoryPlanning);
      tileFailures.push_back(std::move(failure));
      continue;
    }
    physicalTileModules.push_back(std::move(result.module));
    physicalTileRelations.push_back(std::move(result.materializationRelations));
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

  WholeCardExecutableLoweringFailure loweringFailure;
  mlir::FailureOr<WholeCardExecutable> executable =
      lowerPhysicalTileModulesToExecutable(
          std::move(physicalTileModules), program, executionConfig, diagnostics,
          loweringFailure, statistics, tilePipelineParallelism);
  if (mlir::failed(executable))
    return reportFailure(fail(
        loweringFailure.isProvenExactRejection()
            ? CardExecutableCompilationStatus::ProvenExactRejection
            : CardExecutableCompilationStatus::IndeterminateFailure,
        loweringFailure.getDiagnosticLabel(),
        loweringFailure.detail.empty() ? "physical-Tile module lowering failed"
                                       : loweringFailure.detail,
        {}, rotatingSlotAllocationsMaterialized));

  if (executable->tiles.size() != expectedTileIds.size())
    return reportFailure(
        fail(CardExecutableCompilationStatus::IndeterminateFailure,
             "whole-card-executable-domain",
             "physical Tile executable domain is incomplete", {},
             rotatingSlotAllocationsMaterialized));
  for (auto [index, tile] : llvm::enumerate(executable->tiles))
    if (tile.getPhysicalCardId() != expectedCardId ||
        tile.getPhysicalTileId() != expectedTileIds[index])
      return reportFailure(
          fail(CardExecutableCompilationStatus::IndeterminateFailure,
               "whole-card-executable-domain",
               "executable lowering changed the physical Tile identity", {},
               rotatingSlotAllocationsMaterialized));

  CardExecutableCompilationResult result;
  result.status = CardExecutableCompilationStatus::Accepted;
  result.executable.emplace(std::move(*executable));
  result.tileDataflowIRTrace = std::move(tileDataflowIRTrace);
  for (auto [tileIndex, tile] : llvm::enumerate(result.executable->tiles)) {
    const StructuredMaterializationRelations &relations =
        physicalTileRelations[tileIndex];
    tile.getModule().walk([&](mlir::Operation *operation) {
      for (uint32_t node :
           collectStructuredNodesUsedByOperation(operation, relations))
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
