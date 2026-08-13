//===- CardExecutableCompilation.cpp - Policy-free executable seam ------===//

#include "CardExecutableCompilation.h"

#include "BoundedTileExecutor.h"

#include "Wafer/Conversion/WaferCardProgramToTileModules/WaferCardProgramToTileModules.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Support/CompileTiming.h"

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
  std::string detail;
  PhysicalTileFinalizationFailure finalization;
  SelectedBufferMaterializationFailure selectedBuffer;
  unsigned rotatingSlotAllocationCount = 0;
  bool conversionFailed = false;
  bool finalizationFailed = false;
};

static std::shared_ptr<const std::string> captureTileIR(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return std::make_shared<const std::string>(std::move(text));
}

static bool
isExactFinalizationFailure(const PhysicalTileFinalizationFailure &failure) {
  switch (failure.kind) {
  case PhysicalTileFinalizationFailureKind::Contract:
  case PhysicalTileFinalizationFailureKind::PrematureWholeCardFacts:
  case PhysicalTileFinalizationFailureKind::Completion:
  case PhysicalTileFinalizationFailureKind::Verification:
  case PhysicalTileFinalizationFailureKind::FunctionBoundaryBufferization:
  case PhysicalTileFinalizationFailureKind::SelectedBufferMaterialization:
    return true;
  case PhysicalTileFinalizationFailureKind::SPMAllocation:
    return failure.spmPlanningFailureKind ==
               SPMMemoryPlanningFailureKind::CapacityOverflow ||
           failure.spmPlanningFailureKind ==
               SPMMemoryPlanningFailureKind::UnsupportedLifetime;
  case PhysicalTileFinalizationFailureKind::None:
    return false;
  }
  return false;
}

static bool isExactAdmissionFailure(llvm::StringRef gate) {
  return gate == "physical-tile-domain" || gate == "physical-tile-verifier" ||
         gate == "direct-dte" || gate == "whole-card-resources" ||
         gate == "accepted-physical-tile-verifier" ||
         gate == "accepted-call-closure" ||
         gate == "partition-resource-projection" ||
         gate == "runtime-launch-contract" || gate == "target-abi-preparation";
}

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
  mlir::FailureOr<llvm::SmallVector<ProjectedPhysicalTileModule, 16>> projected;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "conversion", "card-program-to-executable",
        "card-program-to-physical-tiles");
    projected = projectCardProgramToPhysicalTileModules(*cardProgram, &detail);
  }
  if (mlir::failed(projected))
    return reportFailure(
        fail(CardExecutableCompilationStatus::ProvenExactRejection,
             "physical-tile-projection", detail));
  if (projected->size() != expectedTileIds.size())
    return reportFailure(fail(
        CardExecutableCompilationStatus::ProvenExactRejection,
        "physical-tile-projection", "physical Tile projection is incomplete"));

  std::vector<std::shared_ptr<const std::string>> selectedTileIR;
  selectedTileIR.reserve(projected->size());
  for (auto [index, tile] : llvm::enumerate(*projected)) {
    if (tile.cardId != expectedCardId || tile.tileId != expectedTileIds[index])
      return reportFailure(fail(
          CardExecutableCompilationStatus::ProvenExactRejection,
          "physical-tile-projection",
          "CardProgram projection changed the selected physical Tile domain"));
    selectedTileIR.push_back(captureTileIR(*tile.module));
  }

  std::vector<PhysicalTileLoweringResult> loweringResults(projected->size());
  auto lowerTile = [&](size_t tileIndex) {
    ProjectedPhysicalTileModule &tile = (*projected)[tileIndex];
    PhysicalTileLoweringResult &result = loweringResults[tileIndex];
    result.module = std::move(tile.module);
    if (mlir::failed(
            convertTileRegionToInstrModule(*result.module, &result.detail)) ||
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
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> finalized =
        finalizePhysicalTileModule(
            std::move(result.module), &result.finalization, requests,
            &result.rotatingSlotAllocationCount, &result.selectedBuffer);
    if (mlir::failed(finalized)) {
      result.finalizationFailed = true;
      result.detail = result.selectedBuffer.detail;
      if (result.detail.empty())
        result.detail = "physical Tile finalization failed";
      return;
    }
    result.module = std::move(*finalized);
  };

  const unsigned requestedWorkers = tilePipelineParallelism == 0
                                        ? kMaximumBoundedTilePipelineWorkers
                                        : tilePipelineParallelism;
  const unsigned workers =
      runBoundedTilePipelines(cardProgram->getContext(), projected->size(),
                              lowerTile, requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);

  llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> physicalTileModules;
  physicalTileModules.reserve(loweringResults.size());
  uint64_t rotatingSlotAllocationsMaterialized = 0;
  bool allFailuresAreExact = true;
  for (auto [tileIndex, result] : llvm::enumerate(loweringResults)) {
    rotatingSlotAllocationsMaterialized += result.rotatingSlotAllocationCount;
    if (result.conversionFailed || result.finalizationFailed ||
        !result.module) {
      CardExecutableTileFailure failure;
      failure.tileId = expectedTileIds[tileIndex];
      failure.gate = result.conversionFailed ? "tile-region-to-instr"
                     : result.finalization.kind ==
                             PhysicalTileFinalizationFailureKind::SPMAllocation
                         ? "spm-allocation"
                     : result.finalization.kind ==
                             PhysicalTileFinalizationFailureKind::
                                 SelectedBufferMaterialization
                         ? "selected-buffer-materialization"
                         : "physical-tile-finalization";
      failure.detail = std::move(result.detail);
      failure.finalization = std::move(result.finalization);
      failure.selectedBuffer = std::move(result.selectedBuffer);
      allFailuresAreExact &= result.conversionFailed ||
                             isExactFinalizationFailure(failure.finalization);
      tileFailures.push_back(std::move(failure));
      continue;
    }
    physicalTileModules.push_back(std::move(result.module));
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

  std::string admissionGate;
  mlir::FailureOr<AcceptedWholeCardExecutable> accepted =
      admitWholeCardExecutable(std::move(physicalTileModules), selectedTileIR,
                               program, executionConfig, diagnostics,
                               &admissionGate, statistics,
                               tilePipelineParallelism);
  if (mlir::failed(accepted))
    return reportFailure(
        fail(isExactAdmissionFailure(admissionGate)
                 ? CardExecutableCompilationStatus::ProvenExactRejection
                 : CardExecutableCompilationStatus::IndeterminateFailure,
             admissionGate,
             "CardExecutable admission rejected the selected CardProgram", {},
             rotatingSlotAllocationsMaterialized));

  if (accepted->tiles.size() != expectedTileIds.size())
    return reportFailure(
        fail(CardExecutableCompilationStatus::ProvenExactRejection,
             "exact-executable-admission",
             "accepted physical Tile domain is incomplete", {},
             rotatingSlotAllocationsMaterialized));
  for (auto [index, tile] : llvm::enumerate(accepted->tiles))
    if (tile.getPhysicalCardId() != expectedCardId ||
        tile.getPhysicalTileId() != expectedTileIds[index] ||
        tile.getSelectedTileIR().empty())
      return reportFailure(fail(
          CardExecutableCompilationStatus::ProvenExactRejection,
          "exact-executable-admission",
          "exact admission changed physical identity or lost selected Tile "
          "IR",
          {}, rotatingSlotAllocationsMaterialized));

  CardExecutableCompilationResult result;
  result.status = CardExecutableCompilationStatus::Accepted;
  result.executable.emplace(std::move(*accepted));
  result.rotatingSlotAllocationsMaterialized =
      rotatingSlotAllocationsMaterialized;
  diagnostics << "wafer-compile: card-executable-compilation outcome=accepted"
              << " physical_tiles=" << result.executable->tiles.size()
              << " rotating_slot_allocations="
              << rotatingSlotAllocationsMaterialized << '\n';
  return result;
}

} // namespace wafer::compiler::detail
