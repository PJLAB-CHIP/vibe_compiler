//===- ExecutableCompilation.cpp - Policy-free executable seam ------===//

#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Analysis/Tile/TileDataflowAnalysis.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/Topology/TargetTopology.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/StructuredNodeUseIndex.h"

#include "Wafer/Support/BoundedTilePipelines.h"

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Instr/MemoryPlanningPipelines.h"
#include "Wafer/Transforms/Instr/RedundantTransferElimination.h"
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

struct TileLoweringResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
  std::string detail;
  TileMemoryPlanningFailure memoryPlanning;
  bool memoryPlanningFailed = false;
};

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
  retarget(relations.ddrBuffers);
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

static ExecutableCompilationResult
fail(ExecutableCompilationStatus status, llvm::StringRef gate,
     llvm::StringRef detail,
     llvm::SmallVector<ExecutableTileFailure, 4> tileFailures = {}) {
  ExecutableCompilationResult result;
  result.status = status;
  result.gate = gate.str();
  result.detail = detail.str();
  result.tileFailures = std::move(tileFailures);
  return result;
}

static ExecutableCompilationStatus
classifyTileMemoryFailureImpl(const TileMemoryPlanningFailure &failure) {
  switch (failure.kind) {
  case TileMemoryPlanningFailureKind::None:
  case TileMemoryPlanningFailureKind::Contract:
  case TileMemoryPlanningFailureKind::PreexistingPlacementFacts:
  case TileMemoryPlanningFailureKind::Verification:
    return ExecutableCompilationStatus::CompilerFailure;
  case TileMemoryPlanningFailureKind::SPMAllocation:
    switch (failure.spmPlanningFailureKind) {
    case SPMMemoryPlanningFailureKind::CapacityOverflow:
      return ExecutableCompilationStatus::ProvenExactRejection;
    case SPMMemoryPlanningFailureKind::ResourceExhausted:
      return ExecutableCompilationStatus::IndeterminateFailure;
    case SPMMemoryPlanningFailureKind::UnsupportedLifetime:
      return ExecutableCompilationStatus::UnsupportedFailure;
    case SPMMemoryPlanningFailureKind::MissingCompletion:
    case SPMMemoryPlanningFailureKind::Other:
    case SPMMemoryPlanningFailureKind::None:
      return ExecutableCompilationStatus::CompilerFailure;
    }
  }
  return ExecutableCompilationStatus::CompilerFailure;
}

static ExecutableCompilationStatus
combineTileMemoryFailureStatus(ExecutableCompilationStatus current,
                               ExecutableCompilationStatus next) {
  auto priority = [](ExecutableCompilationStatus status) {
    switch (status) {
    case ExecutableCompilationStatus::CompilerFailure:
      return 4;
    case ExecutableCompilationStatus::IndeterminateFailure:
      return 3;
    case ExecutableCompilationStatus::UnsupportedFailure:
      return 2;
    case ExecutableCompilationStatus::ProvenExactRejection:
      return 1;
    case ExecutableCompilationStatus::Accepted:
      return 0;
    }
    return 4;
  };
  return priority(next) > priority(current) ? next : current;
}

} // namespace

ExecutableCompilationStatus
classifyTileMemoryPlanningFailure(const TileMemoryPlanningFailure &failure) {
  return classifyTileMemoryFailureImpl(failure);
}

ExecutableCompilationResult compileCanonicalInstructionTilesToExecutable(
    std::vector<CanonicalInstructionTile> tiles, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, ExecutableLoweringStatistics *statistics,
    unsigned tilePipelineParallelism) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "stage", "actual-memory-target-gate", "canonical-instr-to-executable");
  if (statistics)
    ++statistics->actualMemoryTargetGateInvocations;
  auto failLeaf = [&](ExecutableCompilationResult result) {
    totalTiming.markFailed();
    return result;
  };

  if (tiles.empty() || tiles.size() != expectedTileIds.size())
    return failLeaf(fail(ExecutableCompilationStatus::CompilerFailure,
                         "actual-memory-target-input",
                         "canonical Instr Tile domain is incomplete"));

  mlir::MLIRContext *context = nullptr;
  std::vector<TileLoweringResult> loweringResults(tiles.size());
  for (auto [index, tile] : llvm::enumerate(tiles)) {
    if (!tile.module || tile.card != expectedCardId ||
        tile.tile != expectedTileIds[index])
      return failLeaf(fail(ExecutableCompilationStatus::CompilerFailure,
                           "actual-memory-target-input",
                           "canonical Instr Tile identity is inconsistent"));
    if (!context)
      context = tile.module->getContext();
    if (tile.module->getContext() != context ||
        analysis::containsTileDataflowOperations(tile.module->getOperation()) ||
        mlir::failed(mlir::verify(*tile.module)) ||
        mlir::failed(checkStructuredBufferRelationsCurrent(
            tile.module->getOperation(), tile.relations)))
      return failLeaf(fail(
          ExecutableCompilationStatus::CompilerFailure,
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

  const unsigned requestedWorkers =
      tilePipelineParallelism == 0
          ? wafer::support::kMaximumBoundedTilePipelineWorkers
          : tilePipelineParallelism;
  unsigned workers = wafer::support::runBoundedTilePipelines(
      context, loweringResults.size(), planMemory, requestedWorkers);
  if (statistics)
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);

  llvm::SmallVector<ExecutableTileFailure, 4> tileFailures;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> instructionModules;
  std::vector<StructuredMaterializationRelations> tileRelations;
  instructionModules.reserve(loweringResults.size());
  tileRelations.reserve(loweringResults.size());
  ExecutableCompilationStatus failureStatus =
      ExecutableCompilationStatus::ProvenExactRejection;
  for (auto [tileIndex, result] : llvm::enumerate(loweringResults)) {
    if (result.memoryPlanningFailed || !result.module) {
      ExecutableTileFailure failure;
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

  ExecutableLoweringFailure loweringFailure;
  mlir::FailureOr<ExecutableLoweringResult> executable =
      lowerTileModulesToExecutable(
          std::move(instructionModules), program, executionConfig, diagnostics,
          loweringFailure, programData, statistics, tilePipelineParallelism);
  if (mlir::failed(executable))
    return failLeaf(
        fail(loweringFailure.isProvenExactRejection()
                 ? ExecutableCompilationStatus::ProvenExactRejection
                 : ExecutableCompilationStatus::IndeterminateFailure,
             loweringFailure.getDiagnosticLabel(),
             loweringFailure.detail.empty() ? "Tile module lowering failed"
                                            : loweringFailure.detail));

  if (executable->tiles.size() != expectedTileIds.size())
    return failLeaf(fail(ExecutableCompilationStatus::CompilerFailure,
                         "device-executable-domain",
                         "Tile executable domain is incomplete"));
  for (auto [index, tile] : llvm::enumerate(executable->tiles))
    if (tile.getCardId() != expectedCardId ||
        tile.getTileId() != expectedTileIds[index])
      return failLeaf(
          fail(ExecutableCompilationStatus::CompilerFailure,
               "device-executable-domain",
               "executable lowering changed the selected Tile identity"));

  ExecutableCompilationResult result;
  result.status = ExecutableCompilationStatus::Accepted;
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

} // namespace wafer::compiler::detail
