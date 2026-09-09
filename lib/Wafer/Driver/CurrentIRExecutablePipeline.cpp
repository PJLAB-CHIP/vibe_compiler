//===- CurrentIRExecutablePipeline.cpp - Current IR downstream --------===//

#include "CurrentIRExecutablePipeline.h"

#include "Wafer/Analysis/Tile/TileDataflowAnalysis.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/NativeDirectDTEMultiSend.h"
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <utility>

namespace wafer::compiler::detail {
namespace {

static ExecutableCompilationResult fail(ExecutableCompilationStatus status,
                                        llvm::StringRef gate,
                                        llvm::StringRef detail) {
  ExecutableCompilationResult result;
  result.status = status;
  result.gate = gate.str();
  result.detail = detail.str();
  return result;
}

static std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

static void
collectInstructionStatistics(mlir::ModuleOp module,
                             CurrentIRDownstreamStatistics &statistics) {
  module.walk([&](mlir::Operation *operation) {
    statistics.instructionOperations +=
        mlir::isa<WaferInstructionOpInterface>(operation);
    statistics.rdmaOperations += mlir::isa<InstrRDMAOp>(operation);
    statistics.wdmaOperations += mlir::isa<InstrWDMAOp>(operation);
    statistics.gatherScatterOperations +=
        mlir::isa<InstrGatherScatterOp>(operation);
    statistics.dteSendOperations +=
        mlir::isa<InstrDTESendOp, InstrDTEBroadcastOp, InstrDTEScatterOp>(
            operation);
    statistics.dteBroadcastOperations +=
        mlir::isa<InstrDTEBroadcastOp>(operation);
    statistics.dteScatterOperations += mlir::isa<InstrDTEScatterOp>(operation);
    statistics.dteReceiveOperations += mlir::isa<InstrDTERecvOp>(operation);
    statistics.dteWaitOperations += mlir::isa<InstrDTEWaitOp>(operation);
    statistics.nccJoinOperations += mlir::isa<SyncNCCJoinOp>(operation);
  });
}

static void eraseDeadSubviewOperations(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::memref::SubViewOp, 16> dead;
    module.walk([&](mlir::memref::SubViewOp subview) {
      if (subview->use_empty())
        dead.push_back(subview);
    });
    for (mlir::memref::SubViewOp subview : llvm::reverse(dead)) {
      if (!subview->use_empty())
        continue;
      rewriter.eraseOp(subview);
      changed = true;
    }
  }
}

static PhysicalDataflowIRInventory
collectPhysicalCandidateInventory(mlir::ModuleOp module) {
  PhysicalDataflowIRInventory inventory;
  bool first = true;
  for (TileModuleOp tile : module.getOps<TileModuleOp>()) {
    PhysicalTileIRInventory tileInventory;
    tileInventory.tile = TileId(tile.getTileIdAttr().getInt());
    tile.walk([&](TileRegionOp region) {
      uint64_t nested = 0;
      uint64_t dataflow = 0;
      region->walk([&](mlir::Operation *operation) {
        if (operation == region.getOperation())
          return;
        ++nested;
        dataflow += mlir::isa<WaferTileDataflowOpInterface>(operation);
      });
      ++inventory.tileRegions;
      inventory.nestedOperations += nested;
      inventory.tileDataflowOperations += dataflow;
      ++tileInventory.regions;
      tileInventory.nestedOperations += nested;
      tileInventory.tileDataflowOperations += dataflow;
      if (first) {
        inventory.minimumNestedOperations = nested;
        inventory.maximumNestedOperations = nested;
        inventory.minimumTileDataflowOperations = dataflow;
        inventory.maximumTileDataflowOperations = dataflow;
        first = false;
      } else {
        inventory.minimumNestedOperations =
            std::min(inventory.minimumNestedOperations, nested);
        inventory.maximumNestedOperations =
            std::max(inventory.maximumNestedOperations, nested);
        inventory.minimumTileDataflowOperations =
            std::min(inventory.minimumTileDataflowOperations, dataflow);
        inventory.maximumTileDataflowOperations =
            std::max(inventory.maximumTileDataflowOperations, dataflow);
      }
      inventory.singletonDataflowRegions += dataflow == 1;
    });
    inventory.tiles.push_back(tileInventory);
  }
  llvm::sort(inventory.tiles, [](const PhysicalTileIRInventory &lhs,
                                 const PhysicalTileIRInventory &rhs) {
    return lhs.tile.getValue() < rhs.tile.getValue();
  });
  inventory.tileModules = inventory.tiles.size();
  return inventory;
}

} // namespace

ExecutableCompilationResult compileCurrentIRCandidateToExecutable(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    StructuredMaterializationRelations relations, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const CurrentIRDownstreamOptions &options,
    CurrentIRDownstreamStatistics *downstreamStatistics,
    ExecutableLoweringStatistics *executableStatistics) {
  if (!module || mlir::failed(verifyPhysicalTileDataflow(*module)) ||
      !relations.boundaryRelations.empty() ||
      !relations.structuralOutputs.empty() ||
      mlir::failed(checkStructuredBufferRelationsCurrent(*module, relations)))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "current-ir-downstream-input",
                "downstream orchestration requires physical, "
                "movement-closed current IR and live buffer relations");

  const PhysicalDataflowIRInventory physicalInventory =
      collectPhysicalCandidateInventory(*module);

  PreparedExecutionStructureResult prepared = prepareTileExecutionStructure(
      *module, options.executionPipelines, options.executionLimits);
  if (!prepared.succeeded()) {
    const ExecutionStructureFailureKind kind = prepared.failure->kind;
    return fail(kind == ExecutionStructureFailureKind::Indeterminate
                    ? ExecutableCompilationStatus::IndeterminateFailure
                    : (kind == ExecutionStructureFailureKind::Unsupported
                           ? ExecutableCompilationStatus::UnsupportedFailure
                           : ExecutableCompilationStatus::CompilerFailure),
                "execution-structure", prepared.failure->detail);
  }
  MaterializedExecutionStructureResult execution =
      materializeExecutionStructure(std::move(module),
                                    std::move(*prepared.prepared));
  if (!execution.succeeded()) {
    const ExecutionStructureFailureKind kind = execution.failure->kind;
    return fail(kind == ExecutionStructureFailureKind::Indeterminate
                    ? ExecutableCompilationStatus::IndeterminateFailure
                    : (kind == ExecutionStructureFailureKind::Unsupported
                           ? ExecutableCompilationStatus::UnsupportedFailure
                           : ExecutableCompilationStatus::CompilerFailure),
                "execution-structure", execution.failure->detail);
  }
  module = std::move(execution.materialized->module);
  rebuildCurrentBufferOwnerRelations(module->getOperation(), relations);
  if (downstreamStatistics)
    downstreamStatistics->materializedExecutionPipelines +=
        execution.materialized->pipelines.size();

  std::string fanoutFailure;
  auto standalone = createStandaloneTileModules(std::move(module),
                                                &fanoutFailure, &relations);
  if (mlir::failed(standalone))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "standalone-tile-fanout", fanoutFailure);

  std::vector<CanonicalInstructionTile> canonicalTiles;
  canonicalTiles.reserve(standalone->size());
  std::vector<std::string> tileDataflowTrace;
  for (StandaloneTileModule &tile : *standalone) {
    if (options.captureTileDataflowIR)
      tileDataflowTrace.push_back(printModule(*tile.module));
    llvm::SmallVector<TileRegionOp, 8> regions;
    tile.module->walk([&](TileRegionOp region) { regions.push_back(region); });
    if (downstreamStatistics)
      downstreamStatistics->tileRegionsLowered += regions.size();

    TileRegionToInstrLoweringSession session(*tile.module->getContext());
    for (TileRegionOp region : regions)
      if (mlir::failed(convertTileRegionToInstr(region, session)))
        return fail(ExecutableCompilationStatus::UnsupportedFailure,
                    "tile-to-instr",
                    "one physical TileRegion has no exact Instr lowering");
    if (mlir::failed(convertBufferizationCopiesToInstr(*tile.module, session)))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "tile-to-instr-relations",
                  "Tile-to-Instr left an unclassified current movement");
    eraseDeadSubviewOperations(*tile.module);
  }

  llvm::SmallVector<mlir::ModuleOp, 16> instructionModules;
  for (StandaloneTileModule &tile : *standalone)
    instructionModules.push_back(*tile.module);
  auto timed = [](llvm::StringRef stage, auto &&run) {
    wafer::support::ScopedCompileTimingSpan timing(
        "stage", "current-ir-downstream", stage);
    return run();
  };
  NativeDirectDTEMultiSendResult coalesced = timed("coalesce-direct-dte", [&] {
    return coalesceExactDirectDTETransfers(instructionModules);
  });
  if (!coalesced.succeeded())
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "exact-direct-dte-coalescing", coalesced.detail);
  wafer::support::addCompileCounter("communication", "coalesced-p2p-transfers",
                                    coalesced.statistics.coalescedP2PTransfers);
  wafer::support::addCompileCounter(
      "communication", "coalesced-unicast-sends",
      coalesced.statistics.unicastSendOperationsRemoved);
  wafer::support::addCompileCounter(
      "communication", "coalesced-unicast-receives",
      coalesced.statistics.unicastReceiveOperationsRemoved);
  for (StandaloneTileModule &tile : *standalone)
    eraseDeadSubviewOperations(*tile.module);
  NativeDirectDTEMultiSendResult multiSend = timed("native-direct-dte", [&] {
    return materializeNativeDirectDTEMultiSends(instructionModules);
  });
  if (!multiSend.succeeded())
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "native-direct-dte-multi-send", multiSend.detail);
  for (StandaloneTileModule &tile : *standalone)
    eraseDeadSubviewOperations(*tile.module);
  wafer::support::addCompileCounter("communication",
                                    "native-broadcast-operations",
                                    multiSend.statistics.broadcastOperations);
  wafer::support::addCompileCounter("communication",
                                    "native-scatter-operations",
                                    multiSend.statistics.scatterOperations);
  if (downstreamStatistics)
    downstreamStatistics->dteUnicastSendsCoalesced +=
        coalesced.statistics.unicastSendOperationsRemoved +
        multiSend.statistics.unicastSendOperationsRemoved;
  DirectDTECompletionResult initialDTECompletion =
      timed("initial-direct-dte-completion",
            [&] { return rebuildRequiredDirectDTEWaits(instructionModules); });
  if (!initialDTECompletion.succeeded())
    return fail(initialDTECompletion.failure ==
                        DirectDTECompletionFailureKind::Unsupported
                    ? ExecutableCompilationStatus::UnsupportedFailure
                    : ExecutableCompilationStatus::CompilerFailure,
                "direct-dte-completion", initialDTECompletion.detail);

  for (StandaloneTileModule &tile : *standalone) {
    rebuildCurrentBufferOwnerRelations(tile.module->getOperation(),
                                       tile.materializationRelations);
    mlir::FailureOr<unsigned> eliminated = timed("instr-transfer-cleanup", [&] {
      return cleanupCanonicalInstructionTransfers(
          *tile.module, tile.materializationRelations);
    });
    if (mlir::failed(eliminated))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "instr-transfer-cleanup",
                  "canonical Instr transfer cleanup failed");
  }

  llvm::SmallVector<TileId> completionTileIds;
  for (const StandaloneTileModule &tile : *standalone)
    completionTileIds.push_back(tile.tileId);
  auto sharedCompletion = timed("shared-ddr-completion", [&] {
    return materializeSharedDDRCompletion(instructionModules,
                                          completionTileIds);
  });
  if (!sharedCompletion.succeeded())
    return fail(sharedCompletion.failure ==
                        SharedDDRCompletionFailure::Unsupported
                    ? ExecutableCompilationStatus::UnsupportedFailure
                    : ExecutableCompilationStatus::CompilerFailure,
                "shared-ddr-completion", sharedCompletion.detail);

  DirectDTECompletionResult finalDTECompletion =
      timed("final-direct-dte-completion",
            [&] { return rebuildRequiredDirectDTEWaits(instructionModules); });
  if (!finalDTECompletion.succeeded())
    return fail(finalDTECompletion.failure ==
                        DirectDTECompletionFailureKind::Unsupported
                    ? ExecutableCompilationStatus::UnsupportedFailure
                    : ExecutableCompilationStatus::CompilerFailure,
                "direct-dte-completion", finalDTECompletion.detail);

  for (StandaloneTileModule &tile : *standalone) {
    if (mlir::failed(timed("ncc-completion", [&] {
          return rebuildRequiredNCCJoins(*tile.module);
        })))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "instr-completion", "fresh NCC completion placement failed");
    rebuildCurrentBufferOwnerRelations(tile.module->getOperation(),
                                       tile.materializationRelations);
    if (analysis::containsTileDataflowOperations(tile.module->getOperation()) ||
        mlir::failed(mlir::verify(*tile.module)) ||
        mlir::failed(checkStructuredBufferRelationsCurrent(
            tile.module->getOperation(), tile.materializationRelations)))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "canonical-instr",
                  "Tile-to-Instr produced invalid or incomplete current IR");
    if (downstreamStatistics)
      collectInstructionStatistics(*tile.module, *downstreamStatistics);
    canonicalTiles.push_back(CanonicalInstructionTile{
        tile.cardId, tile.tileId, std::move(tile.module),
        std::move(tile.materializationRelations)});
  }

  ExecutableCompilationResult result =
      compileCanonicalInstructionTilesToExecutable(
          std::move(canonicalTiles), expectedCardId, expectedTileIds, program,
          executionConfig, diagnostics, programData, executableStatistics,
          options.tilePipelineParallelism);
  if (result.isAccepted() && result.executable)
    result.physicalIRInventory = physicalInventory;
  if (options.captureTileDataflowIR)
    result.tileDataflowIRTrace = std::move(tileDataflowTrace);
  return result;
}

} // namespace wafer::compiler::detail
