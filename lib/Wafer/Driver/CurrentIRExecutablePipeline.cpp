//===- CurrentIRExecutablePipeline.cpp - Current IR downstream --------===//

#include "CurrentIRExecutablePipeline.h"

#include "Wafer/Analysis/Tile/TileDataflowAnalysis.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

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
    statistics.dteSendOperations += mlir::isa<InstrDTESendOp>(operation);
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
  DirectDTECompletionResult initialDTECompletion =
      rebuildRequiredDirectDTEWaits(instructionModules);
  if (!initialDTECompletion.succeeded())
    return fail(initialDTECompletion.failure ==
                        DirectDTECompletionFailureKind::Unsupported
                    ? ExecutableCompilationStatus::UnsupportedFailure
                    : ExecutableCompilationStatus::CompilerFailure,
                "direct-dte-completion", initialDTECompletion.detail);

  for (StandaloneTileModule &tile : *standalone) {
    rebuildCurrentBufferOwnerRelations(tile.module->getOperation(),
                                       tile.materializationRelations);
    mlir::FailureOr<unsigned> eliminated = cleanupCanonicalInstructionTransfers(
        *tile.module, tile.materializationRelations);
    if (mlir::failed(eliminated))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "instr-transfer-cleanup",
                  "canonical Instr transfer cleanup failed");
  }

  DirectDTECompletionResult finalDTECompletion =
      rebuildRequiredDirectDTEWaits(instructionModules);
  if (!finalDTECompletion.succeeded())
    return fail(finalDTECompletion.failure ==
                        DirectDTECompletionFailureKind::Unsupported
                    ? ExecutableCompilationStatus::UnsupportedFailure
                    : ExecutableCompilationStatus::CompilerFailure,
                "direct-dte-completion", finalDTECompletion.detail);

  for (StandaloneTileModule &tile : *standalone) {
    if (mlir::failed(rebuildRequiredNCCJoins(*tile.module)))
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
  if (options.captureTileDataflowIR)
    result.tileDataflowIRTrace = std::move(tileDataflowTrace);
  return result;
}

} // namespace wafer::compiler::detail
