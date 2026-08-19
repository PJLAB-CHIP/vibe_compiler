//===- CardBaselineCompilation.cpp -----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineCompilation.h"
#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"
#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"
#include "Wafer/Support/CompileTiming.h"

namespace wafer::compiler::detail {

mlir::FailureOr<CardBaselineCompilationResult> compileCardBaseline(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    llvm::raw_ostream &diagnostics, ProgramDataHandoff &programData,
    BaselineStatistics *baselineStatistics,
    unsigned tilePipelineParallelism,
    std::vector<std::string> *tileDataflowIRTrace) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "stage", "tensor-program-to-executable", "card-baseline-compilation");
  if (baselineStatistics)
    *baselineStatistics = {};

  mlir::FailureOr<std::unique_ptr<CardProgramAnalysis>> analysis = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "deterministic-baseline", "analyze-card-program");
    return analyzeCardProgram(tensorProgram, program, executionConfig,
                              diagnostics);
  }();
  if (mlir::failed(analysis))
    return mlir::failure();

  constexpr CardId cardId(0);
  mlir::FailureOr<CardBaselineAssignment> assignment =
      computeCardBaselineAssignment(**analysis, cardId, baselineStatistics,
                                    diagnostics);
  if (mlir::failed(assignment))
    return mlir::failure();

  mlir::FailureOr<CardBaselineModule> materialized = materializeCardBaseline(
      tensorProgram, cardId, (*analysis)->operationNodes, *assignment,
      baselineStatistics, diagnostics);
  if (mlir::failed(materialized))
    return mlir::failure();

  std::string failureReason;
  if (mlir::failed(verifyCardBaselineMaterialization(
          *materialized->module, *assignment, (*analysis)->dag,
          materialized->relations,
          (*analysis)->availableTileIds, failureReason))) {
    diagnostics << "wafer-compile: baseline CardModule verification failed: "
                << failureReason << '\n';
    return mlir::failure();
  }

  CardExecutableCompilationResult compilation = compileCardModuleToExecutable(
      std::move(materialized->module), cardId, (*analysis)->availableTileIds,
      /*selectedBufferRequests=*/{}, materialized->relations, program,
      executionConfig,
      diagnostics, programData,
      baselineStatistics ? &baselineStatistics->exactGates : nullptr,
      tilePipelineParallelism, tileDataflowIRTrace != nullptr);
  if (baselineStatistics)
    baselineStatistics->rotatingSlotAllocationsMaterialized +=
        compilation.rotatingSlotAllocationsMaterialized;
  if (!compilation.isAccepted()) {
    if (baselineStatistics) {
      ++baselineStatistics->materializationRejections;
      if (compilation.status ==
          CardExecutableCompilationStatus::IndeterminateFailure)
        ++baselineStatistics->indeterminateCompilationFailures;
    }
    diagnostics << "wafer-compile: baseline CardExecutable gate failed: gate="
                << compilation.gate << " detail=" << compilation.detail
                << '\n';
    if (!compilation.tileFailures.empty()) {
      const CardExecutableTileFailure &tile = compilation.tileFailures.front();
      const TileMemoryPlanningFailure &memory = tile.memoryPlanning;
      if (memory.spmCapacityOverflow) {
        diagnostics << "wafer-compile: baseline SPM conflict tile="
                    << tile.tileId.getValue();
        for (const auto &demand : memory.spmCapacityConflictDemands) {
          diagnostics << " demand(bytes=" << demand.bytes
                      << ", type=" << demand.type << ", location="
                      << demand.location << ", result_nodes=[";
          for (uint32_t node : demand.operationResultNodes) {
            const StructuredDAGNode *owner = (*analysis)->dag.getNode(node);
            diagnostics << node;
            if (owner && owner->operation)
              diagnostics << ':' << owner->operation->getName();
            diagnostics << ',';
          }
          diagnostics << "], operand_nodes=[";
          for (uint32_t node : demand.operandDemandNodes) {
            const StructuredDAGNode *owner = (*analysis)->dag.getNode(node);
            diagnostics << node;
            if (owner && owner->operation)
              diagnostics << ':' << owner->operation->getName();
            diagnostics << ',';
          }
          diagnostics << "])";
        }
        diagnostics << '\n';
      }
    }
    return mlir::failure();
  }

  if (baselineStatistics)
    baselineStatistics->baselineTileIRPrints +=
        compilation.tileDataflowIRTrace.size();
  if (tileDataflowIRTrace)
    *tileDataflowIRTrace = std::move(compilation.tileDataflowIRTrace);
  return CardBaselineCompilationResult(compilation.takeExecutable());
}

} // namespace wafer::compiler::detail
