//===- CardBaselineCompilation.cpp -----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineCompilation.h"

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Planning/Baseline/BaselineTemporalPlan.h"
#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Support/CompileTiming.h"

#include <algorithm>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

mlir::FailureOr<llvm::SmallVector<SemanticRootKey, 8>>
collectActualSPMRejectedRoots(
    const CardExecutableCompilationResult &compilation,
    const CardProgramAnalysis &program, const CanonicalBaselinePlan &plan,
    llvm::ArrayRef<CandidateNodeRootRelation> nodeRoots,
    std::string *failureReason) {
  if (!compilation.isProvenExactRejection() ||
      compilation.tileFailures.empty()) {
    if (failureReason)
      *failureReason =
          "candidate rejection is not an actual Tile SPM capacity result";
    return mlir::failure();
  }

  std::map<uint32_t, SemanticRootKey> rootsByNode;
  for (const CandidateNodeRootRelation &relation : nodeRoots)
    if (!rootsByNode.try_emplace(relation.structuredNodeId, relation.root)
             .second) {
      if (failureReason)
        *failureReason = "candidate node/root relation is duplicated";
      return mlir::failure();
    }

  std::map<uint32_t, SemanticRootKey> sourceRoots;
  for (const analysis::RootRegionWork &work : plan.rootWorks) {
    if (!work.rootOperation)
      continue;
    auto node = llvm::find_if(program.dag.getNodes(), [&](const auto &entry) {
      return entry.operation == work.rootOperation;
    });
    if (node == program.dag.getNodes().end())
      continue;
    auto [position, inserted] = sourceRoots.try_emplace(node->id, work.id.root);
    if (!inserted && position->second != work.id.root) {
      if (failureReason)
        *failureReason = "source node maps to several semantic roots";
      return mlir::failure();
    }
  }

  std::set<SemanticRootKey> affected;
  auto collectDemand =
      [&](const TileMemoryPlanningFailure::SPMDemandEvidence &demand)
      -> mlir::LogicalResult {
    std::set<SemanticRootKey> demandRoots;
    auto collectNode = [&](uint32_t node) {
      auto root = rootsByNode.find(node);
      if (root == rootsByNode.end())
        return false;
      demandRoots.insert(root->second);
      return true;
    };
    for (uint32_t node : demand.operationResultNodes)
      if (!collectNode(node))
        return mlir::failure();
    for (uint32_t node : demand.operandDemandNodes)
      if (!collectNode(node))
        return mlir::failure();
    for (uint32_t node : demand.scratchNodes)
      if (!collectNode(node))
        return mlir::failure();
    for (unsigned output : demand.outputIndices) {
      auto outputRoots = program.dag.getObservableOutputRootNodes();
      if (output >= outputRoots.size() || outputRoots[output].empty())
        return mlir::failure();
      for (uint32_t node : outputRoots[output]) {
        auto root = sourceRoots.find(node);
        if (root == sourceRoots.end())
          return mlir::failure();
        demandRoots.insert(root->second);
      }
    }
    if (demandRoots.empty())
      return mlir::failure();
    affected.insert(demandRoots.begin(), demandRoots.end());
    return mlir::success();
  };

  bool sawCapacityDemand = false;
  for (const CardExecutableTileFailure &tile : compilation.tileFailures) {
    if (!isProvenExactTileMemoryPlanningFailure(tile.memoryPlanning)) {
      if (failureReason)
        *failureReason =
            "candidate rejection mixes SPM capacity with another failure";
      return mlir::failure();
    }
    for (const auto &demand : tile.memoryPlanning.spmCapacityConflictDemands) {
      sawCapacityDemand = true;
      if (mlir::failed(collectDemand(demand))) {
        if (failureReason)
          *failureReason =
              "actual SPM conflict demand has no exact semantic owner";
        return mlir::failure();
      }
    }
    for (const auto &demand :
         tile.memoryPlanning.spmIndividuallyOversizedDemands) {
      sawCapacityDemand = true;
      if (mlir::failed(collectDemand(demand))) {
        if (failureReason)
          *failureReason =
              "actual oversized SPM demand has no exact semantic owner";
        return mlir::failure();
      }
    }
  }
  if (!sawCapacityDemand || affected.empty()) {
    if (failureReason)
      *failureReason = "actual SPM rejection has no causal conflict demand";
    return mlir::failure();
  }
  return llvm::SmallVector<SemanticRootKey, 8>(affected.begin(),
                                               affected.end());
}

void printSPMFailure(const CardExecutableCompilationResult &compilation,
                     llvm::raw_ostream &diagnostics) {
  if (compilation.tileFailures.empty())
    return;
  const CardExecutableTileFailure &tile = compilation.tileFailures.front();
  const TileMemoryPlanningFailure &memory = tile.memoryPlanning;
  if (!memory.spmCapacityOverflow)
    return;
  diagnostics << "wafer-compile: baseline actual SPM conflict tile="
              << tile.tileId.getValue();
  for (const auto &demand : memory.spmCapacityConflictDemands) {
    diagnostics << " demand(bytes=" << demand.bytes << ", type=" << demand.type
                << ", result_nodes=[";
    llvm::interleaveComma(demand.operationResultNodes, diagnostics);
    diagnostics << "], operand_nodes=[";
    llvm::interleaveComma(demand.operandDemandNodes, diagnostics);
    diagnostics << "], scratch_nodes=[";
    llvm::interleaveComma(demand.scratchNodes, diagnostics);
    diagnostics << "], outputs=[";
    llvm::interleaveComma(demand.outputIndices, diagnostics);
    diagnostics << "], users=[";
    llvm::interleaveComma(
        demand.userOperationNames, diagnostics,
        [&](mlir::OperationName name) { diagnostics << name.getStringRef(); });
    diagnostics << "])";
  }
  diagnostics << '\n';
}

void accumulateMaterializationStatistics(
    BaselineStatistics &baseline,
    const CandidateMaterializationStatistics &candidate) {
  baseline.spatialCoordinateQueries += candidate.spatialCoordinateQueries;
  baseline.exactDemandSatisfiedEdges = std::max(
      baseline.exactDemandSatisfiedEdges, candidate.exactDemandSatisfiedEdges);
  baseline.baselineSourcePreparations += candidate.sourcePreparations;
  baseline.baselineMaterializationPreparations +=
      candidate.materializationPreparations;
  baseline.baselineCardModuleMaterializations +=
      candidate.cardModuleMaterializations;
  baseline.baselineTileEntryMaterializations +=
      candidate.tileEntryMaterializations;
  baseline.baselineMaximumTileMaterializationWorkers =
      std::max(baseline.baselineMaximumTileMaterializationWorkers,
               candidate.maximumTileMaterializationWorkers);
}

} // namespace

mlir::FailureOr<CardBaselineCompilationResult> compileCardBaseline(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, BaselineStatistics *baselineStatistics,
    unsigned tilePipelineParallelism, bool captureTileDataflowIRTrace) {
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
  std::string failureReason;
  mlir::FailureOr<CanonicalBaselinePlan> resolved = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "deterministic-baseline", "build-initial-candidate");
    return buildCanonicalBaselinePlan(**analysis, &failureReason);
  }();
  if (mlir::failed(resolved)) {
    diagnostics << "wafer-compile: canonical baseline planning failed: "
                << failureReason << '\n';
    return mlir::failure();
  }

  while (true) {
    CompleteCandidatePlan candidatePlan{resolved->spatial,
                                        resolved->demand,
                                        resolved->rootWorks,
                                        resolved->regions,
                                        resolved->temporal,
                                        resolved->representations.plan,
                                        resolved->movements.plan,
                                        resolved->movements.resources,
                                        resolved->storage.plan,
                                        resolved->storage.resources,
                                        {},
                                        resolved->preparedAttention};
    CandidateMaterializationStatistics candidateStatistics;
    mlir::FailureOr<MaterializedCardCandidate> materialized =
        materializeCardCandidate(
            tensorProgram, cardId, **analysis, candidatePlan,
            SpatialDataflowMaterializationMode::IndependentDDRStages,
            baselineStatistics ? &candidateStatistics : nullptr, diagnostics);
    if (baselineStatistics)
      accumulateMaterializationStatistics(*baselineStatistics,
                                          candidateStatistics);
    if (mlir::failed(materialized))
      return mlir::failure();

    if (mlir::failed(verifyMaterializedCardCandidate(
            *materialized->module, materialized->assignment, (*analysis)->dag,
            materialized->relations, (*analysis)->availableTileIds,
            failureReason))) {
      diagnostics << "wafer-compile: baseline CardModule verification failed: "
                  << failureReason << '\n';
      return mlir::failure();
    }

    llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots =
        materialized->nodeRoots;
    CardExecutablePreparation preparation;
    CardExecutableCompilationResult compilation = compileCardModuleToExecutable(
        std::move(materialized->module), cardId, (*analysis)->availableTileIds,
        materialized->relations, preparation, program, executionConfig,
        diagnostics, programData,
        baselineStatistics ? &baselineStatistics->exactGates : nullptr,
        tilePipelineParallelism, captureTileDataflowIRTrace);
    if (compilation.isAccepted()) {
      if (baselineStatistics)
        baselineStatistics->baselineTileIRPrints +=
            compilation.tileDataflowIRTrace.size();
      return CardBaselineCompilationResult(
          compilation.takeExecutable(),
          std::move(compilation.tileDataflowIRTrace));
    }

    if (baselineStatistics) {
      ++baselineStatistics->materializationRejections;
      if (compilation.status ==
          CardExecutableCompilationStatus::IndeterminateFailure)
        ++baselineStatistics->indeterminateCompilationFailures;
    }
    printSPMFailure(compilation, diagnostics);

    std::string feedbackFailure;
    mlir::FailureOr<llvm::SmallVector<SemanticRootKey, 8>> affected =
        collectActualSPMRejectedRoots(compilation, **analysis, *resolved,
                                      nodeRoots, &feedbackFailure);
    if (mlir::succeeded(affected)) {
      if (baselineStatistics)
        ++baselineStatistics->actualSPMCapacityRejections;
      mlir::FailureOr<bool> refined = refineTemporalPlanFromActualSPMFeedback(
          resolved->temporal, resolved->rootWorks, *affected, &feedbackFailure);
      if (mlir::succeeded(refined) && *refined) {
        if (mlir::failed(
                recloseCanonicalBaselinePlan(*resolved, &feedbackFailure))) {
          diagnostics << "wafer-compile: baseline candidate reclose failed: "
                      << feedbackFailure << '\n';
          return mlir::failure();
        }
        if (baselineStatistics)
          ++baselineStatistics->actualTemporalRefinements;
        continue;
      }
    }

    diagnostics << "wafer-compile: baseline CardExecutable gate failed: gate="
                << compilation.gate << " detail=" << compilation.detail;
    if (!feedbackFailure.empty())
      diagnostics << " feedback=" << feedbackFailure;
    diagnostics << '\n';
    return mlir::failure();
  }
}

} // namespace wafer::compiler::detail
