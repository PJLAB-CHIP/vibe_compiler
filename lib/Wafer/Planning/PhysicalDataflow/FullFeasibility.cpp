//===- FullFeasibility.cpp - Current structural actual admission ------===//

#include "Wafer/Planning/PhysicalDataflow/FullFeasibility.h"

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <map>
#include <set>
#include <type_traits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

FullFeasibilityResult result(FullFeasibilityStatus status,
                             llvm::StringRef detail) {
  FullFeasibilityResult output;
  output.status = status;
  output.detail = detail.str();
  return output;
}

std::string demandDetail(const analysis::ExactDemandOutcome &outcome) {
  return std::visit(
      [](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
          return {};
        else
          return value.detail;
      },
      outcome);
}

mlir::FailureOr<std::vector<SemanticRootKey>> validateCapacityOwners(
    const CardExecutableCompilationResult &compilation,
    const CardProgramAnalysis &program,
    llvm::ArrayRef<CandidateNodeRootRelation> nodeRoots,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    std::string *failureReason) {
  std::map<uint32_t, SemanticRootKey> rootsByNode;
  for (const CandidateNodeRootRelation &relation : nodeRoots)
    if (!rootsByNode.try_emplace(relation.structuredNodeId, relation.root)
             .second) {
      if (failureReason)
        *failureReason = "candidate has duplicate node/root relations";
      return mlir::failure();
    }
  std::map<mlir::Operation *, SemanticRootKey> rootsByOperation;
  for (const analysis::RootRegionWork &work : rootWorks)
    if (work.rootOperation)
      rootsByOperation.try_emplace(work.rootOperation, work.id.root);
  std::set<SemanticRootKey> causal;
  auto addNode = [&](uint32_t node) {
    auto root = rootsByNode.find(node);
    if (root == rootsByNode.end()) {
      if (failureReason)
        *failureReason =
            (llvm::Twine("actual SPM demand has no semantic root for node ") +
             llvm::Twine(node))
                .str();
      return false;
    }
    causal.insert(root->second);
    return true;
  };
  bool sawDemand = false;
  for (const CardExecutableTileFailure &tile : compilation.tileFailures) {
    if (!isProvenExactTileMemoryPlanningFailure(tile.memoryPlanning))
      continue;
    auto validateDemand = [&](const auto &demand) {
      sawDemand = true;
      for (uint32_t node : demand.operationResultNodes)
        if (!addNode(node))
          return false;
      for (uint32_t node : demand.operandDemandNodes)
        if (!addNode(node))
          return false;
      for (uint32_t node : demand.scratchNodes)
        if (!addNode(node))
          return false;
      for (unsigned output : demand.outputIndices) {
        auto roots = program.dag.getObservableOutputRootNodes();
        if (output >= roots.size()) {
          if (failureReason)
            *failureReason =
                (llvm::Twine("actual SPM demand names output ") +
                 llvm::Twine(output) + " but the source has " +
                 llvm::Twine(roots.size()))
                    .str();
          return false;
        }
        for (uint32_t node : roots[output]) {
          const StructuredDAGNode *entry = program.dag.getNode(node);
          auto root = entry ? rootsByOperation.find(entry->operation)
                            : rootsByOperation.end();
          if (!entry || root == rootsByOperation.end()) {
            if (failureReason)
              *failureReason =
                  "actual SPM output demand has no semantic root";
            return false;
          }
          causal.insert(root->second);
        }
      }
      const bool nonempty = !demand.operationResultNodes.empty() ||
                            !demand.operandDemandNodes.empty() ||
                            !demand.scratchNodes.empty() ||
                            !demand.outputIndices.empty();
      if (!nonempty && failureReason)
        *failureReason = "actual SPM rejection has an ownerless demand";
      return nonempty;
    };
    for (const auto &demand : tile.memoryPlanning.spmCapacityConflictDemands)
      if (!validateDemand(demand))
        return mlir::failure();
    for (const auto &demand :
         tile.memoryPlanning.spmIndividuallyOversizedDemands)
      if (!validateDemand(demand))
        return mlir::failure();
  }
  if (!sawDemand || causal.empty()) {
    if (failureReason)
      *failureReason = !sawDemand
                           ? "actual SPM rejection has no conflict demand"
                           : "actual SPM rejection has no causal semantic root";
    return mlir::failure();
  }
  return std::vector<SemanticRootKey>(causal.begin(), causal.end());
}

} // namespace

FullFeasibilityResult evaluateCurrentStructuralCandidate(
    mlir::ModuleOp tensorProgram,
    const PhysicalDataflowPlanningProblem &problem,
    const TemporalState &state,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, FullFeasibilityStatistics *statistics,
    unsigned tilePipelineParallelism, bool captureTileDataflowIRTrace) {
  if (statistics) {
    *statistics = {};
    ++statistics->evaluations;
  }
  if (!tensorProgram)
    return result(FullFeasibilityStatus::CompilerBug,
                  "current structural candidate requires a TensorProgram");

  SpatialDomainEvaluation spatial = problem.getSpatialDomain().evaluate(
      problem.getProgram().dag, state.getSpatialPlan(),
      problem.getRelationLimits());
  const analysis::ExactDemandProof *demand =
      spatial.demand ? analysis::getExactDemandProof(*spatial.demand) : nullptr;
  if (!spatial.assignment || !demand)
    return result(FullFeasibilityStatus::CompilerBug,
                  spatial.failure  ? spatial.failure->detail
                  : spatial.demand ? demandDetail(*spatial.demand)
                                   : "structural candidate lost spatial demand");

  std::string detail;
  auto rootDomain = RootWorkDomain::create(
      problem.getProgram().dag, *spatial.assignment, *demand,
      problem.getProgram().availableTileIds, &detail);
  if (mlir::failed(rootDomain))
    return result(FullFeasibilityStatus::CompilerBug, detail);
  RootWorkCollectionOutcome rootOutcome = collectRootWorks(*rootDomain);
  RootWorkCollection *rootWorks = getRootWorkCollection(rootOutcome);
  if (!rootWorks || rootWorks->works.empty())
    return result(FullFeasibilityStatus::CompilerBug,
                  "structural candidate has no root-work inventory");

  auto regionDomain = RegionDomain::create(rootWorks->works, &detail);
  if (mlir::failed(regionDomain) ||
      !regionDomain->contains(state.getRegionPlan()))
    return result(FullFeasibilityStatus::CompilerBug,
                  detail.empty()
                      ? "structural candidate region is outside its domain"
                      : detail);
  TemporalDomainResult temporalDomain =
      buildTemporalDomain(state.getRegionPlan(), rootWorks->works);
  if (!temporalDomain.succeeded() ||
      !temporalDomain.domain->contains(state.getTemporalPlan()))
    return result(temporalDomain.failure &&
                          temporalDomain.failure->kind ==
                              TemporalDomainFailureKind::UnsupportedSemantics
                      ? FullFeasibilityStatus::Unsupported
                      : FullFeasibilityStatus::CompilerBug,
                  temporalDomain.failure
                      ? temporalDomain.failure->detail
                      : "structural candidate temporal choice is outside its "
                        "domain");

  CurrentStructuralCandidatePlan structural{
      *spatial.assignment, *demand, rootWorks->works, state.getRegionPlan(),
      state.getTemporalPlan()};
  if (statistics)
    ++statistics->candidateActualizations;
  mlir::FailureOr<MaterializedCardCandidate> materialized =
      materializeSearchStructuralCandidate(
          tensorProgram, problem.getCardId(), problem.getProgram(), structural,
          /*statistics=*/nullptr, diagnostics);
  if (mlir::failed(materialized))
    return result(FullFeasibilityStatus::Unsupported,
                  "current structural candidate is not materializable by the "
                  "current TileRegion transformation");

  llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots =
      materialized->nodeRoots;
  if (statistics)
    ++statistics->executableGateInvocations;
  CardExecutablePreparation preparation;
  CardExecutableCompilationResult compilation = compileCardModuleToExecutable(
      std::move(materialized->module), problem.getCardId(),
      problem.getProgram().availableTileIds, materialized->relations,
      preparation, program, executionConfig, diagnostics, programData,
      statistics ? &statistics->exactGates : nullptr, tilePipelineParallelism,
      captureTileDataflowIRTrace);

  FullFeasibilityResult output;
  if (compilation.isAccepted()) {
    output.status = FullFeasibilityStatus::Accepted;
    output.compilation.emplace(std::move(compilation));
    return output;
  }
  if (compilation.isProvenExactRejection()) {
    const bool hasSPMCapacityFailure = llvm::any_of(
        compilation.tileFailures, [](const CardExecutableTileFailure &failure) {
          return isProvenExactTileMemoryPlanningFailure(failure.memoryPlanning);
        });
    if (hasSPMCapacityFailure) {
      auto owners = validateCapacityOwners(
          compilation, problem.getProgram(), nodeRoots, rootWorks->works,
          &detail);
      if (mlir::failed(owners))
        return result(FullFeasibilityStatus::CompilerBug,
                      detail.empty()
                          ? "actual SPM rejection has incomplete owner evidence"
                          : detail);
      output.causalRoots = std::move(*owners);
    }
    output.status = FullFeasibilityStatus::ExactRejection;
    output.detail = compilation.detail;
    output.compilation.emplace(std::move(compilation));
    return output;
  }
  output.status =
      compilation.status == CardExecutableCompilationStatus::UnsupportedFailure
          ? FullFeasibilityStatus::Unsupported
      : compilation.status == CardExecutableCompilationStatus::CompilerFailure
          ? FullFeasibilityStatus::CompilerBug
          : FullFeasibilityStatus::Indeterminate;
  output.detail = compilation.detail;
  output.compilation.emplace(std::move(compilation));
  return output;
}

} // namespace wafer::compiler::detail
