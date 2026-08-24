//===- FullFeasibility.cpp - Complete candidate actual admission ------===//

#include "Wafer/Planning/PhysicalDataflow/FullFeasibility.h"

#include "Wafer/Planning/PhysicalDataflow/CanonicalAttentionWorkProjection.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSchedulePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RepresentationDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Planning/PhysicalDataflow/ScheduleDomain.h"
#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

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

struct PreparedCandidate {
  CompleteCandidatePlan materialization;
};

struct PreparationResult {
  std::optional<PreparedCandidate> candidate;
  FullFeasibilityStatus failure = FullFeasibilityStatus::CompilerBug;
  std::string detail;
};

ScheduleDomainInput
makeScheduleInput(const ScheduledState &state, const EventGraph &eventGraph,
                  llvm::ArrayRef<SlotLifetimeRequirement> slotLifetimes) {
  ScheduleDomainInput input;
  input.structure = state.getExecutionStructurePlan();
  input.buffers = state.getBufferPlan();
  input.events.assign(eventGraph.getEvents().begin(),
                      eventGraph.getEvents().end());
  input.hardDependencies.assign(eventGraph.getHardDependencies().begin(),
                                eventGraph.getHardDependencies().end());
  input.orderChoices.assign(eventGraph.getOrderChoices().begin(),
                            eventGraph.getOrderChoices().end());
  input.completionObligations.assign(
      eventGraph.getCompletionObligations().begin(),
      eventGraph.getCompletionObligations().end());
  input.resourceUses.assign(eventGraph.getResourceUses().begin(),
                            eventGraph.getResourceUses().end());
  input.components.assign(eventGraph.getComponents().begin(),
                          eventGraph.getComponents().end());
  input.slotLifetimes.assign(slotLifetimes.begin(), slotLifetimes.end());
  return input;
}

PreparationResult
prepareCandidate(const PhysicalDataflowPlanningProblem &problem,
                 const ScheduledState &state, const EventGraph &eventGraph,
                 llvm::ArrayRef<SlotLifetimeRequirement> slotLifetimes) {
  SpatialDomainEvaluation spatial = problem.getSpatialDomain().evaluate(
      problem.getProgram().dag, state.getSpatialPlan(),
      problem.getRelationLimits());
  const analysis::ExactDemandProof *demand =
      spatial.demand ? analysis::getExactDemandProof(*spatial.demand) : nullptr;
  if (!spatial.assignment || !demand)
    return {{},
            FullFeasibilityStatus::CompilerBug,
            spatial.failure  ? spatial.failure->detail
            : spatial.demand ? demandDetail(*spatial.demand)
                             : "complete candidate lost spatial demand"};

  std::string detail;
  auto rootDomain = RootWorkDomain::create(
      problem.getProgram().dag, *spatial.assignment, *demand,
      problem.getProgram().availableTileIds, &detail);
  if (mlir::failed(rootDomain))
    return {{}, FullFeasibilityStatus::CompilerBug, std::move(detail)};
  RootWorkCollectionOutcome rootOutcome = collectRootWorks(*rootDomain);
  RootWorkCollection *rootWorks = getRootWorkCollection(rootOutcome);
  if (!rootWorks || rootWorks->works.empty())
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate has no root-work inventory"};

  auto regionDomain = RegionDomain::create(rootWorks->works, &detail);
  if (mlir::failed(regionDomain))
    return {{},
            FullFeasibilityStatus::CompilerBug,
            detail.empty() ? "complete candidate cannot rebuild region domain"
                           : std::move(detail)};
  if (!regionDomain->contains(state.getRegionPlan()))
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate region plan is outside its rebuilt domain"};

  CanonicalRepresentationPlanOutcome representationOutcome =
      buildCanonicalRepresentationPlan(
          state.getRegionPlan(), state.getTemporalPlan(), rootWorks->works);
  const CanonicalRepresentationCoordinate *representations =
      getCanonicalRepresentationCoordinate(representationOutcome);
  if (!representations)
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate cannot rebuild representation facts"};
  RepresentationDomainResult representationDomain =
      buildRepresentationDomain(*representations);
  if (!representationDomain.succeeded() ||
      !representationDomain.domain->contains(state.getRepresentationPlan()))
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate representation is outside its rebuilt domain"};

  CanonicalRepresentationCoordinate selectedPrimary;
  for (const LogicalRepresentationPlan &logical :
       state.getRepresentationPlan().logicalValues) {
    auto version = llvm::find_if(state.getRepresentationPlan().physicalVersions,
                                 [&](const PhysicalVersionPlan &candidate) {
                                   return candidate.id == logical.primary;
                                 });
    const RepresentationResourceDescription *resource =
        representationDomain.domain->findResource(logical.value);
    if (version == state.getRepresentationPlan().physicalVersions.end() ||
        !resource)
      return {{},
              FullFeasibilityStatus::CompilerBug,
              "complete candidate primary representation is incomplete"};
    selectedPrimary.plan.logicalValues.push_back(logical);
    selectedPrimary.plan.physicalVersions.push_back(*version);
    RepresentationResourceDescription selected = *resource;
    selected.version = logical.primary;
    selected.encoding = version->encoding;
    selectedPrimary.resources.push_back(std::move(selected));
  }
  llvm::sort(selectedPrimary.plan.logicalValues);
  llvm::sort(selectedPrimary.plan.physicalVersions);
  llvm::sort(selectedPrimary.resources,
             [](const RepresentationResourceDescription &lhs,
                const RepresentationResourceDescription &rhs) {
               return lhs.version < rhs.version;
             });
  CanonicalRepresentationCoordinate selectedRepresentations;
  selectedRepresentations.plan = state.getRepresentationPlan();
  for (const PhysicalVersionPlan &version :
       state.getRepresentationPlan().physicalVersions) {
    const RepresentationResourceDescription *resource =
        representationDomain.domain->findResource(version.id.logicalValue);
    if (!resource)
      return {{},
              FullFeasibilityStatus::CompilerBug,
              "complete candidate physical version has no exact resource"};
    RepresentationResourceDescription selected = *resource;
    selected.version = version.id;
    selected.encoding = version.encoding;
    selectedRepresentations.resources.push_back(std::move(selected));
  }
  llvm::sort(selectedRepresentations.resources,
             [](const RepresentationResourceDescription &lhs,
                const RepresentationResourceDescription &rhs) {
               return lhs.version < rhs.version;
             });

  CanonicalMovementPlanOutcome movementOutcome = buildCanonicalMovementPlan(
      state.getRegionPlan(), selectedPrimary, rootWorks->works);
  const CanonicalMovementCoordinate *movements =
      getCanonicalMovementCoordinate(movementOutcome);
  if (!movements)
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate cannot rebuild movement facts"};
  if (!(movements->plan == state.getMovementPlan()))
    return {{},
            FullFeasibilityStatus::Unsupported,
            "current Card materializer does not yet construct selected peer "
            "or relay movement"};

  CanonicalSerializedExecutionPlanOutcome serializedOutcome =
      buildCanonicalSerializedExecutionPlan(state.getRegionPlan(),
                                            state.getTemporalPlan());
  const SerializedExecutionPlan *serialized =
      getSerializedExecutionPlan(serializedOutcome);
  if (!serialized)
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate cannot rebuild serialized executions"};
  if (llvm::any_of(
          state.getExecutionStructurePlan().scopes,
          [](const ExecutionStructureChoice &choice) {
            return !std::holds_alternative<SerializedExecutionStructure>(
                choice);
          }))
    return {{},
            FullFeasibilityStatus::Unsupported,
            "current Card materializer does not yet construct a Pipelined "
            "execution structure"};

  CanonicalStoragePlanOutcome storageOutcome = buildCanonicalStoragePlan(
      selectedRepresentations, *movements, *serialized);
  const CanonicalStorageCoordinate *storage =
      getCanonicalStorageCoordinate(storageOutcome);
  if (!storage)
    return {{},
            FullFeasibilityStatus::CompilerBug,
            std::get<BrokenStoragePlan>(storageOutcome).detail};
  if (!(storage->plan == state.getBufferPlan()))
    return {{},
            FullFeasibilityStatus::Unsupported,
            "current Card materializer does not yet construct alias, reuse, "
            "or rotating storage"};

  ScheduleDomainResult scheduleDomain =
      buildScheduleDomain(makeScheduleInput(state, eventGraph, slotLifetimes));
  if (!scheduleDomain.succeeded())
    return {{},
            FullFeasibilityStatus::CompilerBug,
            scheduleDomain.failure
                ? scheduleDomain.failure->detail
                : "complete candidate cannot rebuild schedule domain"};
  ScheduleSuccessor firstSchedule = scheduleDomain.domain->getFirstPlan();
  if (firstSchedule.getKind() != ScheduleSuccessorKind::Plan ||
      !firstSchedule.getPlan())
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate schedule domain has no first leaf"};
  if (!(*firstSchedule.getPlan() == state.getSchedulePlan()))
    return {{},
            FullFeasibilityStatus::Unsupported,
            "current Card materializer does not yet construct a noncanonical "
            "event schedule"};

  CanonicalSchedulePlanOutcome scheduleOutcome =
      buildCanonicalSchedulePlan(*storage, *serialized);
  const CanonicalScheduleCoordinate *schedule =
      getCanonicalScheduleCoordinate(scheduleOutcome);
  if (!schedule)
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate cannot rebuild its canonical action prefix"};
  CanonicalAttentionWorkProjectionOutcome attentionOutcome =
      buildCanonicalAttentionWorkProjection(rootWorks->works, *representations,
                                            *movements, *storage, *schedule,
                                            &state.getTemporalPlan());
  const CanonicalAttentionWorkCoordinate *attention =
      getCanonicalAttentionWorkCoordinate(attentionOutcome);
  if (!attention)
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate cannot rebuild attention work"};
  PreparedAttentionDecompositionOutcome preparedOutcome =
      prepareSelectedAttentionDecomposition(*attention);
  const auto *prepared =
      std::get_if<PreparedAttentionDecomposition>(&preparedOutcome);
  if (!prepared)
    return {{},
            FullFeasibilityStatus::CompilerBug,
            "complete candidate cannot prepare attention decomposition"};

  PreparedCandidate candidate;
  candidate.materialization = {*spatial.assignment,
                               *demand,
                               rootWorks->works,
                               state.getRegionPlan(),
                               state.getTemporalPlan(),
                               state.getRepresentationPlan(),
                               *prepared};
  return {std::move(candidate), FullFeasibilityStatus::Accepted, {}};
}

mlir::FailureOr<std::vector<SemanticRootKey>>
validateCapacityOwners(const CardExecutableCompilationResult &compilation,
                       const CardProgramAnalysis &program,
                       llvm::ArrayRef<CandidateNodeRootRelation> nodeRoots,
                       llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  std::map<uint32_t, SemanticRootKey> rootsByNode;
  for (const CandidateNodeRootRelation &relation : nodeRoots)
    if (!rootsByNode.try_emplace(relation.structuredNodeId, relation.root)
             .second)
      return mlir::failure();
  std::map<mlir::Operation *, SemanticRootKey> rootsByOperation;
  for (const analysis::RootRegionWork &work : rootWorks)
    if (work.rootOperation)
      rootsByOperation.try_emplace(work.rootOperation, work.id.root);
  std::set<SemanticRootKey> causal;
  auto addNode = [&](uint32_t node) {
    auto root = rootsByNode.find(node);
    if (root == rootsByNode.end())
      return false;
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
        if (output >= roots.size())
          return false;
        for (uint32_t node : roots[output]) {
          const StructuredDAGNode *entry = program.dag.getNode(node);
          auto root = entry ? rootsByOperation.find(entry->operation)
                            : rootsByOperation.end();
          if (!entry || root == rootsByOperation.end())
            return false;
          causal.insert(root->second);
        }
      }
      return !demand.operationResultNodes.empty() ||
             !demand.operandDemandNodes.empty() ||
             !demand.scratchNodes.empty() || !demand.outputIndices.empty();
    };
    for (const auto &demand : tile.memoryPlanning.spmCapacityConflictDemands)
      if (!validateDemand(demand))
        return mlir::failure();
    for (const auto &demand :
         tile.memoryPlanning.spmIndividuallyOversizedDemands)
      if (!validateDemand(demand))
        return mlir::failure();
  }
  if (!sawDemand || causal.empty())
    return mlir::failure();
  return std::vector<SemanticRootKey>(causal.begin(), causal.end());
}

} // namespace

FullFeasibilityResult evaluateCompleteCandidate(
    mlir::ModuleOp tensorProgram,
    const PhysicalDataflowPlanningProblem &problem, const ScheduledState &state,
    const EventGraph &eventGraph,
    llvm::ArrayRef<SlotLifetimeRequirement> slotLifetimes,
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
                  "full feasibility requires a TensorProgram");
  PreparationResult prepared =
      prepareCandidate(problem, state, eventGraph, slotLifetimes);
  if (!prepared.candidate)
    return result(prepared.failure, prepared.detail);

  if (statistics)
    ++statistics->candidateActualizations;
  mlir::FailureOr<MaterializedCardCandidate> materialized =
      materializeCardCandidate(tensorProgram, problem.getCardId(),
                               problem.getProgram(),
                               prepared.candidate->materialization,
                               /*statistics=*/nullptr, diagnostics);
  if (mlir::failed(materialized))
    return result(FullFeasibilityStatus::CompilerBug,
                  "complete candidate materialization failed");
  std::string verificationFailure;
  if (mlir::failed(verifyMaterializedCardCandidate(
          *materialized->module, materialized->assignment,
          problem.getProgram().dag, materialized->relations,
          problem.getProgram().availableTileIds, verificationFailure)))
    return result(FullFeasibilityStatus::CompilerBug, verificationFailure);

  llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots =
      materialized->nodeRoots;
  std::vector<analysis::RootRegionWork> rootWorks =
      prepared.candidate->materialization.rootWorks;
  if (statistics)
    ++statistics->executableGateInvocations;
  CardExecutableCompilationResult compilation = compileCardModuleToExecutable(
      std::move(materialized->module), problem.getCardId(),
      problem.getProgram().availableTileIds, materialized->relations, program,
      executionConfig, diagnostics, programData,
      statistics ? &statistics->exactGates : nullptr, tilePipelineParallelism,
      captureTileDataflowIRTrace,
      /*applySelectedInstructionSchedule=*/false);

  FullFeasibilityResult output;
  if (compilation.isAccepted()) {
    output.status = FullFeasibilityStatus::Accepted;
    output.compilation.emplace(std::move(compilation));
    return output;
  }
  if (compilation.isProvenExactRejection()) {
    bool hasSPMCapacityFailure = llvm::any_of(
        compilation.tileFailures, [](const CardExecutableTileFailure &failure) {
          return isProvenExactTileMemoryPlanningFailure(failure.memoryPlanning);
        });
    if (hasSPMCapacityFailure) {
      auto owners = validateCapacityOwners(compilation, problem.getProgram(),
                                           nodeRoots, rootWorks);
      if (mlir::failed(owners))
        return result(FullFeasibilityStatus::CompilerBug,
                      "actual SPM rejection has incomplete owner evidence");
      output.causalRoots = std::move(*owners);
    }
    output.status = FullFeasibilityStatus::ExactRejection;
    output.detail = compilation.detail;
    output.compilation.emplace(std::move(compilation));
    return output;
  }
  output.status = FullFeasibilityStatus::Indeterminate;
  output.detail = compilation.detail;
  output.compilation.emplace(std::move(compilation));
  return output;
}

} // namespace wafer::compiler::detail
