//===- PlanningSession.cpp - Spatial planning frontier -----------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningSession.h"

#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Planning/PhysicalDataflow/StructuralReadiness.h"

#include <type_traits>
#include <variant>

namespace wafer::compiler::detail {
namespace {

std::string getDemandDetail(const analysis::ExactDemandOutcome &outcome) {
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

std::string getRootWorkFailureDetail(const RootWorkDomainFailure &failure) {
  return std::visit([](const auto &value) { return value.detail; }, failure);
}

} // namespace

SpatialChoiceOutcomeKind
classifySpatialChoiceOutcome(const analysis::ExactDemandOutcome &outcome) {
  switch (analysis::classifyExactDemandOutcome(outcome)) {
  case analysis::ExactDemandOutcomeCategory::Satisfied:
    return SpatialChoiceOutcomeKind::Satisfied;
  case analysis::ExactDemandOutcomeCategory::UnsupportedSemantics:
    return SpatialChoiceOutcomeKind::Unsupported;
  case analysis::ExactDemandOutcomeCategory::IndeterminateResourceExhaustion:
    return SpatialChoiceOutcomeKind::Indeterminate;
  case analysis::ExactDemandOutcomeCategory::CompilerContractError:
    return SpatialChoiceOutcomeKind::CompilerBug;
  }
  return SpatialChoiceOutcomeKind::CompilerBug;
}

SpatialExpansionResult
PhysicalDataflowPlanningSession::evaluateAndQueue(SpatialPlan choice,
                                                  bool proposalChoice) {
  ++work.spatialDemandQueries;
  SpatialDomainEvaluation evaluation = problem.getSpatialDomain().evaluate(
      problem.getProgram().dag, choice, problem.getRelationLimits());
  if (evaluation.failure) {
    if (evaluation.failure->kind ==
        SpatialDomainFailureKind::UnsupportedSemantics) {
      if (proposalChoice)
        resolvedProposalChoices.insert(std::move(choice));
      ++work.unsupportedSpatialChoices;
      return {SpatialExpansionKind::Unsupported, evaluation.failure->detail};
    }
    return {SpatialExpansionKind::CompilerBug, evaluation.failure->detail};
  }
  if (!evaluation.demand)
    return {SpatialExpansionKind::CompilerBug,
            "spatial demand query produced no typed outcome"};

  const SpatialChoiceOutcomeKind outcome =
      classifySpatialChoiceOutcome(*evaluation.demand);
  if (outcome == SpatialChoiceOutcomeKind::Indeterminate) {
    pausedSpatialChoice = std::move(choice);
    pausedChoiceIsProposal = proposalChoice;
    ++work.indeterminateSpatialChoices;
    return {SpatialExpansionKind::Indeterminate,
            getDemandDetail(*evaluation.demand)};
  }
  pausedSpatialChoice.reset();
  pausedChoiceIsProposal = false;
  if (outcome == SpatialChoiceOutcomeKind::Unsupported) {
    if (proposalChoice)
      resolvedProposalChoices.insert(std::move(choice));
    ++work.unsupportedSpatialChoices;
    return {SpatialExpansionKind::Unsupported,
            getDemandDetail(*evaluation.demand)};
  }
  if (outcome == SpatialChoiceOutcomeKind::CompilerBug)
    return {SpatialExpansionKind::CompilerBug,
            getDemandDetail(*evaluation.demand)};

  const analysis::ExactDemandProof *proof =
      analysis::getExactDemandProof(*evaluation.demand);
  if (!evaluation.assignment || !proof)
    return {SpatialExpansionKind::CompilerBug,
            "satisfied spatial choice has no assignment or demand proof"};
  std::string detail;
  mlir::FailureOr<RootWorkDomain> rootDomain = RootWorkDomain::create(
      problem.getProgram().dag, *evaluation.assignment, *proof,
      problem.getProgram().availableTileIds, &detail);
  if (mlir::failed(rootDomain))
    return {SpatialExpansionKind::CompilerBug, std::move(detail)};
  RootWorkSuccessor rootWork =
      rootDomain->getFirstWork(problem.getRelationLimits());
  uint64_t validatedRootWorks = 0;
  while (rootWork.getKind() == RootWorkSuccessorKind::Work) {
    ++work.rootWorkSuccessorSteps;
    ++work.rootWorksValidated;
    ++validatedRootWorks;
    const analysis::RootRegionWork *value = rootWork.getWork();
    const RootWorkCursor *cursor = rootWork.getCursor();
    if (!value || !cursor)
      return {SpatialExpansionKind::CompilerBug,
              "root-work successor omitted its work value"};
    rootWork = rootDomain->getNextWork(*cursor, problem.getRelationLimits());
  }
  ++work.rootWorkSuccessorSteps;
  if (rootWork.getKind() != RootWorkSuccessorKind::End) {
    const RootWorkDomainFailure *failure = rootWork.getFailure();
    if (!failure)
      return {SpatialExpansionKind::CompilerBug,
              "root-work successor omitted its typed failure"};
    if (rootWork.getKind() == RootWorkSuccessorKind::Indeterminate) {
      pausedSpatialChoice = std::move(choice);
      pausedChoiceIsProposal = proposalChoice;
      ++work.indeterminateSpatialChoices;
      return {SpatialExpansionKind::Indeterminate,
              getRootWorkFailureDetail(*failure)};
    }
    if (rootWork.getKind() == RootWorkSuccessorKind::Unsupported) {
      if (proposalChoice)
        resolvedProposalChoices.insert(std::move(choice));
      ++work.unsupportedSpatialChoices;
      return {SpatialExpansionKind::Unsupported,
              getRootWorkFailureDetail(*failure)};
    }
    return {SpatialExpansionKind::CompilerBug,
            getRootWorkFailureDetail(*failure)};
  }
  if (validatedRootWorks == 0)
    return {SpatialExpansionKind::CompilerBug,
            "spatial choice produced an empty root-work domain"};

  if (proposalChoice)
    resolvedProposalChoices.insert(choice);
  mlir::FailureOr<SpatialState> state =
      SpatialState::create(problem, std::move(choice), &detail);
  if (mlir::failed(state))
    return {SpatialExpansionKind::CompilerBug, std::move(detail)};
  if (!frontier.insert(*state).second)
    return {SpatialExpansionKind::CompilerBug,
            "resolved spatial choice produced a duplicate state"};
  ++work.spatialStatesQueued;
  return {SpatialExpansionKind::StateQueued};
}

SpatialExpansionResult PhysicalDataflowPlanningSession::resumeSpatial() {
  while (true) {
    if (pausedSpatialChoice) {
      ++work.spatialSuccessorSteps;
      return evaluateAndQueue(*pausedSpatialChoice, pausedChoiceIsProposal);
    }

    for (const SpatialPlan &proposal :
         problem.getSpatialDomain().getProposals()) {
      if (resolvedProposalChoices.count(proposal))
        continue;
      ++work.spatialSuccessorSteps;
      return evaluateAndQueue(proposal, /*proposalChoice=*/true);
    }

    SpatialPlan choice;
    if (canonicalCursor == CanonicalCursor::NotStarted) {
      choice = problem.getSpatialDomain().getFirstPlan();
      if (!problem.getSpatialDomain().contains(choice))
        return {SpatialExpansionKind::CompilerBug,
                "spatial canonical domain has no valid first choice"};
      lastCanonicalChoice = choice;
      canonicalCursor = CanonicalCursor::LastChoice;
    } else if (canonicalCursor == CanonicalCursor::LastChoice) {
      SpatialPlanSuccessor next =
          problem.getSpatialDomain().getNextPlan(*lastCanonicalChoice);
      if (next.kind == SpatialPlanSuccessorKind::Failure)
        return {SpatialExpansionKind::CompilerBug,
                next.failure ? next.failure->detail
                             : "spatial successor returned no failure detail"};
      if (next.kind == SpatialPlanSuccessorKind::End) {
        canonicalCursor = CanonicalCursor::Exhausted;
        lastCanonicalChoice.reset();
        return {SpatialExpansionKind::ParentExhausted};
      }
      if (!next.plan)
        return {SpatialExpansionKind::CompilerBug,
                "spatial successor returned no plan"};
      choice = std::move(*next.plan);
      lastCanonicalChoice = choice;
    } else {
      return {SpatialExpansionKind::ParentExhausted};
    }

    ++work.spatialSuccessorSteps;
    if (resolvedProposalChoices.count(choice)) {
      ++work.duplicateSpatialChoices;
      continue;
    }
    return evaluateAndQueue(std::move(choice), /*proposalChoice=*/false);
  }
}

std::optional<SpatialState>
PhysicalDataflowPlanningSession::takeNextSpatialState() {
  if (frontier.empty())
    return std::nullopt;
  auto first = frontier.begin();
  SpatialState state = *first;
  frontier.erase(first);
  return state;
}

mlir::FailureOr<RegionDomain *>
PhysicalDataflowPlanningSession::getOrCreateRegionDomain(
    const SpatialState &spatial, std::string *failureReason) {
  auto cached = regionDomainCache.find(spatial.getPlan());
  if (cached != regionDomainCache.end()) {
    if (!rootWorkCache.count(spatial.getPlan())) {
      if (failureReason)
        *failureReason = "region domain cache has no derived root work";
      return mlir::failure();
    }
    return &cached->second;
  }
  SpatialDomainEvaluation evaluation = problem.getSpatialDomain().evaluate(
      problem.getProgram().dag, spatial.getPlan(), problem.getRelationLimits());
  if (!evaluation.isSatisfied()) {
    if (failureReason) {
      if (evaluation.failure)
        *failureReason = evaluation.failure->detail;
      else if (evaluation.demand)
        *failureReason = getDemandDetail(*evaluation.demand);
      else
        *failureReason = "region transition has no spatial demand outcome";
    }
    return mlir::failure();
  }
  const analysis::ExactDemandProof *proof =
      analysis::getExactDemandProof(*evaluation.demand);
  if (!proof) {
    if (failureReason)
      *failureReason = "satisfied region transition has no demand proof";
    return mlir::failure();
  }
  std::string detail;
  auto rootDomain = RootWorkDomain::create(
      problem.getProgram().dag, *evaluation.assignment, *proof,
      problem.getProgram().availableTileIds, &detail);
  if (mlir::failed(rootDomain)) {
    if (failureReason)
      *failureReason = std::move(detail);
    return mlir::failure();
  }
  RootWorkCollectionOutcome collected =
      collectRootWorks(*rootDomain, problem.getRelationLimits());
  RootWorkCollection *rootWorks = getRootWorkCollection(collected);
  if (!rootWorks || rootWorks->works.empty()) {
    if (failureReason)
      *failureReason =
          rootWorks ? "region transition has an empty root-work domain"
                    : std::visit(
                          [](const auto &value) -> std::string {
                            using T = std::decay_t<decltype(value)>;
                            if constexpr (std::is_same_v<T, RootWorkCollection>)
                              return {};
                            else
                              return value.detail;
                          },
                          collected);
    return mlir::failure();
  }
  auto regionDomain = RegionDomain::create(rootWorks->works, &detail);
  if (mlir::failed(regionDomain)) {
    if (failureReason)
      *failureReason = std::move(detail);
    return mlir::failure();
  }
  auto [storedWorks, insertedWorks] =
      rootWorkCache.try_emplace(spatial.getPlan(), std::move(rootWorks->works));
  (void)storedWorks;
  auto [stored, inserted] = regionDomainCache.try_emplace(
      spatial.getPlan(), std::move(*regionDomain));
  if (!insertedWorks || !inserted) {
    if (failureReason)
      *failureReason = "region domain cache changed during construction";
    return mlir::failure();
  }
  return &stored->second;
}

PhysicalDataflowPlanningSession::TemporalDomainLookup
PhysicalDataflowPlanningSession::getOrCreateTemporalDomain(
    const RegionState &region) {
  auto cached = temporalDomainCache.find(region);
  if (cached != temporalDomainCache.end())
    return {&cached->second, {}};
  auto rootWorks = rootWorkCache.find(region.getSpatialPlan());
  if (rootWorks == rootWorkCache.end())
    return {nullptr, TemporalDomainFailure{
                         TemporalDomainFailureKind::BrokenContract,
                         {},
                         "temporal transition has no derived root work"}};
  TemporalDomainResult result =
      buildTemporalDomain(region.getRegionPlan(), rootWorks->second);
  if (!result.succeeded()) {
    if (result.failure)
      return {nullptr, std::move(result.failure)};
    return {nullptr, TemporalDomainFailure{
                         TemporalDomainFailureKind::BrokenContract,
                         {},
                         "temporal domain returned no failure detail"}};
  }
  auto [stored, inserted] =
      temporalDomainCache.try_emplace(region, std::move(*result.domain));
  if (!inserted)
    return {nullptr, TemporalDomainFailure{
                         TemporalDomainFailureKind::BrokenContract,
                         {},
                         "temporal domain cache changed during construction"}};
  return {&stored->second, {}};
}

PhysicalDataflowPlanningSession::RepresentationDomainLookup
PhysicalDataflowPlanningSession::getOrCreateRepresentationDomain(
    const TemporalState &temporal) {
  auto cached = representationDomainCache.find(temporal);
  if (cached != representationDomainCache.end())
    return {&cached->second, {}};
  auto rootWorks = rootWorkCache.find(temporal.getSpatialPlan());
  if (rootWorks == rootWorkCache.end())
    return {nullptr, RepresentationDomainFailure{
                         RepresentationDomainFailureKind::BrokenContract,
                         {},
                         "representation transition has no derived root work"}};
  CanonicalRepresentationPlanOutcome canonical =
      buildCanonicalRepresentationPlan(temporal.getRegionPlan(),
                                       temporal.getTemporalPlan(),
                                       rootWorks->second);
  const CanonicalRepresentationCoordinate *coordinate =
      getCanonicalRepresentationCoordinate(canonical);
  if (!coordinate) {
    if (const auto *unsupported =
            std::get_if<UnsupportedRepresentationPlan>(&canonical))
      return {nullptr,
              RepresentationDomainFailure{
                  RepresentationDomainFailureKind::UnsupportedSemantics,
                  {},
                  unsupported->detail}};
    return {nullptr, RepresentationDomainFailure{
                         RepresentationDomainFailureKind::BrokenContract,
                         {},
                         std::get<BrokenRepresentationPlan>(canonical).detail}};
  }
  RepresentationDomainResult result = buildRepresentationDomain(*coordinate);
  if (!result.succeeded()) {
    if (result.failure)
      return {nullptr, std::move(result.failure)};
    return {nullptr, RepresentationDomainFailure{
                         RepresentationDomainFailureKind::BrokenContract,
                         {},
                         "representation domain returned no typed outcome"}};
  }
  auto [stored, inserted] = representationDomainCache.try_emplace(
      temporal, std::move(*result.domain));
  if (!inserted)
    return {nullptr,
            RepresentationDomainFailure{
                RepresentationDomainFailureKind::BrokenContract,
                {},
                "representation domain cache changed during construction"}};
  return {&stored->second, {}};
}

PhysicalDataflowPlanningSession::MovementDomainLookup
PhysicalDataflowPlanningSession::getOrCreateMovementDomain(
    const RepresentationState &representations) {
  auto cached = movementDomainCache.find(representations);
  if (cached != movementDomainCache.end())
    return {&cached->second, {}};
  RepresentationDomainLookup representationDomain =
      getOrCreateRepresentationDomain(representations.getTemporalState());
  if (!representationDomain.domain)
    return {nullptr,
            MovementDomainFailure{
                MovementDomainFailureKind::BrokenContract,
                representationDomain.failure
                    ? representationDomain.failure->detail
                    : "movement transition lost its representation domain"}};
  auto rootWorks = rootWorkCache.find(representations.getSpatialPlan());
  if (rootWorks == rootWorkCache.end())
    return {nullptr, MovementDomainFailure{
                         MovementDomainFailureKind::BrokenContract,
                         "movement transition has no derived root work"}};

  CanonicalRepresentationCoordinate primary;
  for (const LogicalRepresentationPlan &logical :
       representations.getRepresentationPlan().logicalValues) {
    auto version =
        llvm::find_if(representations.getRepresentationPlan().physicalVersions,
                      [&](const PhysicalVersionPlan &candidate) {
                        return candidate.id == logical.primary;
                      });
    const RepresentationResourceDescription *resource =
        representationDomain.domain->findResource(logical.value);
    if (version ==
            representations.getRepresentationPlan().physicalVersions.end() ||
        !resource)
      return {nullptr, MovementDomainFailure{
                           MovementDomainFailureKind::BrokenContract,
                           "movement transition has an incomplete primary "
                           "representation"}};
    primary.plan.logicalValues.push_back(logical);
    primary.plan.physicalVersions.push_back(*version);
    RepresentationResourceDescription selected = *resource;
    selected.version = logical.primary;
    selected.encoding = version->encoding;
    primary.resources.push_back(std::move(selected));
  }
  llvm::sort(primary.plan.logicalValues);
  llvm::sort(primary.plan.physicalVersions);
  llvm::sort(primary.resources,
             [](const RepresentationResourceDescription &lhs,
                const RepresentationResourceDescription &rhs) {
               return lhs.version < rhs.version;
             });
  CanonicalMovementPlanOutcome canonical = buildCanonicalMovementPlan(
      representations.getRegionPlan(), primary, rootWorks->second);
  const CanonicalMovementCoordinate *coordinate =
      getCanonicalMovementCoordinate(canonical);
  if (!coordinate) {
    if (const auto *unsupported =
            std::get_if<UnsupportedMovementPlan>(&canonical))
      return {nullptr, MovementDomainFailure{
                           MovementDomainFailureKind::UnsupportedSemantics,
                           unsupported->detail}};
    return {nullptr, MovementDomainFailure{
                         MovementDomainFailureKind::BrokenContract,
                         std::get<BrokenMovementPlan>(canonical).detail}};
  }
  MovementDomainResult result =
      buildMovementDomain(*coordinate, representations.getRepresentationPlan(),
                          problem.getProgram().availableTileIds);
  if (!result.succeeded()) {
    if (result.failure)
      return {nullptr, std::move(result.failure)};
    return {nullptr,
            MovementDomainFailure{MovementDomainFailureKind::BrokenContract,
                                  "movement domain returned no typed outcome"}};
  }
  auto [stored, inserted] = movementDomainCache.try_emplace(
      representations, std::move(*result.domain));
  if (!inserted)
    return {nullptr, MovementDomainFailure{
                         MovementDomainFailureKind::BrokenContract,
                         "movement domain cache changed during construction"}};
  return {&stored->second, {}};
}

PhysicalDataflowPlanningSession::StorageDomainLookup
PhysicalDataflowPlanningSession::getOrCreateStorageDomain(
    const MovementState &movement) {
  auto cached = storageDomainCache.find(movement);
  if (cached != storageDomainCache.end())
    return {&cached->second, {}};
  RepresentationDomainLookup representationDomain =
      getOrCreateRepresentationDomain(
          movement.getRepresentationState().getTemporalState());
  MovementDomainLookup movementDomain =
      getOrCreateMovementDomain(movement.getRepresentationState());
  if (!representationDomain.domain || !movementDomain.domain)
    return {nullptr,
            StorageDomainFailure{StorageDomainFailureKind::BrokenContract,
                                 "storage transition lost an upstream domain"}};
  CanonicalRepresentationCoordinate representations;
  representations.plan = movement.getRepresentationPlan();
  for (const PhysicalVersionPlan &version :
       representations.plan.physicalVersions) {
    if (!version.id.derivation.empty())
      return {nullptr, StorageDomainFailure{
                           StorageDomainFailureKind::UnsupportedSemantics,
                           "pre-structure storage currently requires primary "
                           "physical versions"}};
    const RepresentationResourceDescription *resource =
        representationDomain.domain->findResource(version.id.logicalValue);
    if (!resource)
      return {nullptr,
              StorageDomainFailure{
                  StorageDomainFailureKind::BrokenContract,
                  "storage transition has a missing version resource"}};
    RepresentationResourceDescription selected = *resource;
    selected.version = version.id;
    selected.encoding = version.encoding;
    representations.resources.push_back(std::move(selected));
  }
  CanonicalMovementCoordinate movements;
  movements.plan = movement.getMovementPlan();
  movements.resources.assign(movementDomain.domain->getResources().begin(),
                             movementDomain.domain->getResources().end());
  CanonicalSerializedExecutionPlanOutcome serializedOutcome =
      buildCanonicalSerializedExecutionPlan(movement.getRegionPlan(),
                                            movement.getTemporalPlan());
  const SerializedExecutionPlan *serialized =
      getSerializedExecutionPlan(serializedOutcome);
  if (!serialized)
    return {
        nullptr,
        StorageDomainFailure{
            StorageDomainFailureKind::BrokenContract,
            std::get<BrokenSerializedExecutionPlan>(serializedOutcome).detail}};
  CanonicalStoragePlanOutcome canonical =
      buildCanonicalStoragePlan(representations, movements, *serialized);
  const CanonicalStorageCoordinate *coordinate =
      getCanonicalStorageCoordinate(canonical);
  if (!coordinate)
    return {nullptr, StorageDomainFailure{
                         StorageDomainFailureKind::BrokenContract,
                         std::get<BrokenStoragePlan>(canonical).detail}};
  StorageDomainResult result = buildStorageDomain(*coordinate);
  if (!result.succeeded()) {
    if (result.failure)
      return {nullptr, std::move(result.failure)};
    return {nullptr,
            StorageDomainFailure{StorageDomainFailureKind::BrokenContract,
                                 "storage domain returned no typed outcome"}};
  }
  auto [stored, inserted] =
      storageDomainCache.try_emplace(movement, std::move(*result.domain));
  if (!inserted)
    return {nullptr, StorageDomainFailure{
                         StorageDomainFailureKind::BrokenContract,
                         "storage domain cache changed during construction"}};
  return {&stored->second, {}};
}

PhysicalDataflowPlanningSession::EventGraphLookup
PhysicalDataflowPlanningSession::getOrCreateEventGraph(
    const InitialBufferState &buffers) {
  auto cached = eventGraphCache.find(buffers);
  if (cached != eventGraphCache.end())
    return {&cached->second, {}};
  StorageDomainLookup storageDomain =
      getOrCreateStorageDomain(buffers.getMovementState());
  MovementDomainLookup movementDomain = getOrCreateMovementDomain(
      buffers.getMovementState().getRepresentationState());
  if (!storageDomain.domain || !movementDomain.domain)
    return {nullptr, EventGraphFailure{
                         EventGraphFailureKind::Deferred,
                         EventGraphFailureReason::MissingPlanFact,
                         {},
                         "event graph transition lost an upstream domain"}};

  CanonicalSerializedExecutionPlanOutcome serializedOutcome =
      buildCanonicalSerializedExecutionPlan(buffers.getRegionPlan(),
                                            buffers.getTemporalPlan());
  const SerializedExecutionPlan *serialized =
      getSerializedExecutionPlan(serializedOutcome);
  if (!serialized)
    return {
        nullptr,
        EventGraphFailure{
            EventGraphFailureKind::Deferred,
            EventGraphFailureReason::MissingPlanFact,
            {},
            std::get<BrokenSerializedExecutionPlan>(serializedOutcome).detail}};

  CanonicalMovementCoordinate movement;
  movement.plan = buffers.getMovementPlan();
  movement.resources.assign(movementDomain.domain->getResources().begin(),
                            movementDomain.domain->getResources().end());
  EventGraphBuildResult result = buildEventGraph(
      problem.getCardId(), buffers.getRegionPlan(), buffers.getTemporalPlan(),
      *serialized, movement, storageDomain.domain->getCanonicalCoordinate(),
      buffers.getBufferPlan());
  if (!result.succeeded())
    return {nullptr,
            result.failure
                ? std::move(result.failure)
                : std::optional<EventGraphFailure>(EventGraphFailure{
                      EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      {},
                      "event graph builder returned no typed outcome"})};
  auto [stored, inserted] =
      eventGraphCache.try_emplace(buffers, std::move(*result.graph));
  if (!inserted)
    return {nullptr,
            EventGraphFailure{EventGraphFailureKind::CompilerBug,
                              EventGraphFailureReason::MalformedPlan,
                              {},
                              "event graph cache changed during construction"}};
  ++work.eventGraphsBuilt;
  return {&stored->second, {}};
}

PhysicalDataflowPlanningSession::ExecutionStructureDomainLookup
PhysicalDataflowPlanningSession::getOrCreateExecutionStructureDomain(
    const InitialBufferState &buffers, const EventGraph &eventGraph) {
  auto cached = executionStructureDomainCache.find(buffers);
  if (cached != executionStructureDomainCache.end())
    return {&cached->second, {}};
  const TemporalState &temporal =
      buffers.getMovementState().getRepresentationState().getTemporalState();
  TemporalDomainLookup temporalDomain =
      getOrCreateTemporalDomain(temporal.getRegionState());
  if (!temporalDomain.domain)
    return {nullptr, ExecutionStructureDomainFailure{
                         ExecutionStructureDomainFailureKind::BrokenContract,
                         temporalDomain.failure
                             ? temporalDomain.failure->detail
                             : "execution structure lost its temporal domain"}};
  ExecutionStructureDomainResult result = buildExecutionStructureDomain(
      eventGraph, temporalDomain.domain->getScopeDescriptors(),
      buffers.getTemporalPlan());
  if (!result.succeeded())
    return {
        nullptr,
        result.failure
            ? std::move(result.failure)
            : std::optional<ExecutionStructureDomainFailure>(
                  ExecutionStructureDomainFailure{
                      ExecutionStructureDomainFailureKind::BrokenContract,
                      "execution-structure domain returned no typed outcome"})};
  auto [stored, inserted] = executionStructureDomainCache.try_emplace(
      buffers, std::move(*result.domain));
  if (!inserted)
    return {nullptr,
            ExecutionStructureDomainFailure{
                ExecutionStructureDomainFailureKind::BrokenContract,
                "execution-structure cache changed during construction"}};
  return {&stored->second, {}};
}

PhysicalDataflowPlanningSession::StructureSpecificStorageDomainLookup
PhysicalDataflowPlanningSession::getOrCreateStructureSpecificStorage(
    const ExecutionStructureState &structure, const EventGraph &eventGraph) {
  auto cached = structureSpecificStorageDomainCache.find(structure);
  if (cached != structureSpecificStorageDomainCache.end())
    return {&cached->second, {}};
  StructureSpecificStorageDomainResult result =
      buildStructureSpecificStorageDomain(structure.getExecutionStructurePlan(),
                                          structure.getInitialBufferPlan(),
                                          eventGraph);
  if (!result.succeeded())
    return {
        nullptr,
        result.failure
            ? std::move(result.failure)
            : std::optional<StructureSpecificStorageFailure>(
                  StructureSpecificStorageFailure{
                      StructureSpecificStorageFailureKind::BrokenContract,
                      "structure-specific storage returned no typed outcome"})};
  auto [stored, inserted] = structureSpecificStorageDomainCache.try_emplace(
      structure, std::move(*result.domain));
  if (!inserted)
    return {
        nullptr,
        StructureSpecificStorageFailure{
            StructureSpecificStorageFailureKind::BrokenContract,
            "structure-specific storage cache changed during construction"}};
  return {&stored->second, {}};
}

PhysicalDataflowPlanningSession::ScheduleDomainLookup
PhysicalDataflowPlanningSession::getOrCreateScheduleDomain(
    const BufferState &buffers, const EventGraph &eventGraph) {
  auto cached = scheduleDomainCache.find(buffers);
  if (cached != scheduleDomainCache.end())
    return {&cached->second, {}};
  StructureSpecificStorageDomainLookup fixedStorage =
      getOrCreateStructureSpecificStorage(buffers.getExecutionStructureState(),
                                          eventGraph);
  if (!fixedStorage.domain)
    return {nullptr,
            ScheduleDomainFailure{
                ScheduleDomainFailureKind::BrokenContract,
                fixedStorage.failure
                    ? fixedStorage.failure->detail
                    : "schedule transition lost fixed-structure storage"}};
  ScheduleDomainInput input;
  input.structure = buffers.getExecutionStructurePlan();
  input.buffers = buffers.getBufferPlan();
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
  input.slotLifetimes.assign(
      fixedStorage.domain->getLifetimeRequirements().begin(),
      fixedStorage.domain->getLifetimeRequirements().end());
  ScheduleDomainResult result = buildScheduleDomain(std::move(input));
  if (!result.succeeded())
    return {nullptr,
            result.failure
                ? std::move(result.failure)
                : std::optional<ScheduleDomainFailure>(ScheduleDomainFailure{
                      ScheduleDomainFailureKind::BrokenContract,
                      "schedule domain returned no typed outcome"})};
  auto [stored, inserted] =
      scheduleDomainCache.try_emplace(buffers, std::move(*result.domain));
  if (!inserted)
    return {nullptr, ScheduleDomainFailure{
                         ScheduleDomainFailureKind::BrokenContract,
                         "schedule domain cache changed during construction"}};
  return {&stored->second, {}};
}

mlir::FailureOr<std::optional<RegionState>>
PhysicalDataflowPlanningSession::resumeRegion(RegionContinuation &continuation,
                                              std::string *failureReason) {
  if (continuation.exhausted)
    return std::optional<RegionState>{};
  mlir::FailureOr<RegionDomain *> regionDomain =
      getOrCreateRegionDomain(continuation.parent, failureReason);
  if (mlir::failed(regionDomain))
    return mlir::failure();
  ++work.regionSuccessorSteps;
  RegionSuccessor next =
      continuation.started ? (*regionDomain)->getNextPlan(*continuation.cursor)
                           : (*regionDomain)->getFirstPlan();
  if (next.getKind() == RegionSuccessorKind::End) {
    continuation.exhausted = true;
    continuation.cursor.reset();
    return std::optional<RegionState>{};
  }
  if (next.getKind() != RegionSuccessorKind::Plan || !next.getPlan() ||
      !next.getCursor()) {
    if (failureReason)
      *failureReason = next.getDetail().empty()
                           ? "region successor omitted its plan or cursor"
                           : next.getDetail().str();
    return mlir::failure();
  }
  std::string detail;
  auto state = RegionState::create(**regionDomain, continuation.parent,
                                   *next.getPlan(), &detail);
  if (mlir::failed(state)) {
    if (failureReason)
      *failureReason = std::move(detail);
    return mlir::failure();
  }
  continuation.cursor = *next.getCursor();
  continuation.started = true;
  ++work.regionStatesQueued;
  return std::optional<RegionState>(std::move(*state));
}

TemporalExpansionResult PhysicalDataflowPlanningSession::resumeTemporal(
    TemporalContinuation &continuation) {
  if (continuation.exhausted)
    return {TemporalExpansionKind::ParentExhausted};
  TemporalDomainLookup lookup = getOrCreateTemporalDomain(continuation.parent);
  if (!lookup.domain) {
    if (!lookup.failure)
      return {TemporalExpansionKind::CompilerBug,
              {},
              "temporal domain lookup returned no typed outcome"};
    switch (lookup.failure->kind) {
    case TemporalDomainFailureKind::UnsupportedSemantics:
      ++work.unsupportedTemporalChoices;
      return {TemporalExpansionKind::Unsupported,
              {},
              std::move(lookup.failure->detail)};
    case TemporalDomainFailureKind::Indeterminate:
      ++work.indeterminateTemporalChoices;
      return {TemporalExpansionKind::Indeterminate,
              {},
              std::move(lookup.failure->detail)};
    case TemporalDomainFailureKind::BrokenContract:
      return {TemporalExpansionKind::CompilerBug,
              {},
              std::move(lookup.failure->detail)};
    }
    return {TemporalExpansionKind::CompilerBug,
            {},
            "temporal domain failure has an invalid category"};
  }
  ++work.temporalSuccessorSteps;
  TemporalSuccessor next =
      continuation.started ? lookup.domain->getNextPlan(*continuation.cursor)
                           : lookup.domain->getFirstPlan();
  if (next.getKind() == TemporalSuccessorKind::End) {
    continuation.exhausted = true;
    continuation.cursor.reset();
    return {TemporalExpansionKind::ParentExhausted};
  }
  if (next.getKind() == TemporalSuccessorKind::Unsupported) {
    ++work.unsupportedTemporalChoices;
    return {TemporalExpansionKind::Unsupported, {}, next.getDetail().str()};
  }
  if (next.getKind() == TemporalSuccessorKind::Indeterminate) {
    ++work.indeterminateTemporalChoices;
    return {TemporalExpansionKind::Indeterminate, {}, next.getDetail().str()};
  }
  if (next.getKind() != TemporalSuccessorKind::Plan || !next.getPlan() ||
      !next.getCursor())
    return {TemporalExpansionKind::CompilerBug,
            {},
            next.getDetail().empty()
                ? "temporal successor omitted its plan or cursor"
                : next.getDetail().str()};
  std::string detail;
  auto state = TemporalState::create(*lookup.domain, continuation.parent,
                                     *next.getPlan(), &detail);
  if (mlir::failed(state))
    return {TemporalExpansionKind::CompilerBug, {}, std::move(detail)};
  continuation.cursor = *next.getCursor();
  continuation.started = true;
  ++work.temporalStatesQueued;
  return {TemporalExpansionKind::State, std::move(*state)};
}

RepresentationExpansionResult
PhysicalDataflowPlanningSession::resumeRepresentation(
    RepresentationContinuation &continuation) {
  if (continuation.exhausted)
    return {RepresentationExpansionKind::ParentExhausted};
  RepresentationDomainLookup lookup =
      getOrCreateRepresentationDomain(continuation.parent);
  if (!lookup.domain) {
    if (!lookup.failure)
      return {RepresentationExpansionKind::CompilerBug,
              {},
              "representation domain lookup returned no typed outcome"};
    if (lookup.failure->kind ==
        RepresentationDomainFailureKind::UnsupportedSemantics) {
      ++work.unsupportedRepresentationChoices;
      return {RepresentationExpansionKind::Unsupported,
              {},
              std::move(lookup.failure->detail)};
    }
    return {RepresentationExpansionKind::CompilerBug,
            {},
            std::move(lookup.failure->detail)};
  }
  ++work.representationSuccessorSteps;
  RepresentationSuccessor next =
      continuation.started ? lookup.domain->getNextPlan(*continuation.cursor)
                           : lookup.domain->getFirstPlan();
  if (next.getKind() == RepresentationSuccessorKind::End) {
    continuation.exhausted = true;
    continuation.cursor.reset();
    return {RepresentationExpansionKind::ParentExhausted};
  }
  if (next.getKind() != RepresentationSuccessorKind::Plan || !next.getPlan() ||
      !next.getCursor())
    return {RepresentationExpansionKind::CompilerBug,
            {},
            next.getDetail().empty()
                ? "representation successor omitted its plan or cursor"
                : next.getDetail().str()};
  std::string detail;
  auto state = RepresentationState::create(*lookup.domain, continuation.parent,
                                           *next.getPlan(), &detail);
  if (mlir::failed(state))
    return {RepresentationExpansionKind::CompilerBug, {}, std::move(detail)};
  continuation.cursor = *next.getCursor();
  continuation.started = true;
  ++work.representationStatesQueued;
  return {RepresentationExpansionKind::State, std::move(*state)};
}

MovementExpansionResult PhysicalDataflowPlanningSession::resumeMovement(
    MovementContinuation &continuation) {
  if (continuation.exhausted)
    return {MovementExpansionKind::ParentExhausted};
  MovementDomainLookup lookup = getOrCreateMovementDomain(continuation.parent);
  if (!lookup.domain) {
    if (!lookup.failure)
      return {MovementExpansionKind::CompilerBug,
              {},
              "movement domain lookup returned no typed outcome"};
    if (lookup.failure->kind ==
        MovementDomainFailureKind::UnsupportedSemantics) {
      ++work.unsupportedMovementChoices;
      return {MovementExpansionKind::Unsupported,
              {},
              std::move(lookup.failure->detail)};
    }
    return {MovementExpansionKind::CompilerBug,
            {},
            std::move(lookup.failure->detail)};
  }
  ++work.movementSuccessorSteps;
  MovementSuccessor next =
      continuation.started ? lookup.domain->getNextPlan(*continuation.cursor)
                           : lookup.domain->getFirstPlan();
  if (next.getKind() == MovementSuccessorKind::End) {
    continuation.exhausted = true;
    continuation.cursor.reset();
    return {MovementExpansionKind::ParentExhausted};
  }
  if (next.getKind() != MovementSuccessorKind::Plan || !next.getPlan() ||
      !next.getCursor())
    return {MovementExpansionKind::CompilerBug,
            {},
            next.getDetail().empty()
                ? "movement successor omitted its plan or cursor"
                : next.getDetail().str()};
  std::string detail;
  auto state = MovementState::create(*lookup.domain, continuation.parent,
                                     *next.getPlan(), &detail);
  if (mlir::failed(state))
    return {MovementExpansionKind::CompilerBug, {}, std::move(detail)};
  continuation.cursor = *next.getCursor();
  continuation.started = true;
  ++work.movementStatesQueued;
  return {MovementExpansionKind::State, std::move(*state)};
}

StorageExpansionResult PhysicalDataflowPlanningSession::resumeStorage(
    StorageContinuation &continuation) {
  if (continuation.exhausted)
    return {StorageExpansionKind::ParentExhausted};
  StorageDomainLookup lookup = getOrCreateStorageDomain(continuation.parent);
  if (!lookup.domain) {
    if (!lookup.failure)
      return {StorageExpansionKind::CompilerBug,
              {},
              "storage domain lookup returned no typed outcome"};
    if (lookup.failure->kind ==
        StorageDomainFailureKind::UnsupportedSemantics) {
      ++work.unsupportedStorageChoices;
      return {StorageExpansionKind::Unsupported,
              {},
              std::move(lookup.failure->detail)};
    }
    return {StorageExpansionKind::CompilerBug,
            {},
            std::move(lookup.failure->detail)};
  }
  ++work.storageSuccessorSteps;
  StorageSuccessor next = continuation.started
                              ? lookup.domain->getNextPlan(*continuation.cursor)
                              : lookup.domain->getFirstPlan();
  if (next.getKind() == StorageSuccessorKind::End) {
    continuation.exhausted = true;
    continuation.cursor.reset();
    return {StorageExpansionKind::ParentExhausted};
  }
  if (next.getKind() != StorageSuccessorKind::Plan || !next.getPlan() ||
      !next.getCursor())
    return {StorageExpansionKind::CompilerBug,
            {},
            next.getDetail().empty()
                ? "storage successor omitted its plan or cursor"
                : next.getDetail().str()};
  std::string detail;
  auto state = InitialBufferState::create(*lookup.domain, continuation.parent,
                                          *next.getPlan(), &detail);
  if (mlir::failed(state))
    return {StorageExpansionKind::CompilerBug, {}, std::move(detail)};
  continuation.cursor = *next.getCursor();
  continuation.started = true;
  ++work.storageStatesQueued;
  return {StorageExpansionKind::State, std::move(*state)};
}

mlir::FailureOr<IncompletePlanningDomain>
PhysicalDataflowPlanningSession::getFirstIncompleteState(
    std::string *failureReason) {
  while (frontier.empty()) {
    SpatialExpansionResult expansion = resumeSpatial();
    switch (expansion.getKind()) {
    case SpatialExpansionKind::StateQueued:
      break;
    case SpatialExpansionKind::Unsupported:
      if (failureReason)
        *failureReason = expansion.getDetail().str();
      return mlir::failure();
    case SpatialExpansionKind::Indeterminate:
      if (failureReason)
        *failureReason = expansion.getDetail().str();
      return mlir::failure();
    case SpatialExpansionKind::ParentExhausted:
      if (failureReason)
        *failureReason = "spatial planning domain has no supported state";
      return mlir::failure();
    case SpatialExpansionKind::CompilerBug:
      if (failureReason)
        *failureReason = expansion.getDetail().str();
      return mlir::failure();
    }
  }
  std::optional<SpatialState> state = takeNextSpatialState();
  if (!state) {
    if (failureReason)
      *failureReason = "spatial frontier lost a queued state";
    return mlir::failure();
  }
  RegionContinuation continuation = createRegionContinuation(std::move(*state));
  mlir::FailureOr<std::optional<RegionState>> region =
      resumeRegion(continuation, failureReason);
  if (mlir::failed(region) || !*region)
    return mlir::failure();
  TemporalContinuation temporalContinuation =
      createTemporalContinuation(std::move(**region));
  TemporalExpansionResult temporal = resumeTemporal(temporalContinuation);
  if (temporal.getKind() != TemporalExpansionKind::State) {
    if (failureReason)
      *failureReason = temporal.getDetail().empty()
                           ? "temporal planning domain has no supported state"
                           : temporal.getDetail().str();
    return mlir::failure();
  }
  std::optional<TemporalState> temporalState = temporal.takeState();
  if (!temporalState) {
    if (failureReason)
      *failureReason = "temporal state result lost its value";
    return mlir::failure();
  }
  TemporalDomainLookup temporalDomain =
      getOrCreateTemporalDomain(temporalState->getRegionState());
  if (!temporalDomain.domain) {
    if (failureReason)
      *failureReason = temporalDomain.failure
                           ? temporalDomain.failure->detail
                           : "structural readiness lost its temporal domain";
    return mlir::failure();
  }
  ++work.structuralReadinessQueries;
  StructuralReadinessResult readiness = checkStructuralReadiness(
      *temporalDomain.domain, temporalState->getTemporalPlan());
  if (readiness.getKind() != StructuralReadinessKind::ReadyForNextCoordinate ||
      readiness.getRequiredCoordinate() !=
          RequiredPlanningCoordinate::Representation) {
    if (failureReason)
      *failureReason = readiness.getDetail().empty()
                           ? "closed prefix failed structural readiness"
                           : readiness.getDetail().str();
    return mlir::failure();
  }
  RepresentationContinuation representationContinuation =
      createRepresentationContinuation(std::move(*temporalState));
  RepresentationExpansionResult representation =
      resumeRepresentation(representationContinuation);
  if (representation.getKind() != RepresentationExpansionKind::State) {
    if (failureReason)
      *failureReason =
          representation.getDetail().empty()
              ? "representation planning domain has no supported state"
              : representation.getDetail().str();
    return mlir::failure();
  }
  std::optional<RepresentationState> representationState =
      representation.takeState();
  if (!representationState) {
    if (failureReason)
      *failureReason = "representation state result lost its value";
    return mlir::failure();
  }
  MovementContinuation movementContinuation =
      createMovementContinuation(std::move(*representationState));
  MovementExpansionResult movement = resumeMovement(movementContinuation);
  if (movement.getKind() != MovementExpansionKind::State) {
    if (failureReason)
      *failureReason = movement.getDetail().empty()
                           ? "movement planning domain has no supported state"
                           : movement.getDetail().str();
    return mlir::failure();
  }
  std::optional<MovementState> movementState = movement.takeState();
  if (!movementState) {
    if (failureReason)
      *failureReason = "movement state result lost its value";
    return mlir::failure();
  }
  StorageContinuation storageContinuation =
      createStorageContinuation(std::move(*movementState));
  StorageExpansionResult storage = resumeStorage(storageContinuation);
  if (storage.getKind() != StorageExpansionKind::State) {
    if (failureReason)
      *failureReason = storage.getDetail().empty()
                           ? "storage planning domain has no supported state"
                           : storage.getDetail().str();
    return mlir::failure();
  }
  std::optional<InitialBufferState> storageState = storage.takeState();
  if (!storageState) {
    if (failureReason)
      *failureReason = "storage state result lost its value";
    return mlir::failure();
  }
  ++work.eventGraphQueries;
  EventGraphLookup eventGraph = getOrCreateEventGraph(*storageState);
  if (!eventGraph.graph) {
    if (failureReason)
      *failureReason = eventGraph.failure
                           ? eventGraph.failure->detail
                           : "event graph transition returned no typed outcome";
    return mlir::failure();
  }
  ++work.executionStructureQueries;
  ExecutionStructureDomainLookup structureDomain =
      getOrCreateExecutionStructureDomain(*storageState, *eventGraph.graph);
  if (!structureDomain.domain) {
    if (failureReason)
      *failureReason =
          structureDomain.failure
              ? structureDomain.failure->detail
              : "execution-structure transition returned no typed outcome";
    return mlir::failure();
  }
  ExecutionStructureSuccessor structure =
      structureDomain.domain->getFirstPlan();
  if (structure.getKind() != ExecutionStructureSuccessorKind::Plan ||
      !structure.getPlan()) {
    if (failureReason)
      *failureReason = structure.getDetail().empty()
                           ? "execution-structure domain has no first plan"
                           : structure.getDetail().str();
    return mlir::failure();
  }
  std::string detail;
  mlir::FailureOr<ExecutionStructureState> structureState =
      ExecutionStructureState::create(*structureDomain.domain,
                                      std::move(*storageState),
                                      *structure.getPlan(), &detail);
  if (mlir::failed(structureState)) {
    if (failureReason)
      *failureReason = std::move(detail);
    return mlir::failure();
  }
  ++work.executionStructureStatesQueued;
  ++work.structureSpecificStorageQueries;
  StructureSpecificStorageDomainLookup fixedStorage =
      getOrCreateStructureSpecificStorage(*structureState, *eventGraph.graph);
  if (!fixedStorage.domain) {
    if (failureReason)
      *failureReason =
          fixedStorage.failure
              ? fixedStorage.failure->detail
              : "structure-specific storage returned no typed outcome";
    return mlir::failure();
  }
  StructureSpecificStorageSuccessor buffers =
      fixedStorage.domain->getFirstPlan();
  if (buffers.getKind() != StructureSpecificStorageSuccessorKind::Plan ||
      !buffers.getPlan()) {
    if (failureReason)
      *failureReason = buffers.getDetail().empty()
                           ? "structure-specific storage has no first plan"
                           : buffers.getDetail().str();
    return mlir::failure();
  }
  mlir::FailureOr<BufferState> bufferState =
      BufferState::create(*fixedStorage.domain, std::move(*structureState),
                          *buffers.getPlan(), &detail);
  if (mlir::failed(bufferState)) {
    if (failureReason)
      *failureReason = std::move(detail);
    return mlir::failure();
  }
  ++work.structureSpecificStorageStatesQueued;
  ++work.scheduleQueries;
  ScheduleDomainLookup schedule =
      getOrCreateScheduleDomain(*bufferState, *eventGraph.graph);
  if (!schedule.domain) {
    if (failureReason)
      *failureReason = schedule.failure
                           ? schedule.failure->detail
                           : "schedule transition returned no typed outcome";
    return mlir::failure();
  }
  ScheduleSuccessor scheduled = schedule.domain->getFirstPlan();
  if (scheduled.getKind() != ScheduleSuccessorKind::Plan ||
      !scheduled.getPlan()) {
    if (failureReason)
      *failureReason = scheduled.getDetail().empty()
                           ? "schedule domain has no first plan"
                           : scheduled.getDetail().str();
    return mlir::failure();
  }
  mlir::FailureOr<ScheduledState> scheduledState = ScheduledState::create(
      *schedule.domain, std::move(*bufferState), *scheduled.getPlan(), &detail);
  if (mlir::failed(scheduledState)) {
    if (failureReason)
      *failureReason = std::move(detail);
    return mlir::failure();
  }
  ++work.scheduleStatesQueued;
  return IncompletePlanningDomain(std::move(*scheduledState), *eventGraph.graph,
                                  RequiredPlanningCoordinate::FullFeasibility,
                                  hasRemainingSpatialWork(), work);
}

bool PhysicalDataflowPlanningSession::hasRemainingSpatialWork() const {
  return canonicalCursor != CanonicalCursor::Exhausted ||
         pausedSpatialChoice.has_value() || !frontier.empty();
}

} // namespace wafer::compiler::detail
