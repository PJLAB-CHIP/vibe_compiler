//===- PlanningSession.cpp - Spatial planning frontier -----------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningSession.h"

#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"

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
  return IncompletePlanningDomain(std::move(*temporalState),
                                  hasRemainingSpatialWork(), work);
}

bool PhysicalDataflowPlanningSession::hasRemainingSpatialWork() const {
  return canonicalCursor != CanonicalCursor::Exhausted ||
         pausedSpatialChoice.has_value() || !frontier.empty();
}

} // namespace wafer::compiler::detail
