//===- PlanningSession.cpp - Structural planning frontier -------------===//

#include "Wafer/Planning/PhysicalDataflow/PlanningSession.h"

#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/STLExtras.h"

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
  SpatialDomainEvaluation evaluation = [&]() {
    wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                   "evaluate-spatial-demand");
    return problem.getSpatialDomain().evaluate(problem.getProgram().dag, choice,
                                               problem.getRelationLimits());
  }();
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
  std::vector<analysis::RootRegionWork> rootWorks;
  {
    wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                   "validate-root-work-domain");
    auto rootDomain = RootWorkDomain::create(
        problem.getProgram().dag, *evaluation.assignment, *proof,
        problem.getProgram().availableTileIds, &detail);
    if (mlir::failed(rootDomain))
      return {SpatialExpansionKind::CompilerBug, std::move(detail)};
    RootWorkSuccessor current =
        rootDomain->getFirstWork(problem.getRelationLimits());
    while (current.getKind() == RootWorkSuccessorKind::Work) {
      ++work.rootWorkSuccessorSteps;
      ++work.rootWorksValidated;
      const analysis::RootRegionWork *value = current.getWork();
      const RootWorkCursor *cursor = current.getCursor();
      if (!value || !cursor)
        return {SpatialExpansionKind::CompilerBug,
                "root-work successor omitted its work value"};
      rootWorks.push_back(*value);
      current = rootDomain->getNextWork(*cursor, problem.getRelationLimits());
    }
    ++work.rootWorkSuccessorSteps;
    if (current.getKind() != RootWorkSuccessorKind::End) {
      const RootWorkDomainFailure *failure = current.getFailure();
      if (!failure)
        return {SpatialExpansionKind::CompilerBug,
                "root-work successor omitted its typed failure"};
      if (current.getKind() == RootWorkSuccessorKind::Indeterminate) {
        pausedSpatialChoice = std::move(choice);
        pausedChoiceIsProposal = proposalChoice;
        ++work.indeterminateSpatialChoices;
        return {SpatialExpansionKind::Indeterminate,
                getRootWorkFailureDetail(*failure)};
      }
      if (current.getKind() == RootWorkSuccessorKind::Unsupported) {
        if (proposalChoice)
          resolvedProposalChoices.insert(std::move(choice));
        ++work.unsupportedSpatialChoices;
        return {SpatialExpansionKind::Unsupported,
                getRootWorkFailureDetail(*failure)};
      }
      return {SpatialExpansionKind::CompilerBug,
              getRootWorkFailureDetail(*failure)};
    }
  }
  if (rootWorks.empty())
    return {SpatialExpansionKind::CompilerBug,
            "spatial choice produced an empty root-work domain"};

  auto regionDomain = [&]() {
    wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                   "build-region-domain");
    return RegionDomain::create(rootWorks, &detail);
  }();
  if (mlir::failed(regionDomain))
    return {SpatialExpansionKind::CompilerBug, std::move(detail)};

  if (proposalChoice)
    resolvedProposalChoices.insert(choice);
  auto state = SpatialState::create(problem, std::move(choice), &detail);
  if (mlir::failed(state))
    return {SpatialExpansionKind::CompilerBug, std::move(detail)};
  auto [storedWorks, insertedWorks] =
      rootWorkCache.try_emplace(state->getPlan(), std::move(rootWorks));
  auto [storedRegion, insertedRegion] =
      regionDomainCache.try_emplace(state->getPlan(), std::move(*regionDomain));
  (void)storedWorks;
  (void)storedRegion;
  if (!insertedWorks || !insertedRegion)
    return {SpatialExpansionKind::CompilerBug,
            "resolved spatial choice changed its session memo"};
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

    auto proposals = spatialProposalCache.find(0);
    if (proposals == spatialProposalCache.end()) {
      auto generated = [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "physical-search", "build-spatial-proposals");
        return problem.getSpatialDomain().getGraphCoherentProposals(
            problem.getProgram().dag, problem.getRelationLimits());
      }();
      if (mlir::failed(generated))
        return {SpatialExpansionKind::CompilerBug,
                "graph-coherent spatial proposal construction failed"};
      std::vector<SpatialPlan> values(generated->begin(), generated->end());
      proposals = spatialProposalCache.try_emplace(0, std::move(values)).first;
    }
    for (const SpatialPlan &proposal : proposals->second) {
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

mlir::FailureOr<llvm::ArrayRef<analysis::RootRegionWork>>
PhysicalDataflowPlanningSession::getCurrentRootWorks(
    const SpatialState &spatial, std::string *failureReason) {
  auto found = rootWorkCache.find(spatial.getPlan());
  if (found == rootWorkCache.end()) {
    if (failureReason)
      *failureReason =
          "current SpatialState has no session-owned exact root work";
    return mlir::failure();
  }
  return llvm::ArrayRef<analysis::RootRegionWork>(found->second);
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

mlir::FailureOr<std::optional<RegionState>>
PhysicalDataflowPlanningSession::resumeRegion(RegionContinuation &continuation,
                                              std::string *failureReason) {
  if (continuation.exhausted)
    return std::optional<RegionState>{};
  auto regionDomain =
      getOrCreateRegionDomain(continuation.parent, failureReason);
  if (mlir::failed(regionDomain))
    return mlir::failure();
  if (!continuation.proposalsInitialized) {
    continuation.proposals = [&]() {
      wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                     "build-region-proposals");
      return (*regionDomain)->getProposals(maximumRegionProposals);
    }();
    continuation.proposalsInitialized = true;
  }
  auto makeState = [&](const RegionPlan &plan)
      -> mlir::FailureOr<std::optional<RegionState>> {
    std::string detail;
    auto state =
        RegionState::create(**regionDomain, continuation.parent, plan, &detail);
    if (mlir::failed(state)) {
      if (failureReason)
        *failureReason = std::move(detail);
      return mlir::failure();
    }
    ++work.regionStatesQueued;
    return std::optional<RegionState>(std::move(*state));
  };
  while (continuation.nextProposal < continuation.proposals.size()) {
    ++work.regionSuccessorSteps;
    const RegionPlan &proposal =
        continuation.proposals[continuation.nextProposal++];
    if (!continuation.emitted.insert(proposal).second)
      continue;
    return makeState(proposal);
  }

  while (true) {
    ++work.regionSuccessorSteps;
    RegionSuccessor next =
        continuation.rawStarted
            ? (*regionDomain)->getNextPlan(*continuation.cursor)
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
    continuation.cursor = *next.getCursor();
    continuation.rawStarted = true;
    if (!continuation.emitted.insert(*next.getPlan()).second)
      continue;
    return makeState(*next.getPlan());
  }
}

bool PhysicalDataflowPlanningSession::hasRemainingSpatialWork() const {
  return canonicalCursor != CanonicalCursor::Exhausted ||
         pausedSpatialChoice.has_value() || !frontier.empty();
}

} // namespace wafer::compiler::detail
