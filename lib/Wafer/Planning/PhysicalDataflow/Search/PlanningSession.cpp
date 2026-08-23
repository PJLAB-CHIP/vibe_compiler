//===- PlanningSession.cpp - Spatial planning frontier -----------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningSession.h"

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
  return IncompletePlanningDomain(std::move(*state), hasRemainingSpatialWork(),
                                  work);
}

bool PhysicalDataflowPlanningSession::hasRemainingSpatialWork() const {
  return canonicalCursor != CanonicalCursor::Exhausted ||
         pausedSpatialChoice.has_value() || !frontier.empty();
}

} // namespace wafer::compiler::detail
