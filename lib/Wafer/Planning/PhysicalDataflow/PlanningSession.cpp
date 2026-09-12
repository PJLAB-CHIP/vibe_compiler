//===- PlanningSession.cpp - Structural planning frontier -------------===//

#include "Wafer/Planning/PhysicalDataflow/PlanningSession.h"

#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

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

void recordRegionPlanMetrics(const RegionDomain &domain,
                             llvm::ArrayRef<RegionPlan> plans,
                             llvm::StringRef prefix) {
  wafer::support::addCompileCounter(
      "search", llvm::formatv("{0}-count", prefix).str(), plans.size());
  for (auto [index, plan] : llvm::enumerate(plans)) {
    auto counter = [&](llvm::StringRef metric, uint64_t value) {
      wafer::support::addCompileCounter(
          "search", llvm::formatv("{0}-{1}-{2}", prefix, index, metric).str(),
          value);
    };
    RegionProposalMetrics metrics = domain.getProposalMetrics(plan);
    counter("merges", metrics.fusionMerges);
    counter("regions", metrics.regions);
    counter("maximum-roots", metrics.maximumRootsPerRegion);
    counter("local-bindings", metrics.localBindings);
    counter("external-bindings", metrics.externalBindings);
    counter("exact-bytes-known", metrics.exactLogicalBytesKnown);
    counter("exact-bytes", metrics.exactLogicalBytes);
    uint64_t roots1 = 0;
    uint64_t roots2 = 0;
    uint64_t roots3To4 = 0;
    uint64_t roots5To8 = 0;
    uint64_t roots9Plus = 0;
    for (const RegionGroupPlan &group : plan.groups) {
      const size_t roots = group.mandatoryRoots.size();
      roots1 += roots == 1;
      roots2 += roots == 2;
      roots3To4 += roots >= 3 && roots <= 4;
      roots5To8 += roots >= 5 && roots <= 8;
      roots9Plus += roots >= 9;
    }
    counter("roots-1-regions", roots1);
    counter("roots-2-regions", roots2);
    counter("roots-3-to-4-regions", roots3To4);
    counter("roots-5-to-8-regions", roots5To8);
    counter("roots-9-plus-regions", roots9Plus);
  }
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
PhysicalDataflowPlanningSession::evaluateAndQueue(SpatialPlan choice) {
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
      resolvedChoices.insert(std::move(choice));
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
    ++work.indeterminateSpatialChoices;
    return {SpatialExpansionKind::Indeterminate,
            getDemandDetail(*evaluation.demand)};
  }
  pausedSpatialChoice.reset();
  if (outcome == SpatialChoiceOutcomeKind::Unsupported) {
    resolvedChoices.insert(std::move(choice));
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
        ++work.indeterminateSpatialChoices;
        return {SpatialExpansionKind::Indeterminate,
                getRootWorkFailureDetail(*failure)};
      }
      if (current.getKind() == RootWorkSuccessorKind::Unsupported) {
        resolvedChoices.insert(std::move(choice));
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

  resolvedChoices.insert(choice);
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
      return evaluateAndQueue(*pausedSpatialChoice);
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
      wafer::support::addCompileCounter("search", "spatial-proposals",
                                        values.size());
      proposals = spatialProposalCache.try_emplace(0, std::move(values)).first;
    }
    for (const SpatialPlan &proposal : proposals->second) {
      if (resolvedChoices.count(proposal))
        continue;
      ++work.spatialSuccessorSteps;
      return evaluateAndQueue(proposal);
    }

    if (isSpatialExhausted())
      return {SpatialExpansionKind::ParentExhausted};
    bool axis = takeAxis;
    takeAxis = !takeAxis;
    if (axis && axisCursor == CanonicalCursor::Exhausted)
      axis = false;
    if (!axis && canonicalCursor == CanonicalCursor::Exhausted)
      axis = true;
    CanonicalCursor &cursor = axis ? axisCursor : canonicalCursor;
    auto &last = axis ? lastAxisChoice : lastCanonicalChoice;
    SpatialPlan choice;
    if (cursor == CanonicalCursor::NotStarted) {
      choice = problem.getSpatialDomain().getFirstPlan();
      if (!problem.getSpatialDomain().contains(choice))
        return {SpatialExpansionKind::CompilerBug,
                "spatial domain has no valid first choice"};
      last = choice;
      cursor = CanonicalCursor::LastChoice;
    } else {
      SpatialPlanSuccessor next = problem.getSpatialDomain().getNextPlan(
          *last, axis ? SpatialSuccessorDomain::AxisSchemes
                      : SpatialSuccessorDomain::AllPlacements);
      if (next.kind == SpatialPlanSuccessorKind::Failure)
        return {SpatialExpansionKind::CompilerBug,
                next.failure ? next.failure->detail
                             : "spatial successor returned no failure detail"};
      if (next.kind == SpatialPlanSuccessorKind::End) {
        cursor = CanonicalCursor::Exhausted;
        last.reset();
        continue;
      }
      if (!next.plan)
        return {SpatialExpansionKind::CompilerBug,
                "spatial successor returned no plan"};
      choice = std::move(*next.plan);
      last = choice;
    }

    ++work.spatialSuccessorSteps;
    if (resolvedChoices.count(choice)) {
      ++work.duplicateSpatialChoices;
      continue;
    }
    return evaluateAndQueue(std::move(choice));
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
    if (!regionProposalMetricsRecorded) {
      recordRegionPlanMetrics(**regionDomain, continuation.proposals,
                              "region-proposal");
      regionProposalMetricsRecorded = true;
    }
    continuation.initialProposalCount = continuation.proposals.size();
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

mlir::LogicalResult
PhysicalDataflowPlanningSession::addRegionRefinementProposals(
    RegionContinuation &continuation, const RegionPlan &center,
    uint64_t maximumProposals, std::string *failureReason) {
  if (!continuation.needsRefinementProposals(maximumProposals)) {
    if (failureReason)
      *failureReason = "Region refinement was requested outside its boundary";
    return mlir::failure();
  }
  auto regionDomain =
      getOrCreateRegionDomain(continuation.parent, failureReason);
  if (mlir::failed(regionDomain))
    return mlir::failure();
  if (!(*regionDomain)->contains(center)) {
    if (failureReason)
      *failureReason = "Region refinement center is outside the current domain";
    return mlir::failure();
  }
  std::vector<RegionPlan> refinements = [&]() {
    wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                   "build-region-refinements");
    return (*regionDomain)->getRefinementProposals(center, maximumProposals);
  }();
  if (!regionRefinementMetricsRecorded) {
    recordRegionPlanMetrics(**regionDomain, refinements, "region-refinement");
    regionRefinementMetricsRecorded = true;
  }
  auto insertion = continuation.proposals.begin() + continuation.nextProposal;
  for (RegionPlan &plan : refinements) {
    if (continuation.emitted.count(plan) ||
        llvm::is_contained(continuation.proposals, plan))
      continue;
    insertion = continuation.proposals.insert(insertion, std::move(plan)) + 1;
  }
  continuation.refinementProposalsInitialized = true;
  return mlir::success();
}

void PhysicalDataflowPlanningSession::skipRegionRefinementProposals(
    RegionContinuation &continuation) const {
  if (continuation.proposalsInitialized &&
      !continuation.refinementProposalsInitialized)
    continuation.refinementProposalsInitialized = true;
}

bool PhysicalDataflowPlanningSession::hasRemainingSpatialWork() const {
  return canonicalCursor != CanonicalCursor::Exhausted ||
         axisCursor != CanonicalCursor::Exhausted ||
         pausedSpatialChoice.has_value() || !frontier.empty();
}

} // namespace wafer::compiler::detail
