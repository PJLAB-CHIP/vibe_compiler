//===- SearchCurrentIR.cpp - Current-IR physical search ----------------===//

#include "SearchCurrentIR.h"

#include "PhysicalDataflowInstrumentation.h"
#include "StructuredProgramAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningProblem.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Transforms/Linalg/CommunicationRegionClosure.h"
#include "Wafer/Transforms/Linalg/OnlineAttentionDecomposition.h"
#include "Wafer/Transforms/Linalg/SpatialRegionMaterialization.h"
#include "Wafer/Transforms/Linalg/StructuredGraphNormalization.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct CurrentCandidate {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

struct FinishedCandidate {
  ExecutableCompilationResult compilation;
  uint64_t actualLeaves = 1;
};

struct TemporalAxis {
  TemporalDomain domain;
  TemporalSuccessor current;
};

static ExecutableCompilationResult fail(ExecutableCompilationStatus status,
                                        llvm::StringRef gate,
                                        llvm::StringRef detail) {
  ExecutableCompilationResult result;
  result.status = status;
  result.gate = gate.str();
  result.detail = detail.str();
  return result;
}

static ActualCandidateResult actualFailure(ActualCandidateStatus status,
                                           llvm::StringRef detail) {
  ActualCandidateResult result;
  result.status = status;
  result.detail = detail.str();
  return result;
}

static mlir::FailureOr<mlir::func::FuncOp>
getProgramFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp function;
  for (mlir::func::FuncOp candidate : module.getOps<mlir::func::FuncOp>()) {
    if (candidate.isExternal())
      continue;
    if (function)
      return mlir::failure();
    function = candidate;
  }
  return function ? mlir::FailureOr<mlir::func::FuncOp>(function)
                  : mlir::FailureOr<mlir::func::FuncOp>(mlir::failure());
}

static ExecutableCompilationStatus
classifyLayoutFailure(ExactPBQPStatus status) {
  switch (status) {
  case ExactPBQPStatus::Optimal:
  case ExactPBQPStatus::Feasible:
    return ExecutableCompilationStatus::Accepted;
  case ExactPBQPStatus::NoSolution:
    return ExecutableCompilationStatus::UnsupportedFailure;
  case ExactPBQPStatus::Indeterminate:
    return ExecutableCompilationStatus::IndeterminateFailure;
  case ExactPBQPStatus::BrokenContract:
    return ExecutableCompilationStatus::CompilerFailure;
  }
  return ExecutableCompilationStatus::CompilerFailure;
}

static ActualCandidateStatus
classifyActualStatus(ExecutableCompilationStatus status) {
  switch (status) {
  case ExecutableCompilationStatus::Accepted:
    return ActualCandidateStatus::Accepted;
  case ExecutableCompilationStatus::ProvenExactRejection:
    return ActualCandidateStatus::ExactRejection;
  case ExecutableCompilationStatus::UnsupportedFailure:
    return ActualCandidateStatus::Unsupported;
  case ExecutableCompilationStatus::IndeterminateFailure:
    return ActualCandidateStatus::Indeterminate;
  case ExecutableCompilationStatus::CompilerFailure:
    return ActualCandidateStatus::CompilerBug;
  }
  return ActualCandidateStatus::CompilerBug;
}

static bool
hasActualSPMCapacityRejection(const ExecutableCompilationResult &result) {
  return result.isProvenExactRejection() &&
         llvm::any_of(result.tileFailures,
                      [](const ExecutableTileFailure &failure) {
                        return failure.memoryPlanning.spmCapacityOverflow;
                      });
}

static std::optional<std::vector<TemporalChoice>>
refineTemporalChoices(llvm::ArrayRef<TemporalAxis> axes,
                      llvm::ArrayRef<TemporalChoice> current,
                      std::string &detail) {
  if (axes.size() != current.size()) {
    detail = "Temporal refinement lost its current domain tuple";
    return std::nullopt;
  }
  std::vector<TemporalChoice> refined(current.begin(), current.end());
  bool changed = false;
  for (auto [axis, choice] : llvm::zip(axes, refined)) {
    llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
        axis.domain.getScopeDescriptors(choice.kind);
    if (descriptors.size() != choice.scopes.size()) {
      detail = "Temporal refinement scope tuple differs from current domain";
      return std::nullopt;
    }
    for (auto [descriptor, scope] : llvm::zip(descriptors, choice.scopes)) {
      for (auto [dimension, capability] :
           llvm::enumerate(descriptor.iteratorCapabilities)) {
        if (capability != IteratorTilingCapability::Tileable ||
            scope.iteratorTileSizes[dimension] <= 1)
          continue;
        int64_t &size = scope.iteratorTileSizes[dimension];
        size = (size + 1) / 2;
        changed = true;
      }
      auto loopOrder = buildFirstTemporalLoopOrder(
          descriptor.iterationExtents, scope.iteratorTileSizes,
          descriptor.precedence, &detail);
      if (mlir::failed(loopOrder))
        return std::nullopt;
      scope.loopOrder = std::move(*loopOrder);
    }
    if (!axis.domain.contains(choice)) {
      detail = "actual-capacity Temporal refinement is outside the current "
               "typed domain";
      return std::nullopt;
    }
  }
  if (!changed)
    return std::nullopt;
  return refined;
}

static mlir::FailureOr<StructuredMaterializationRelations>
remapRelations(const StructuredMaterializationRelations &source,
               const mlir::IRMapping &mapping, std::string &detail) {
  StructuredMaterializationRelations result;
  for (const MaterializedBufferRelation &relation : source.buffers) {
    mlir::Operation *owner = mapping.lookupOrNull(relation.owner);
    mlir::Value buffer = mapping.lookupOrNull(relation.buffer);
    if (!owner || !buffer) {
      detail = "current candidate clone omitted a buffer relation endpoint";
      return mlir::failure();
    }
    result.buffers.push_back({owner, buffer, relation.role});
  }
  for (const StructuredOutputRelation &relation : source.structuralOutputs) {
    mlir::Value endpoint = mapping.lookupOrNull(relation.endpoint);
    if (!endpoint) {
      detail = "current candidate clone omitted a structural output endpoint";
      return mlir::failure();
    }
    result.structuralOutputs.push_back({relation.outputIndex, endpoint});
  }
  for (const StructuredBoundaryRelation &relation : source.boundaryRelations) {
    mlir::Value sourceEndpoint = mapping.lookupOrNull(relation.sourceEndpoint);
    mlir::Value destinationEndpoint =
        mapping.lookupOrNull(relation.destinationEndpoint);
    if (!sourceEndpoint || !destinationEndpoint) {
      detail = "current candidate clone omitted a boundary endpoint";
      return mlir::failure();
    }
    result.boundaryRelations.push_back({sourceEndpoint, destinationEndpoint});
  }
  return result;
}

static mlir::FailureOr<CurrentCandidate>
cloneCandidate(mlir::ModuleOp source,
               const StructuredMaterializationRelations &relations,
               mlir::IRMapping &mapping, std::string &detail) {
  mlir::Operation *cloned = source->clone(mapping);
  auto module = mlir::dyn_cast<mlir::ModuleOp>(cloned);
  if (!module) {
    if (cloned)
      cloned->destroy();
    detail = "current candidate clone did not produce a builtin.module";
    return mlir::failure();
  }
  auto remapped = remapRelations(relations, mapping, detail);
  if (mlir::failed(remapped)) {
    module->destroy();
    return mlir::failure();
  }
  return CurrentCandidate{mlir::OwningOpRef<mlir::ModuleOp>(module),
                          std::move(*remapped)};
}

static mlir::FailureOr<TemporalChoice>
remapTemporalChoice(const TemporalChoice &source,
                    const mlir::IRMapping &mapping, std::string &detail) {
  TemporalChoice result;
  result.kind = source.kind;
  result.scopes.reserve(source.scopes.size());
  for (const TemporalScopeChoice &scope : source.scopes) {
    mlir::Operation *operation = mapping.lookupOrNull(scope.operation);
    if (!operation) {
      detail = "current candidate clone omitted a Temporal scope operation";
      return mlir::failure();
    }
    result.scopes.push_back(
        {operation, scope.iteratorTileSizes, scope.loopOrder});
  }
  return result;
}

static bool advanceTemporalAxes(std::vector<TemporalAxis> &axes,
                                std::string &detail, bool &compilerBug) {
  for (size_t offset = 0; offset < axes.size(); ++offset) {
    const size_t index = axes.size() - offset - 1;
    TemporalAxis &axis = axes[index];
    const TemporalCursor *cursor = axis.current.getCursor();
    if (!cursor) {
      compilerBug = true;
      detail = "Temporal choice omitted its current-IR cursor";
      return false;
    }
    TemporalSuccessor next = axis.domain.getNextChoice(*cursor);
    if (next.getKind() == TemporalSuccessorKind::CompilerBug) {
      compilerBug = true;
      detail = next.getDetail().str();
      return false;
    }
    if (next.getKind() == TemporalSuccessorKind::Choice) {
      axis.current = std::move(next);
      for (size_t reset = index + 1; reset < axes.size(); ++reset) {
        axes[reset].current = axes[reset].domain.getFirstChoice();
        if (axes[reset].current.getKind() != TemporalSuccessorKind::Choice) {
          compilerBug = true;
          detail = "Temporal domain lost its first current-IR choice";
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

static void
addDownstreamStatistics(CurrentIRDownstreamStatistics &total,
                        const CurrentIRDownstreamStatistics &value) {
  total.materializedExecutionPipelines += value.materializedExecutionPipelines;
  total.tileRegionsLowered += value.tileRegionsLowered;
  total.instructionOperations += value.instructionOperations;
  total.rdmaOperations += value.rdmaOperations;
  total.wdmaOperations += value.wdmaOperations;
  total.gatherScatterOperations += value.gatherScatterOperations;
  total.dteSendOperations += value.dteSendOperations;
  total.dteBroadcastOperations += value.dteBroadcastOperations;
  total.dteScatterOperations += value.dteScatterOperations;
  total.dteUnicastSendsCoalesced += value.dteUnicastSendsCoalesced;
  total.dteReceiveOperations += value.dteReceiveOperations;
  total.dteWaitOperations += value.dteWaitOperations;
  total.nccJoinOperations += value.nccJoinOperations;
}

static llvm::StringRef stringifyCoverage(SearchControllerCoverage coverage) {
  switch (coverage) {
  case SearchControllerCoverage::ComparableBest:
    return "comparable-best";
  case SearchControllerCoverage::FeasibleUnranked:
    return "feasible-unranked";
  case SearchControllerCoverage::FeasiblePartial:
    return "feasible-partial";
  case SearchControllerCoverage::NoFeasible:
    return "no-feasible";
  case SearchControllerCoverage::IncompleteNoCandidate:
    return "incomplete-no-candidate";
  case SearchControllerCoverage::Failed:
    return "failed";
  }
  return "failed";
}

class CurrentIRStructuralEvaluator final : public StructuralCandidateEvaluator {
public:
  CurrentIRStructuralEvaluator(
      mlir::ModuleOp tensorProgram, const StructuredProgramAnalysis &analysis,
      PhysicalDataflowPlanningSession &planning,
      const frontend::FrontendProgramVerificationResult &program,
      const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
      ProgramDataHandoff &programData, const SearchCurrentIROptions &options,
      const std::optional<SearchCostCohort> &cohort,
      SearchCurrentIRStatistics *statistics,
      ExecutableLoweringStatistics *executableStatistics)
      : tensorProgram(tensorProgram), analysis(analysis), planning(planning),
        program(program), executionConfig(executionConfig),
        diagnostics(diagnostics), programData(programData), options(options),
        cohort(cohort), statistics(statistics),
        executableStatistics(executableStatistics) {}

  StructuralCandidateEvaluation
  evaluate(const RegionState &state, uint64_t actualizationCredits) override {
    std::string detail;
    auto rootWorks =
        planning.getCurrentRootWorks(state.getSpatialState(), &detail);
    if (mlir::failed(rootWorks))
      return {actualFailure(ActualCandidateStatus::CompilerBug, detail), 0,
              true};

    SpatialRegionMaterializationFailure spatialFailure;
    auto structural = materializeSpatialRegions(
        tensorProgram, planning.getProblem().getCardId(),
        analysis.availableTileIds, analysis.operationNodes, *rootWorks,
        state.getRegionPlan(), &spatialFailure);
    if (mlir::failed(structural)) {
      ActualCandidateStatus status =
          spatialFailure.kind ==
                  SpatialRegionMaterializationFailureKind::Unsupported
              ? ActualCandidateStatus::Unsupported
              : ActualCandidateStatus::CompilerBug;
      return {actualFailure(status, spatialFailure.detail), 0, true};
    }
    if (statistics)
      ++statistics->structuralMaterializations;

    llvm::SmallVector<TileRegionOp, 32> regions;
    structural->module->walk(
        [&](TileRegionOp region) { regions.push_back(region); });
    std::vector<TemporalAxis> axes;
    axes.reserve(regions.size());
    for (TileRegionOp region : regions) {
      TemporalDomainResult domain = buildTemporalDomain(region);
      if (!domain.succeeded())
        return {actualFailure(ActualCandidateStatus::CompilerBug,
                              domain.failure->detail),
                0, true};
      TemporalSuccessor first = domain.domain->getFirstChoice();
      if (first.getKind() != TemporalSuccessorKind::Choice ||
          !first.getChoice() || !first.getCursor())
        return {actualFailure(ActualCandidateStatus::CompilerBug,
                              first.getDetail().empty()
                                  ? "Temporal domain has no current-IR choice"
                                  : first.getDetail()),
                0, true};
      axes.push_back({std::move(*domain.domain), std::move(first)});
      if (statistics)
        ++statistics->temporalDomainsBuilt;
    }

    std::optional<ExecutableCompilationResult> winner;
    std::optional<SearchObjective> winnerObjective;
    uint64_t actualizations = 0;
    bool sawUnsupported = false;
    bool sawIndeterminate = false;
    bool domainExhausted = false;
    bool compilerBug = false;
    std::optional<std::vector<TemporalChoice>> feedbackChoices;
    const uint64_t actualizationLimit =
        std::min(options.maximumTemporalCandidatesPerStructuralState,
                 actualizationCredits);
    do {
      if (options.deadline &&
          std::chrono::steady_clock::now() >= *options.deadline)
        return {actualFailure(ActualCandidateStatus::Indeterminate,
                              "search wall-time budget exhausted"),
                actualizations, false};
      ++actualizations;
      if (statistics)
        ++statistics->temporalCandidateActualizations;
      mlir::IRMapping mapping;
      auto candidate = cloneCandidate(*structural->module,
                                      structural->relations, mapping, detail);
      if (mlir::failed(candidate))
        return {actualFailure(ActualCandidateStatus::CompilerBug, detail),
                actualizations, false};

      std::vector<TemporalChoice> selectedChoices;
      if (feedbackChoices) {
        selectedChoices = *feedbackChoices;
      } else {
        selectedChoices.reserve(axes.size());
        for (const TemporalAxis &axis : axes)
          selectedChoices.push_back(*axis.current.getChoice());
      }
      bool temporalFailed = false;
      for (auto [axis, selectedChoice] : llvm::zip(axes, selectedChoices)) {
        auto mappedRegion = mlir::dyn_cast_or_null<TileRegionOp>(
            mapping.lookupOrNull(axis.domain.getRegion().getOperation()));
        if (!mappedRegion) {
          detail = "current candidate clone omitted a TileRegion";
          temporalFailed = true;
          break;
        }
        TemporalDomainResult mappedDomain = buildTemporalDomain(mappedRegion);
        if (!mappedDomain.succeeded()) {
          detail = mappedDomain.failure->detail;
          temporalFailed = true;
          break;
        }
        auto mappedChoice =
            remapTemporalChoice(selectedChoice, mapping, detail);
        if (mlir::failed(mappedChoice) ||
            !mappedDomain.domain->contains(*mappedChoice)) {
          if (detail.empty())
            detail = "remapped Temporal choice is outside current IR domain";
          temporalFailed = true;
          break;
        }
        TemporalTilingFailure temporalFailure;
        if (mlir::failed(
                applyTemporalTiling(*mappedDomain.domain, *mappedChoice,
                                    candidate->relations, &temporalFailure))) {
          detail = temporalFailure.detail;
          temporalFailed = true;
          break;
        }
        if (statistics)
          ++statistics->temporalApplications;
      }
      if (temporalFailed)
        return {actualFailure(ActualCandidateStatus::CompilerBug, detail),
                actualizations, false};

      FinishedCandidate finished =
          finishCandidate(std::move(*candidate),
                          actualizationLimit - actualizations + 1, detail);
      if (finished.actualLeaves > 1) {
        actualizations += finished.actualLeaves - 1;
      }
      ExecutableCompilationResult compiled = std::move(finished.compilation);
      const ActualCandidateStatus status =
          classifyActualStatus(compiled.status);
      const bool capacityRejected = hasActualSPMCapacityRejection(compiled);
      const bool structuralChoiceInvariantFailure =
          status != ActualCandidateStatus::Accepted &&
          compiled.failureScope ==
              ExecutableFailureScope::StructuralChoiceInvariant;
      switch (status) {
      case ActualCandidateStatus::Accepted: {
        SearchObjective objective =
            deriveSearchObjective(compiled.executable->resourceCost, cohort);
        bool replace = !winner;
        if (winnerObjective) {
          SearchObjectiveComparison comparison =
              compareSearchObjectives(objective, *winnerObjective);
          replace = comparison == SearchObjectiveComparison::Better;
          if (comparison == SearchObjectiveComparison::Incomparable &&
              statistics)
            ++statistics->incomparableTemporalObjectives;
        }
        if (replace) {
          winner = std::move(compiled);
          winnerObjective = std::move(objective);
        }
        if (statistics)
          ++statistics->acceptedTemporalCandidates;
        break;
      }
      case ActualCandidateStatus::ExactRejection:
        if (statistics)
          ++statistics->exactRejectedTemporalCandidates;
        break;
      case ActualCandidateStatus::Unsupported:
        sawUnsupported = true;
        if (statistics)
          ++statistics->unsupportedTemporalCandidates;
        break;
      case ActualCandidateStatus::Indeterminate:
        sawIndeterminate = true;
        if (statistics)
          ++statistics->indeterminateTemporalCandidates;
        break;
      case ActualCandidateStatus::CompilerBug:
        return {
            actualFailure(ActualCandidateStatus::CompilerBug, compiled.detail),
            actualizations, false};
      }

      if (status == ActualCandidateStatus::Accepted &&
          options.stopTemporalAfterFirstAccepted) {
        domainExhausted = false;
        break;
      }
      if (structuralChoiceInvariantFailure) {
        domainExhausted = true;
        break;
      }

      if (actualizations >= actualizationLimit)
        break;
      if (capacityRejected) {
        auto refined = refineTemporalChoices(axes, selectedChoices, detail);
        if (refined) {
          feedbackChoices = std::move(*refined);
          if (statistics)
            ++statistics->actualCapacityRefinements;
          continue;
        }
        if (statistics)
          ++statistics->unavailableCapacityRefinements;
      }
      feedbackChoices.reset();
      const bool hasNext = advanceTemporalAxes(axes, detail, compilerBug);
      if (compilerBug)
        return {actualFailure(ActualCandidateStatus::CompilerBug, detail),
                actualizations, false};
      domainExhausted = !hasNext;
      if (!hasNext)
        break;
    } while (true);

    if (winner) {
      ActualCandidateResult result;
      result.status = ActualCandidateStatus::Accepted;
      result.compilation.emplace(std::move(*winner));
      return {std::move(result), actualizations, domainExhausted};
    }
    if (!domainExhausted)
      return {actualFailure(ActualCandidateStatus::Indeterminate,
                            "Temporal current-IR traversal reached its "
                            "candidate budget without an accepted owner"),
              actualizations, false};
    if (sawIndeterminate)
      return {actualFailure(ActualCandidateStatus::Indeterminate,
                            "one or more actual Temporal candidates were "
                            "indeterminate"),
              actualizations, true};
    return {actualFailure(
                ActualCandidateStatus::Unsupported,
                sawUnsupported
                    ? "all current Temporal candidates were unsupported or "
                      "rejected by actual downstream gates"
                    : "all current Temporal candidates were rejected by "
                      "actual downstream gates"),
            actualizations, true};
  }

private:
  FinishedCandidate finishCandidate(CurrentCandidate candidate,
                                    uint64_t maximumMovementLeaves,
                                    std::string &detail) {
    SpatialRegionMaterializationFailure closureFailure;
    CommunicationRegionClosureStatistics closureStatistics;
    if (mlir::failed(closeCrossTileCommunicationRegions(
            *candidate.module, candidate.relations, &closureStatistics,
            &closureFailure)))
      return {fail(closureFailure.kind ==
                           SpatialRegionMaterializationFailureKind::Unsupported
                       ? ExecutableCompilationStatus::UnsupportedFailure
                       : ExecutableCompilationStatus::CompilerFailure,
                   "search-communication-region-closure",
                   closureFailure.detail)};
    if (statistics)
      statistics->communicationRegionClosures +=
          closureStatistics.closedExchangeComponents;

    OnlineAttentionDecompositionFailure attentionFailure;
    if (mlir::failed(decomposeOnlineAttention(
            *candidate.module, candidate.relations, &attentionFailure)))
      return {fail(
          attentionFailure.kind ==
                  OnlineAttentionDecompositionFailureKind::UnsupportedSemantics
              ? ExecutableCompilationStatus::UnsupportedFailure
              : ExecutableCompilationStatus::CompilerFailure,
          "search-attention-decomposition", attentionFailure.detail)};

    LayoutOptimizationResult layout = resolveCurrentLayoutsAndBufferize(
        *candidate.module, candidate.relations, options.layoutWorkLimit);
    recordLayoutInstrumentation(layout.statistics);
    if (statistics) {
      ++statistics->layoutInvocations;
      statistics->layoutFeasibleFallbacks +=
          layout.status == ExactPBQPStatus::Feasible;
    }
    if (!layout.succeeded())
      return {fail(classifyLayoutFailure(layout.status), "search-layout",
                   layout.detail)};

    StructuredToTileResult compute =
        lowerStructuredComputeToTile(*candidate.module, candidate.relations);
    if (!compute.succeeded())
      return {fail(compute.failure == StructuredToTileFailureKind::Unsupported
                       ? ExecutableCompilationStatus::UnsupportedFailure
                       : ExecutableCompilationStatus::CompilerFailure,
                   "search-structured-to-tile", compute.detail)};

    bool recursiveAvailable = false;
    DistributedMovementAvailability distributedAvailability;
    if (maximumMovementLeaves > 1) {
      RecursiveDoublingAvailability availability =
          analyzeRecursiveDoublingAvailability(*candidate.module,
                                               candidate.relations);
      if (availability.kind ==
          RecursiveDoublingAvailabilityKind::BrokenContract)
        return {fail(ExecutableCompilationStatus::CompilerFailure,
                     "search-recursive-doubling-availability",
                     availability.detail)};
      recursiveAvailable = availability.isAvailable();
      distributedAvailability = analyzeDistributedMovementAvailability(
          *candidate.module, candidate.relations);
      if (distributedAvailability.brokenContract)
        return {fail(ExecutableCompilationStatus::CompilerFailure,
                     "search-distributed-movement-availability",
                     distributedAvailability.detail)};
    }

    struct PendingMovement {
      CurrentCandidate candidate;
      BoundaryMovementOptions options;
      bool requiresRecursive = false;
      bool requiresDimensionOrderedAllToAll = false;
      bool requiresDistributedRing = false;
    };
    std::vector<PendingMovement> pending;
    pending.push_back(
        PendingMovement{std::move(candidate), BoundaryMovementOptions{}});
    auto appendClone =
        [&](BoundaryMovementOptions movementOptions, bool requiresRecursive,
            bool requiresAllToAll,
            bool requiresDistributedRing) -> mlir::LogicalResult {
      if (pending.size() >= maximumMovementLeaves)
        return mlir::success();
      mlir::IRMapping mapping;
      auto cloned =
          cloneCandidate(*pending.front().candidate.module,
                         pending.front().candidate.relations, mapping, detail);
      if (mlir::failed(cloned))
        return mlir::failure();
      pending.push_back(PendingMovement{std::move(*cloned), movementOptions,
                                        requiresRecursive, requiresAllToAll,
                                        requiresDistributedRing});
      return mlir::success();
    };
    if (recursiveAvailable &&
        mlir::failed(appendClone(
            BoundaryMovementOptions{
                CompleteAllGatherAlgorithm::RecursiveDoubling,
                CompleteAllToAllAlgorithm::Direct,
                DistributedReductionAlgorithm::Centralized},
            /*requiresRecursive=*/true, /*requiresAllToAll=*/false,
            /*requiresDistributedRing=*/false)))
      return {fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-recursive-doubling-clone", detail)};
    if (distributedAvailability.dimensionOrderedAllToAll &&
        mlir::failed(appendClone(
            BoundaryMovementOptions{CompleteAllGatherAlgorithm::Ring,
                                    CompleteAllToAllAlgorithm::DimensionOrdered,
                                    DistributedReductionAlgorithm::Centralized},
            /*requiresRecursive=*/false, /*requiresAllToAll=*/true,
            /*requiresDistributedRing=*/false)))
      return {fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-dimension-ordered-all-to-all-clone", detail)};
    if (recursiveAvailable &&
        distributedAvailability.dimensionOrderedAllToAll &&
        mlir::failed(appendClone(
            BoundaryMovementOptions{
                CompleteAllGatherAlgorithm::RecursiveDoubling,
                CompleteAllToAllAlgorithm::DimensionOrdered,
                DistributedReductionAlgorithm::Centralized},
            /*requiresRecursive=*/true, /*requiresAllToAll=*/true,
            /*requiresDistributedRing=*/false)))
      return {fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-combined-movement-clone", detail)};
    if (distributedAvailability.distributedReduction &&
        mlir::failed(appendClone(
            BoundaryMovementOptions{CompleteAllGatherAlgorithm::Ring,
                                    CompleteAllToAllAlgorithm::Direct,
                                    DistributedReductionAlgorithm::Ring},
            /*requiresRecursive=*/false, /*requiresAllToAll=*/false,
            /*requiresDistributedRing=*/true)))
      return {fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-distributed-ring-clone", detail)};
    if (recursiveAvailable && distributedAvailability.distributedReduction &&
        mlir::failed(appendClone(
            BoundaryMovementOptions{
                CompleteAllGatherAlgorithm::RecursiveDoubling,
                CompleteAllToAllAlgorithm::Direct,
                DistributedReductionAlgorithm::Ring},
            /*requiresRecursive=*/true, /*requiresAllToAll=*/false,
            /*requiresDistributedRing=*/true)))
      return {fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-recursive-distributed-ring-clone", detail)};
    if (distributedAvailability.dimensionOrderedAllToAll &&
        distributedAvailability.distributedReduction &&
        mlir::failed(appendClone(
            BoundaryMovementOptions{CompleteAllGatherAlgorithm::Ring,
                                    CompleteAllToAllAlgorithm::DimensionOrdered,
                                    DistributedReductionAlgorithm::Ring},
            /*requiresRecursive=*/false, /*requiresAllToAll=*/true,
            /*requiresDistributedRing=*/true)))
      return {fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-all-to-all-distributed-ring-clone", detail)};
    if (recursiveAvailable &&
        distributedAvailability.dimensionOrderedAllToAll &&
        distributedAvailability.distributedReduction &&
        mlir::failed(appendClone(
            BoundaryMovementOptions{
                CompleteAllGatherAlgorithm::RecursiveDoubling,
                CompleteAllToAllAlgorithm::DimensionOrdered,
                DistributedReductionAlgorithm::Ring},
            /*requiresRecursive=*/true, /*requiresAllToAll=*/true,
            /*requiresDistributedRing=*/true)))
      return {fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-complete-movement-clone", detail)};

    std::vector<PendingMovement> materialized;
    materialized.reserve(pending.size());
    for (PendingMovement &movementCandidate : pending) {
      BoundaryMovementResult movement = materializeTileBoundaryMovement(
          *movementCandidate.candidate.module,
          movementCandidate.candidate.relations, movementCandidate.options);
      recordMovementInstrumentation(movement.statistics);
      if (!movement.succeeded())
        return {
            fail(movement.failure == BoundaryMovementFailureKind::Unsupported
                     ? ExecutableCompilationStatus::UnsupportedFailure
                     : ExecutableCompilationStatus::CompilerFailure,
                 "search-boundary-movement", movement.detail)};
      if (movementCandidate.requiresRecursive &&
          movement.statistics.recursiveDoublingComponents == 0)
        return {fail(ExecutableCompilationStatus::CompilerFailure,
                     "search-boundary-movement-recursive-doubling",
                     "availability and recursive materialization disagree")};
      if (movementCandidate.requiresDimensionOrderedAllToAll &&
          movement.statistics.dimensionOrderedAllToAllComponents == 0)
        continue;
      if (movementCandidate.requiresDistributedRing &&
          movement.statistics.ringReduceScatterComponents == 0 &&
          movement.statistics.ringAllReduceComponents == 0)
        continue;
      if (statistics) {
        statistics->recursiveDoublingCandidates +=
            movementCandidate.requiresRecursive;
        statistics->dimensionOrderedAllToAllCandidates +=
            movementCandidate.requiresDimensionOrderedAllToAll;
        statistics->distributedRingCandidates +=
            movementCandidate.requiresDistributedRing;
      }
      materialized.push_back(std::move(movementCandidate));
    }

    auto compileMovementCandidate = [&](CurrentCandidate selected) {
      CurrentIRDownstreamStatistics downstream;
      ExecutableCompilationResult result =
          compileCurrentIRCandidateToExecutable(
              std::move(selected.module), std::move(selected.relations),
              planning.getProblem().getCardId(), analysis.availableTileIds,
              program, executionConfig, diagnostics, programData,
              options.downstream, &downstream, executableStatistics);
      if (statistics) {
        ++statistics->movementCandidateActualizations;
        addDownstreamStatistics(statistics->downstream, downstream);
      }
      return result;
    };

    struct CompiledMovement {
      ExecutableCompilationResult result;
      bool recursive = false;
      bool dimensionOrderedAllToAll = false;
      bool distributedRing = false;
    };
    std::vector<std::unique_ptr<CompiledMovement>> compiled;
    compiled.reserve(materialized.size());
    for (PendingMovement &movementCandidate : materialized) {
      ExecutableCompilationResult result =
          compileMovementCandidate(std::move(movementCandidate.candidate));
      if (result.status == ExecutableCompilationStatus::CompilerFailure)
        return {std::move(result), compiled.size() + 1};
      if (statistics && result.isAccepted()) {
        statistics->recursiveDoublingAccepted +=
            movementCandidate.requiresRecursive;
        statistics->dimensionOrderedAllToAllAccepted +=
            movementCandidate.requiresDimensionOrderedAllToAll;
        statistics->distributedRingAccepted +=
            movementCandidate.requiresDistributedRing;
      }
      compiled.push_back(std::make_unique<CompiledMovement>(CompiledMovement{
          std::move(result), movementCandidate.requiresRecursive,
          movementCandidate.requiresDimensionOrderedAllToAll,
          movementCandidate.requiresDistributedRing}));
    }
    if (compiled.empty())
      return {fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-boundary-movement",
                   "default movement candidate was not materialized"),
              0};

    std::optional<size_t> winner;
    std::optional<SearchObjective> winnerObjective;
    for (auto [index, candidateResult] : llvm::enumerate(compiled)) {
      if (!candidateResult->result.isAccepted())
        continue;
      SearchObjective objective = deriveSearchObjective(
          candidateResult->result.executable->resourceCost, cohort);
      if (!winner) {
        winner = index;
        winnerObjective = std::move(objective);
        continue;
      }
      SearchObjectiveComparison comparison =
          compareSearchObjectives(objective, *winnerObjective);
      if (comparison == SearchObjectiveComparison::Better) {
        winner = index;
        winnerObjective = std::move(objective);
      } else if (comparison == SearchObjectiveComparison::Incomparable &&
                 statistics) {
        ++statistics->incomparableMovementObjectives;
      }
    }
    const uint64_t actualLeaves = compiled.size();
    if (winner) {
      if (statistics) {
        statistics->recursiveDoublingWinners += compiled[*winner]->recursive;
        statistics->dimensionOrderedAllToAllWinners +=
            compiled[*winner]->dimensionOrderedAllToAll;
        statistics->distributedRingWinners +=
            compiled[*winner]->distributedRing;
      }
      return {std::move(compiled[*winner]->result), actualLeaves};
    }
    for (std::unique_ptr<CompiledMovement> &candidateResult : compiled)
      if (candidateResult->result.status ==
          ExecutableCompilationStatus::IndeterminateFailure)
        return {std::move(candidateResult->result), actualLeaves};
    for (std::unique_ptr<CompiledMovement> &candidateResult : compiled)
      if (candidateResult->result.isProvenExactRejection())
        return {std::move(candidateResult->result), actualLeaves};
    return {std::move(compiled.front()->result), actualLeaves};
  }

  mlir::ModuleOp tensorProgram;
  const StructuredProgramAnalysis &analysis;
  PhysicalDataflowPlanningSession &planning;
  const frontend::FrontendProgramVerificationResult &program;
  const ExecutionConfig &executionConfig;
  llvm::raw_ostream &diagnostics;
  ProgramDataHandoff &programData;
  const SearchCurrentIROptions &options;
  const std::optional<SearchCostCohort> &cohort;
  SearchCurrentIRStatistics *statistics = nullptr;
  ExecutableLoweringStatistics *executableStatistics = nullptr;
};

} // namespace

ExecutableCompilationResult compileSearchCurrentIR(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const SearchCurrentIROptions &options,
    SearchCurrentIRStatistics *statistics,
    ExecutableLoweringStatistics *executableStatistics) {
  if (!tensorProgram || options.layoutWorkLimit == 0 ||
      options.maximumTemporalCandidatesPerStructuralState == 0 ||
      options.limits.width == 0 || options.limits.trials == 0)
    return fail(ExecutableCompilationStatus::CompilerFailure, "search-input",
                "search requires current TensorProgram and positive work "
                "limits");
  if (options.deadline &&
      std::chrono::steady_clock::now() >= *options.deadline)
    return fail(ExecutableCompilationStatus::IndeterminateFailure,
                "search-controller", "search wall-time budget exhausted");
  const uint64_t maximumRegionRefinementCandidates =
      options.getRefinementLimit();
  const uint64_t maximumInitialRegionProposals =
      options.getInitialProposalLimit();
  mlir::FailureOr<mlir::func::FuncOp> function =
      getProgramFunction(tensorProgram);
  if (mlir::failed(function))
    return fail(ExecutableCompilationStatus::CompilerFailure, "search-input",
                "search requires exactly one defined TensorProgram");
  if (mlir::failed(closeStructuredProgramOutputs(*function)))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "search-output-closure",
                "search cannot close current TensorProgram outputs");
  if (mlir::failed(mlir::verify(tensorProgram)))
    return fail(ExecutableCompilationStatus::CompilerFailure, "search-input",
                "search requires verifier-valid normalized TensorProgram");

  auto structured = analyzeStructuredProgram(tensorProgram, program,
                                             executionConfig, diagnostics);
  if (mlir::failed(structured))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "structured-analysis",
                "cannot derive search structured program facts");
  std::string detail;
  auto problem = PhysicalDataflowPlanningProblem::create(
      **structured, CardId(0), analysis::IndexRelationLimits(), &detail);
  if (mlir::failed(problem))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "search-planning-problem", detail);
  PhysicalDataflowPlanningSession planning(*problem,
                                           maximumInitialRegionProposals);

  auto cohort = SearchCostCohort::create(SearchCostPolicy{}, &detail);
  if (mlir::failed(cohort))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "search-cost-cohort", detail);
  std::optional<SearchCostCohort> currentCohort(std::move(*cohort));
  CurrentIRStructuralEvaluator evaluator(tensorProgram, **structured, planning,
                                         program, executionConfig, diagnostics,
                                         programData, options, currentCohort,
                                         statistics, executableStatistics);

  UnifiedSearchOptions traversal;
  traversal.planningCredits = options.planningCredits;
  traversal.structuralCandidateCredits = options.limits.width;
  traversal.candidateActualizationCredits = options.limits.trials;
  traversal.maximumRegionRefinementCandidates =
      maximumRegionRefinementCandidates;
  traversal.termination = options.termination;
  traversal.costCohort = currentCohort;
  // Actual capacity remains typed, but it is not generalized to a structural
  // no-good until current buffer owners can prove the causal root subset.
  traversal.exactRejectionCache = ExactRejectionCachePolicy::Disabled;
  UnifiedSearchResult searched =
      runUnifiedSearch(planning, evaluator, traversal);
  auto searchCounter = [&](llvm::StringRef name, uint64_t value) {
    wafer::support::addCompileCounter("search", name, value);
  };
  searchCounter("spatial-successor-steps",
                searched.planning.spatialSuccessorSteps);
  searchCounter("spatial-demand-queries",
                searched.planning.spatialDemandQueries);
  searchCounter("spatial-states", searched.planning.spatialStatesQueued);
  searchCounter("root-work-successor-steps",
                searched.planning.rootWorkSuccessorSteps);
  searchCounter("root-works", searched.planning.rootWorksValidated);
  searchCounter("region-successor-steps",
                searched.planning.regionSuccessorSteps);
  searchCounter("region-states", searched.planning.regionStatesQueued);
  searchCounter("successor-steps", searched.work.successorSteps);
  searchCounter("structural-states", searched.work.structuralStatesActualized);
  searchCounter("candidate-actualizations",
                searched.work.candidateActualizations);
  searchCounter("width", options.limits.width);
  searchCounter("trials", options.limits.trials);
  searchCounter("trials-used", searched.work.candidateActualizations);
  searchCounter("trials-remaining",
                options.limits.trials - searched.work.candidateActualizations);
  searchCounter("incomplete-inner-domains",
                searched.work.incompleteInnerDomains);
  searchCounter("accepted-structural-states",
                searched.control.statistics.accepted);
  searchCounter("exact-rejected-structural-states",
                searched.control.statistics.exactRejected);
  searchCounter("unsupported-structural-states",
                searched.control.statistics.unsupported);
  searchCounter("indeterminate-structural-states",
                searched.control.statistics.indeterminate);
  if (statistics) {
    searchCounter("temporal-domains", statistics->temporalDomainsBuilt);
    searchCounter("temporal-applications", statistics->temporalApplications);
    searchCounter("accepted-temporal-candidates",
                  statistics->acceptedTemporalCandidates);
    searchCounter("exact-rejected-temporal-candidates",
                  statistics->exactRejectedTemporalCandidates);
    searchCounter("actual-capacity-refinements",
                  statistics->actualCapacityRefinements);
    searchCounter("unavailable-capacity-refinements",
                  statistics->unavailableCapacityRefinements);
    searchCounter("movement-candidate-actualizations",
                  statistics->movementCandidateActualizations);
    searchCounter("recursive-doubling-candidates",
                  statistics->recursiveDoublingCandidates);
    searchCounter("recursive-doubling-accepted",
                  statistics->recursiveDoublingAccepted);
    searchCounter("recursive-doubling-winners",
                  statistics->recursiveDoublingWinners);
    searchCounter("dimension-ordered-all-to-all-candidates",
                  statistics->dimensionOrderedAllToAllCandidates);
    searchCounter("dimension-ordered-all-to-all-accepted",
                  statistics->dimensionOrderedAllToAllAccepted);
    searchCounter("dimension-ordered-all-to-all-winners",
                  statistics->dimensionOrderedAllToAllWinners);
    searchCounter("distributed-ring-candidates",
                  statistics->distributedRingCandidates);
    searchCounter("distributed-ring-accepted",
                  statistics->distributedRingAccepted);
    searchCounter("distributed-ring-winners",
                  statistics->distributedRingWinners);
  }
  if (statistics) {
    statistics->planning = searched.planning;
    statistics->traversal = searched.work;
    statistics->controller = searched.control.statistics;
    statistics->coverage = searched.control.coverage;
  }
  diagnostics << "wafer-compile: search-current-ir coverage="
              << stringifyCoverage(searched.control.coverage)
              << " structural=" << searched.work.structuralStatesActualized
              << " actual=" << searched.work.candidateActualizations;
  if (statistics)
    diagnostics << " accepted=" << statistics->acceptedTemporalCandidates
                << " exact_rejected="
                << statistics->exactRejectedTemporalCandidates
                << " capacity_refinements="
                << statistics->actualCapacityRefinements
                << " unavailable_refinements="
                << statistics->unavailableCapacityRefinements
                << " unsupported=" << statistics->unsupportedTemporalCandidates
                << " indeterminate="
                << statistics->indeterminateTemporalCandidates;
  diagnostics << '\n';
  if (searched.control.winner)
    return std::move(searched.control.winner->compilation);

  switch (searched.control.coverage) {
  case SearchControllerCoverage::NoFeasible:
    return fail(ExecutableCompilationStatus::UnsupportedFailure,
                "search-controller",
                searched.failureDetail.empty()
                    ? "search exhausted the current domain without a feasible "
                      "candidate"
                    : searched.failureDetail);
  case SearchControllerCoverage::IncompleteNoCandidate:
    return fail(ExecutableCompilationStatus::IndeterminateFailure,
                "search-controller",
                searched.failureDetail.empty()
                    ? "search stopped before finding an accepted current-IR "
                      "candidate"
                    : searched.failureDetail);
  case SearchControllerCoverage::Failed:
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "search-controller",
                searched.failureDetail.empty() ? "search controller failed"
                                               : searched.failureDetail);
  case SearchControllerCoverage::ComparableBest:
  case SearchControllerCoverage::FeasibleUnranked:
  case SearchControllerCoverage::FeasiblePartial:
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "search-controller",
                "search reported feasible coverage without a retained owner");
  }
  return fail(ExecutableCompilationStatus::CompilerFailure, "search-controller",
              "search returned an unknown coverage");
}

} // namespace wafer::compiler::detail
