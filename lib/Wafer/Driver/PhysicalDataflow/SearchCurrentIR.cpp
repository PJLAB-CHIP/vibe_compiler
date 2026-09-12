//===- SearchCurrentIR.cpp - Current-IR physical search ----------------===//

#include "SearchCurrentIR.h"

#include "PhysicalDataflowInstrumentation.h"
#include "StructuredProgramAnalysis.h"
#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningProblem.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Linalg/CommunicationRegionClosure.h"
#include "Wafer/Transforms/Linalg/ContractionAccumulation.h"
#include "Wafer/Transforms/Linalg/OnlineAttentionDecomposition.h"
#include "Wafer/Transforms/Linalg/SpatialRegionMaterialization.h"
#include "Wafer/Transforms/Linalg/StructuredGraphNormalization.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/ReadOnlyInputSharing.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct CurrentCandidate {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
  /// Choice coordinates attached to actual, unchanged Region body blocks.
  /// These are neither buffer owners nor memory-planning facts. A transform
  /// that destroys this boundary drops its bindings before mutation.
  llvm::SmallVector<std::pair<mlir::Block *, size_t>, 16> temporalBodies;
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

static std::optional<std::vector<TemporalChoice>> refineTemporalChoices(
    llvm::ArrayRef<TemporalAxis> axes, llvm::ArrayRef<TemporalChoice> current,
    const std::set<size_t> &affectedAxes, std::string &detail) {
  if (axes.size() != current.size()) {
    detail = "Temporal refinement lost its current domain tuple";
    return std::nullopt;
  }
  std::vector<TemporalChoice> refined(current.begin(), current.end());
  bool changed = false;
  // Only the live allocation certificate's current owner can select a scope.
  // The scalar size orders alternative parameters after that proof; it never
  // predicts capacity, identifies an owner or rules out another choice.
  llvm::SmallVector<std::optional<std::tuple<size_t, size_t, int64_t>>, 16>
      refinements(axes.size());
  size_t axisIndex = 0;
  for (auto [axis, choice] : llvm::zip(axes, refined)) {
    if (!affectedAxes.count(axisIndex)) {
      ++axisIndex;
      continue;
    }
    llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
        axis.domain.getScopeDescriptors(choice.kind);
    if (descriptors.size() != choice.scopes.size()) {
      detail = "Temporal refinement scope tuple differs from current domain";
      return std::nullopt;
    }
    // A Region owner alone cannot distinguish independent roots or a fused
    // traversal from its internal reduction scope.
    if (descriptors.size() != 1) {
      ++axisIndex;
      continue;
    }
    for (auto [descriptorIndex, descriptorAndScope] :
         llvm::enumerate(llvm::zip(descriptors, choice.scopes))) {
      const TemporalScopeDescriptor &descriptor =
          std::get<0>(descriptorAndScope);
      TemporalScopeChoice &scope = std::get<1>(descriptorAndScope);
      for (auto [dimension, capability] :
           llvm::enumerate(descriptor.iteratorCapabilities)) {
        if (capability != IteratorTilingCapability::Tileable ||
            scope.iteratorTileSizes[dimension] <= 1)
          continue;
        auto &selected = refinements[axisIndex];
        if (!selected ||
            scope.iteratorTileSizes[dimension] > std::get<2>(*selected))
          selected = std::make_tuple(descriptorIndex, dimension,
                                     scope.iteratorTileSizes[dimension]);
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
    ++axisIndex;
  }
  for (size_t index = 0; index < refinements.size(); ++index) {
    if (!refinements[index])
      continue;
    const auto [scopeIndex, dimension, size] = *refinements[index];
    refined[index].scopes[scopeIndex].iteratorTileSizes[dimension] =
        (size + 1) / 2;
    llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
        axes[index].domain.getScopeDescriptors(refined[index].kind);
    auto loopOrder = buildFirstTemporalLoopOrder(
        descriptors[scopeIndex].iterationExtents,
        refined[index].scopes[scopeIndex].iteratorTileSizes,
        descriptors[scopeIndex].precedence, &detail);
    if (mlir::failed(loopOrder))
      return std::nullopt;
    refined[index].scopes[scopeIndex].loopOrder = std::move(*loopOrder);
    changed = true;
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

static mlir::FailureOr<CurrentCandidate>
cloneCandidate(const CurrentCandidate &source, mlir::IRMapping &mapping,
               std::string &detail) {
  auto result =
      cloneCandidate(*source.module, source.relations, mapping, detail);
  if (mlir::failed(result))
    return mlir::failure();
  for (auto [body, axis] : source.temporalBodies) {
    mlir::Block *mapped = mapping.lookupOrNull(body);
    if (!mapped) {
      detail = "candidate clone omitted a current Temporal body";
      return mlir::failure();
    }
    result->temporalBodies.emplace_back(mapped, axis);
  }
  return result;
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

struct MovementChoice {
  BoundaryMovementOptions options;
  bool recursive = false;
  bool allToAll = false;
  bool reduction = false;
  bool pipeline = false;
  bool shareInput = false;
};

class CurrentIRCandidateSession final : public StructuralCandidateSession {
public:
  CurrentIRCandidateSession(
      RegionState state, mlir::ModuleOp tensorProgram,
      const StructuredProgramAnalysis &analysis,
      PhysicalDataflowPlanningSession &planning,
      const frontend::FrontendProgramVerificationResult &program,
      const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
      ProgramDataHandoff &programData, const SearchCurrentIROptions &options,
      SearchCurrentIRStatistics *statistics,
      ExecutableLoweringStatistics *executableStatistics)
      : state(std::move(state)), tensorProgram(tensorProgram),
        analysis(analysis), planning(planning), program(program),
        executionConfig(executionConfig), diagnostics(diagnostics),
        programData(programData), options(options), statistics(statistics),
        executableStatistics(executableStatistics) {}

  StructuralCandidateEvaluation advance() override {
    while (true) {
      if (exhausted)
        return {actualFailure(ActualCandidateStatus::CompilerBug,
                              "candidate session advanced after exhaustion"),
                0};
      if (options.deadline &&
          std::chrono::steady_clock::now() >= *options.deadline) {
        exhausted = true;
        return {actualFailure(ActualCandidateStatus::Indeterminate,
                              "search wall-time budget exhausted"),
                0};
      }
      // Exactly one charged attempt, even when a transform before Instr fails.
      if (!structural) {
        auto failure = initialize();
        if (failure)
          return finish(std::move(*failure));
      }
      TemporalWork work = nextTemporalWork();
      temporalPhase = (static_cast<unsigned>(work) + 1) % 3;
      if (work != TemporalWork::Resume) {
        auto failure = startTemporal(work);
        if (failure)
          return finish(std::move(*failure));
        if (pending.empty()) {
          exhausted = true;
          return {{}, 0, CandidateContinuation::Exhausted};
        }
      }
      TemporalAttempt &temporal = pending.front();
      RegionAttempt &attempt = temporal.regions.front();
      if (!attempt.lowered) {
        auto prepared = prepareRegion(temporal, attempt);
        if (std::holds_alternative<RegionAlternativesExhausted>(prepared)) {
          completeRegion(temporal);
          continue;
        }
        if (auto *failure =
                std::get_if<ExecutableCompilationResult>(&prepared)) {
          completeRegion(temporal);
          return finish(std::move(*failure));
        }
      }
      if (attempt.nextMovement == attempt.movements.size() &&
          !appendMixedMovement(attempt)) {
        completeRegion(temporal);
        continue;
      }
      const MovementChoice choice = attempt.movements[attempt.nextMovement++];
      std::string detail;
      mlir::IRMapping mapping;
      auto candidate = cloneCandidate(*attempt.lowered, mapping, detail);
      if (mlir::failed(candidate))
        return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                           "search-movement-clone", detail));
      BoundaryMovementOptions movementOptions = choice.options;
      for (auto &component : movementOptions.components) {
        component.anchor.sourceEndpoint =
            mapping.lookupOrNull(component.anchor.sourceEndpoint);
        component.anchor.destinationEndpoint =
            mapping.lookupOrNull(component.anchor.destinationEndpoint);
        if (!component.anchor.sourceEndpoint ||
            !component.anchor.destinationEndpoint)
          return finish(
              fail(ExecutableCompilationStatus::CompilerFailure,
                   "search-movement-clone",
                   "clone omitted a selected communication component"));
      }
      BoundaryMovementResult movement = materializeTileBoundaryMovement(
          *candidate->module, candidate->relations, movementOptions);
      recordMovementInstrumentation(movement.statistics);
      ExecutableCompilationResult compiled;
      std::set<size_t> affectedAxes;
      std::mutex feedbackMutex;
      if (!movement.succeeded()) {
        compiled =
            fail(movement.failure == BoundaryMovementFailureKind::Unsupported
                     ? ExecutableCompilationStatus::UnsupportedFailure
                     : ExecutableCompilationStatus::CompilerFailure,
                 "search-boundary-movement", movement.detail);
      } else if ((choice.recursive &&
                  !movement.statistics.recursiveDoublingComponents) ||
                 (choice.allToAll &&
                  !movement.statistics.dimensionOrderedAllToAllComponents) ||
                 (choice.reduction &&
                  !movement.statistics.ringReduceScatterComponents &&
                  !movement.statistics.ringAllReduceComponents)) {
        compiled = fail(ExecutableCompilationStatus::UnsupportedFailure,
                        "search-boundary-movement",
                        "selected algorithm has no current component");
      } else {
        if (!choice.shareInput && !choice.pipeline &&
            hasReadOnlyInputSharing(*candidate->module)) {
          MovementChoice shared = choice;
          shared.shareInput = true;
          attempt.movements.push_back(std::move(shared));
        }
        if (choice.shareInput) {
          if (statistics)
            ++statistics->inputSharingCandidates;
          auto shared = materializeReadOnlyInputSharing(*candidate->module,
                                                        candidate->relations);
          if (!shared.succeeded())
            return finish(
                fail(shared.failure == BoundaryMovementFailureKind::Unsupported
                         ? ExecutableCompilationStatus::UnsupportedFailure
                         : ExecutableCompilationStatus::CompilerFailure,
                     "search-input-sharing", shared.detail));
          recordMovementInstrumentation(shared.statistics);
          support::addCompileCounter("input-sharing", "removed-ddr-loads",
                                     shared.statistics.peerReceives);
        }
        if (!choice.pipeline &&
            hasDistanceOneLoadPipeline(*candidate->module)) {
          MovementChoice pipelined = choice;
          pipelined.pipeline = true;
          attempt.movements.push_back(std::move(pipelined));
        }
        if (statistics) {
          statistics->recursiveDoublingCandidates += choice.recursive;
          statistics->dimensionOrderedAllToAllCandidates += choice.allToAll;
          statistics->distributedRingCandidates += choice.reduction;
          statistics->sharedDDRCandidates +=
              choice.options.transport == BoundaryMovementTransport::SharedDDR;
        }
        CurrentIRDownstreamStatistics downstream;
        // SCF pipelining replaces loop bodies. Their earlier capacity-owner
        // scope handles do not survive this actual transformation.
        if (choice.pipeline)
          candidate->temporalBodies.clear();
        const auto temporalBodies = candidate->temporalBodies;
        auto observeCapacity =
            [&](const SPMMemoryPlanningFailure &failure,
                const StructuredMaterializationRelations &relations) {
              if (options.downstream.capacityObserver)
                options.downstream.capacityObserver(failure, relations);
              StorageRootMemo roots;
              std::set<size_t> localAxes;
              auto observe =
                  [&](const SPMMemoryPlanningFailure::DemandEvidence &demand) {
                    const auto &allocationRoots =
                        roots.getStorageRoots(demand.allocation);
                    for (const auto &relation : relations.buffers) {
                      const auto &ownerRoots =
                          roots.getStorageRoots(relation.buffer);
                      if (!llvm::any_of(allocationRoots, [&](mlir::Value root) {
                            return ownerRoots.contains(root);
                          }))
                        continue;
                      for (mlir::Operation *owner = relation.owner; owner;
                           owner = owner->getParentOp())
                        for (auto [body, axis] : temporalBodies)
                          if (owner->getBlock() == body)
                            localAxes.insert(axis);
                    }
                  };
              for (const auto &demand : failure.individuallyOversizedDemands)
                observe(demand);
              for (const auto &demand : failure.capacityConflictDemands)
                observe(demand);
              std::lock_guard<std::mutex> lock(feedbackMutex);
              affectedAxes.insert(localAxes.begin(), localAxes.end());
            };
        CurrentIRDownstreamOptions downstreamOptions = options.downstream;
        downstreamOptions.distanceOneLoadPipeline = choice.pipeline;
        if (choice.pipeline)
          support::addCompileCounter("search", "pipeline-candidates", 1);
        downstreamOptions.capacityObserver = observeCapacity;
        compiled = compileCurrentIRCandidateToExecutable(
            std::move(candidate->module), std::move(candidate->relations),
            planning.getProblem().getCardId(), analysis.availableTileIds,
            program, executionConfig, diagnostics, programData,
            downstreamOptions, &downstream, executableStatistics);
        if (choice.pipeline && compiled.isAccepted())
          support::addCompileCounter("search", "pipeline-accepted", 1);
        if (statistics && choice.shareInput && compiled.isAccepted())
          ++statistics->inputSharingAccepted;
        if (statistics) {
          ++statistics->movementCandidateActualizations;
          addDownstreamStatistics(statistics->downstream, downstream);
          if (compiled.isAccepted()) {
            statistics->recursiveDoublingAccepted += choice.recursive;
            statistics->dimensionOrderedAllToAllAccepted += choice.allToAll;
            statistics->distributedRingAccepted += choice.reduction;
            statistics->sharedDDRAccepted +=
                choice.options.transport ==
                BoundaryMovementTransport::SharedDDR;
            statistics->mergedRegionAccepted += attempt.merged;
            statistics->regionPreservingAccepted += !attempt.merged;
          }
        }
      }
      if (hasActualSPMCapacityRejection(compiled)) {
        auto refined =
            refineTemporalChoices(axes, temporal.choices, affectedAxes, detail);
        if (refined && !wasVisited(*refined)) {
          repairs.push_back(std::move(*refined));
          if (statistics)
            ++statistics->actualCapacityRefinements;
        } else if (statistics) {
          ++statistics->unavailableCapacityRefinements;
        }
      }
      // Each Region/layout/movement family advances one leaf before yielding.
      // Live queries and exact actual prefixes move with their continuation.
      if (!pending.empty()) {
        auto next = std::move(pending.front());
        pending.pop_front();
        if (!next.regions.empty()) {
          auto region = std::move(next.regions.front());
          next.regions.pop_front();
          next.regions.push_back(std::move(region));
        }
        pending.push_back(std::move(next));
      }
      return finish(std::move(compiled));
    }
  }

private:
  enum class TemporalWork { Proposal, Repair, Resume };

  TemporalWork nextTemporalWork() const {
    for (unsigned offset = 0; offset < 3; ++offset) {
      auto work = static_cast<TemporalWork>((temporalPhase + offset) % 3);
      if ((work == TemporalWork::Proposal && !proposals.empty()) ||
          (work == TemporalWork::Repair && !repairs.empty()) ||
          (work == TemporalWork::Resume && !pending.empty()))
        return work;
    }
    // With no retained parameter continuation, advance the ordinary raw
    // domain through the same proposal/materialization entry.
    return TemporalWork::Proposal;
  }

  struct RegionPrepared {};
  struct RegionAlternativesExhausted {};
  using RegionPreparation =
      std::variant<RegionPrepared, RegionAlternativesExhausted,
                   ExecutableCompilationResult>;
  struct RegionAttempt {
    std::optional<CurrentCandidate> lowered;
    std::optional<CurrentCandidate> layoutBase;
    std::unique_ptr<LayoutAssignmentQuery> layoutQuery;
    std::optional<ExactPBQPResult> firstLayout;
    std::optional<ExactPBQPResult> localPlacement;
    std::vector<LayoutConstraint> layoutConstraints;
    std::set<std::vector<uint32_t>> visitedLayouts;
    size_t nextLayout = 0;
    std::vector<MovementChoice> movements;
    size_t nextMovement = 0;
    llvm::SmallVector<StructuredBoundaryRelation, 4> components;
    std::vector<MovementChoice> algorithms;
    std::vector<uint8_t> transportMask;
    size_t nextAlgorithm = 0;
    bool hasMixedMask = false;
    bool mixedExhausted = false;
    bool merged = false;
  };
  struct TemporalAttempt {
    std::vector<TemporalChoice> choices;
    CurrentCandidate tiled;
    std::deque<RegionAttempt> regions;
  };

  static bool appendMixedMovement(RegionAttempt &attempt) {
    if (attempt.components.size() < 2 || attempt.mixedExhausted)
      return false;
    while (!attempt.hasMixedMask) {
      size_t bit = 0;
      while (bit < attempt.transportMask.size() && attempt.transportMask[bit])
        attempt.transportMask[bit++] = 0;
      if (bit == attempt.transportMask.size()) {
        attempt.mixedExhausted = true;
        return false;
      }
      attempt.transportMask[bit] = 1;
      if (llvm::all_of(attempt.transportMask,
                       [](uint8_t value) { return value; }))
        continue; // Uniform DDR already has its own actual candidate.
      attempt.hasMixedMask = true;
      attempt.nextAlgorithm = 0;
    }
    MovementChoice choice = attempt.algorithms[attempt.nextAlgorithm++];
    for (auto [index, edge] : llvm::enumerate(attempt.components))
      if (attempt.transportMask[index])
        choice.options.components.push_back(
            {edge, BoundaryMovementTransport::SharedDDR});
    attempt.movements.push_back(std::move(choice));
    if (attempt.nextAlgorithm == attempt.algorithms.size())
      attempt.hasMixedMask = false;
    return true;
  }

  std::optional<ExecutableCompilationResult> initialize() {
    std::string detail;
    auto rootWorks =
        planning.getCurrentRootWorks(state.getSpatialState(), &detail);
    if (mlir::failed(rootWorks))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-root-work", detail);
    SpatialRegionMaterializationFailure failure;
    auto materialized = materializeSpatialRegions(
        tensorProgram, planning.getProblem().getCardId(),
        analysis.availableTileIds, analysis.operationNodes, *rootWorks,
        state.getRegionPlan(), &failure);
    if (mlir::failed(materialized))
      return fail(failure.kind ==
                          SpatialRegionMaterializationFailureKind::Unsupported
                      ? ExecutableCompilationStatus::UnsupportedFailure
                      : ExecutableCompilationStatus::CompilerFailure,
                  "search-structural", failure.detail);
    structural.emplace(CurrentCandidate{std::move(materialized->module),
                                        std::move(materialized->relations)});
    if (statistics)
      ++statistics->structuralMaterializations;
    llvm::SmallVector<TileRegionOp, 32> regions;
    structural->module->walk(
        [&](TileRegionOp region) { regions.push_back(region); });
    for (TileRegionOp region : regions) {
      auto domain = buildTemporalDomain(region);
      if (!domain.succeeded())
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-temporal-domain", domain.failure->detail);
      auto first = domain.domain->getFirstChoice();
      if (first.getKind() != TemporalSuccessorKind::Choice)
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-temporal-domain",
                    "current Region has no first Temporal choice");
      axes.push_back({std::move(*domain.domain), std::move(first)});
      structural->temporalBodies.emplace_back(&region.getBody().front(),
                                              axes.size() - 1);
      if (statistics)
        ++statistics->temporalDomainsBuilt;
    }
    // Structural traversal kinds get explicit entry points before the enormous
    // Cartesian product of numeric sizes and permutations.
    std::vector<TemporalChoice> joint, independent;
    bool hasFusion = false;
    for (const auto &axis : axes) {
      joint.push_back(*axis.current.getChoice());
      auto first = axis.domain.getFirstIndependentChoice();
      if (first.getKind() != TemporalSuccessorKind::Choice)
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-temporal-domain",
                    "current Region has no Independent Temporal choice");
      independent.push_back(*first.getChoice());
      hasFusion |= !axis.domain.getFusions().empty();
    }
    auto appendSeeds =
        [&](const std::vector<TemporalChoice> &initial) -> mlir::LogicalResult {
      if (!llvm::is_contained(proposals, initial))
        proposals.push_back(initial);
      auto interior = initial;
      // Sample the geometric midpoint of each legal extent interval before
      // enumerating adjacent integers. This proposal is constructed before
      // any capacity result and contains no memory-footprint estimate.
      for (auto [axis, choice] : llvm::zip(axes, interior)) {
        auto descriptors = axis.domain.getScopeDescriptors(choice.kind);
        for (auto [descriptor, scope] :
             llvm::zip_equal(descriptors, choice.scopes)) {
          for (auto [dimension, capability] :
               llvm::enumerate(descriptor.iteratorCapabilities)) {
            if (capability != IteratorTilingCapability::Tileable)
              continue;
            int64_t extent = descriptor.iterationExtents[dimension];
            int64_t size = 1;
            while (size < extent / size)
              size *= 2;
            scope.iteratorTileSizes[dimension] = std::min(size, extent);
          }
          auto order = buildFirstTemporalLoopOrder(
              descriptor.iterationExtents, scope.iteratorTileSizes,
              descriptor.precedence, &detail);
          if (mlir::failed(order))
            return mlir::failure();
          scope.loopOrder = std::move(*order);
        }
        if (!axis.domain.contains(choice)) {
          detail = "geometric Temporal proposal is outside its typed domain";
          return mlir::failure();
        }
      }
      auto append = [&](std::vector<TemporalChoice> choices) {
        if (!llvm::is_contained(proposals, choices))
          proposals.push_back(std::move(choices));
      };
      // Preserve reuse on other axes while sampling the interior of the
      // largest tileable extent. Shrinking all dimensions together can turn
      // a feasible contraction into thousands of unnecessarily small issues.
      auto singleAxis = initial;
      for (auto [axisIndex, choice] : llvm::enumerate(singleAxis)) {
        auto &axis = axes[axisIndex];
        auto descriptors = axis.domain.getScopeDescriptors(choice.kind);
        for (auto [scopeIndex, scope] : llvm::enumerate(choice.scopes)) {
          const auto &descriptor = descriptors[scopeIndex];
          std::optional<size_t> largest;
          for (auto [dimension, capability] :
               llvm::enumerate(descriptor.iteratorCapabilities))
            if (capability == IteratorTilingCapability::Tileable &&
                descriptor.iterationExtents[dimension] > 1 &&
                (!largest || descriptor.iterationExtents[dimension] >
                                 descriptor.iterationExtents[*largest]))
              largest = dimension;
          if (largest)
            scope.iteratorTileSizes[*largest] =
                interior[axisIndex]
                    .scopes[scopeIndex]
                    .iteratorTileSizes[*largest];
          auto order = buildFirstTemporalLoopOrder(
              descriptor.iterationExtents, scope.iteratorTileSizes,
              descriptor.precedence, &detail);
          if (mlir::failed(order))
            return mlir::failure();
          scope.loopOrder = std::move(*order);
        }
        if (!axis.domain.contains(choice)) {
          detail = "single-axis Temporal proposal is outside its typed domain";
          return mlir::failure();
        }
      }
      append(std::move(singleAxis));
      auto coordinated = interior;
      for (auto [axis, choice] : llvm::zip(axes, coordinated))
        if (auto coupled = axis.domain.getCoupledStateProposal(choice))
          choice = std::move(*coupled);
      append(coordinated);
      // A geometric seed also needs a nearby larger scale. Raw enumeration
      // otherwise returns to extent-1 and can spend the entire budget far
      // away from the feasible interior. This is an ordinary domain choice,
      // independent of capacity evidence and cost acceptance.
      auto larger = coordinated;
      for (auto [axis, choice] : llvm::zip(axes, larger)) {
        auto descriptors = axis.domain.getScopeDescriptors(choice.kind);
        for (auto [descriptor, scope] :
             llvm::zip_equal(descriptors, choice.scopes)) {
          for (auto [dimension, capability] :
               llvm::enumerate(descriptor.iteratorCapabilities)) {
            if (capability != IteratorTilingCapability::Tileable)
              continue;
            int64_t extent = descriptor.iterationExtents[dimension];
            int64_t &size = scope.iteratorTileSizes[dimension];
            size = size > extent / 2 ? extent : size * 2;
          }
          auto order = buildFirstTemporalLoopOrder(
              descriptor.iterationExtents, scope.iteratorTileSizes,
              descriptor.precedence, &detail);
          if (mlir::failed(order))
            return mlir::failure();
          scope.loopOrder = std::move(*order);
        }
        if (!axis.domain.contains(choice)) {
          detail = "scaled Temporal proposal is outside its typed domain";
          return mlir::failure();
        }
      }
      append(std::move(larger));
      append(std::move(interior));
      return mlir::success();
    };
    if (mlir::failed(appendSeeds(joint)) ||
        (hasFusion && mlir::failed(appendSeeds(independent))))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-temporal-proposals", detail);
    return std::nullopt;
  }

  bool wasVisited(llvm::ArrayRef<TemporalChoice> choices) const {
    return llvm::any_of(visited, [&](const auto &previous) {
      return llvm::ArrayRef<TemporalChoice>(previous) == choices;
    });
  }

  std::optional<ExecutableCompilationResult> startTemporal(TemporalWork work) {
    std::vector<TemporalChoice> choices;
    std::string detail;
    if (work == TemporalWork::Repair) {
      choices = std::move(repairs.front());
      repairs.pop_front();
    } else if (!proposals.empty()) {
      choices = std::move(proposals.front());
      proposals.pop_front();
    } else {
      bool compilerBug = false;
      while (true) {
        if (!advanceTemporalAxes(axes, detail, compilerBug)) {
          if (compilerBug)
            return fail(ExecutableCompilationStatus::CompilerFailure,
                        "search-temporal-next", detail);
          exhausted = pending.empty();
          return std::nullopt;
        }
        choices.clear();
        for (const auto &axis : axes)
          choices.push_back(*axis.current.getChoice());
        if (!wasVisited(choices))
          break;
      }
    }
    if (wasVisited(choices))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-temporal-next",
                  "Temporal proposal was enqueued twice");
    visited.push_back(choices);
    if (statistics)
      ++statistics->temporalCandidateActualizations;
    mlir::IRMapping mapping;
    auto candidate = cloneCandidate(*structural, mapping, detail);
    if (mlir::failed(candidate))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-temporal-clone", detail);
    for (auto [axis, choice] : llvm::zip(axes, choices)) {
      auto region = mlir::dyn_cast_or_null<TileRegionOp>(
          mapping.lookupOrNull(axis.domain.getRegion().getOperation()));
      if (!region)
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-temporal-remap",
                    "clone omitted a current TileRegion");
      auto domain = remapTemporalDomain(axis.domain, region, mapping, &detail);
      auto remapped = remapTemporalChoice(choice, mapping, detail);
      if (mlir::failed(domain) || mlir::failed(remapped))
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-temporal-remap", detail);
      TemporalTilingFailure failure;
      if (mlir::failed(applyTemporalTiling(*domain, *remapped,
                                           candidate->relations, &failure)))
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-temporal-apply", failure.detail);
      if (statistics)
        ++statistics->temporalApplications;
    }
    SpatialRegionMaterializationFailure failure;
    auto availability = analyzeCommunicationRegionClosure(
        *candidate->module, candidate->relations, &failure);
    if (mlir::failed(availability))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-communication-region-availability", failure.detail);
    const bool canMerge =
        *availability == CommunicationRegionClosureAvailability::Available;
    TemporalAttempt attempt;
    attempt.choices = std::move(choices);
    attempt.tiled = std::move(*candidate);
    attempt.regions.emplace_back();
    if (canMerge) {
      attempt.regions.emplace_back();
      attempt.regions.back().merged = true;
    }
    pending.push_front(std::move(attempt));
    return std::nullopt;
  }

  RegionPreparation prepareRegion(TemporalAttempt &temporal,
                                  RegionAttempt &attempt) {
    std::string detail;
    if (!attempt.layoutBase) {
      mlir::IRMapping mapping;
      auto candidate = cloneCandidate(temporal.tiled, mapping, detail);
      if (mlir::failed(candidate))
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-region-clone", detail);
      if (attempt.merged) {
        // Closure replaces these actual Region bodies. No source identity is
        // reconstructed from the merged IR for capacity feedback.
        candidate->temporalBodies.clear();
        SpatialRegionMaterializationFailure failure;
        CommunicationRegionClosureStatistics closure;
        if (mlir::failed(closeCrossTileCommunicationRegions(
                *candidate->module, candidate->relations, &closure, &failure)))
          return fail(ExecutableCompilationStatus::CompilerFailure,
                      "search-communication-region-closure", failure.detail);
        if (statistics) {
          statistics->communicationRegionClosures +=
              closure.closedExchangeComponents;
          ++statistics->mergedRegionCandidates;
        }
      } else if (statistics) {
        ++statistics->regionPreservingCandidates;
      }
      OnlineAttentionDecompositionFailure attentionFailure;
      if (mlir::failed(decomposeOnlineAttention(
              *candidate->module, candidate->relations, &attentionFailure)))
        return fail(attentionFailure.kind ==
                            OnlineAttentionDecompositionFailureKind::
                                UnsupportedSemantics
                        ? ExecutableCompilationStatus::UnsupportedFailure
                        : ExecutableCompilationStatus::CompilerFailure,
                    "search-attention-decomposition", attentionFailure.detail);
      auto prepared =
          prepareCurrentLayoutInput(*candidate->module, candidate->relations);
      prepared.statistics.invocations = 0;
      recordLayoutInstrumentation(prepared.statistics);
      if (!prepared.succeeded())
        return fail(classifyLayoutFailure(prepared.status),
                    "search-layout-input", prepared.detail);
      auto query = queryCurrentLayoutAssignment(*candidate->module);
      if (!query.query)
        return fail(classifyLayoutFailure(query.outcome.status),
                    "search-layout-query", query.outcome.detail);
      auto first = query.query->solve(options.layoutWorkLimit);
      if (first.status != ExactPBQPStatus::Optimal &&
          first.status != ExactPBQPStatus::Feasible)
        return fail(classifyLayoutFailure(first.status), "search-layout-query",
                    "current layout PBQP has no complete assignment");
      attempt.layoutConstraints = query.query->alternatives(first);
      attempt.firstLayout = std::move(first);
      attempt.layoutQuery = std::move(query.query);
      attempt.layoutBase.emplace(std::move(*candidate));
    }
    ExactPBQPResult assignment;
    LayoutMaterializationPlacement placement =
        LayoutMaterializationPlacement::FirstUse;
    while (true) {
      if (attempt.localPlacement) {
        assignment = std::move(*attempt.localPlacement);
        attempt.localPlacement.reset();
        break;
      } else if (attempt.firstLayout) {
        assignment = std::move(*attempt.firstLayout);
        attempt.firstLayout.reset();
      } else {
        if (attempt.nextLayout == attempt.layoutConstraints.size())
          return RegionAlternativesExhausted{};
        assignment = attempt.layoutQuery->solve(
            options.layoutWorkLimit,
            attempt.layoutConstraints[attempt.nextLayout++]);
        if (assignment.status == ExactPBQPStatus::NoSolution)
          continue;
        if (assignment.status != ExactPBQPStatus::Optimal &&
            assignment.status != ExactPBQPStatus::Feasible)
          return fail(classifyLayoutFailure(assignment.status),
                      "search-layout-constraint",
                      "current layout constraint query did not complete");
      }
      // Full assignments from this same current-IR query have an exact
      // identity. Equivalent constrained solves consume query work only.
      if (attempt.visitedLayouts.insert(assignment.assignment).second) {
        if (attempt.layoutQuery->hasLoopInvariantPlacement(assignment)) {
          attempt.localPlacement = assignment;
          placement = LayoutMaterializationPlacement::LoopInvariant;
        }
        break;
      }
    }
    mlir::IRMapping mapping;
    auto candidate = cloneCandidate(*attempt.layoutBase, mapping, detail);
    if (mlir::failed(candidate))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-layout-clone", detail);
    LayoutOptimizationResult layout;
    {
      support::ScopedCompileTimingSpan timing("stage", "current-ir-physical",
                                              "layout-and-bufferization");
      layout =
          attempt.layoutQuery->apply(*candidate->module, candidate->relations,
                                     assignment, &mapping, placement);
    }
    recordLayoutInstrumentation(layout.statistics);
    if (statistics) {
      ++statistics->layoutInvocations;
      statistics->layoutFeasibleFallbacks +=
          layout.status == ExactPBQPStatus::Feasible;
    }
    if (!layout.succeeded())
      return fail(classifyLayoutFailure(layout.status), "search-layout",
                  layout.detail);
    StructuredToTileResult compute;
    {
      support::ScopedCompileTimingSpan timing("stage", "current-ir-physical",
                                              "structured-to-tile");
      compute = lowerStructuredComputeToTile(*candidate->module,
                                             candidate->relations);
    }
    if (!compute.succeeded())
      return fail(compute.failure == StructuredToTileFailureKind::Unsupported
                      ? ExecutableCompilationStatus::UnsupportedFailure
                      : ExecutableCompilationStatus::CompilerFailure,
                  "search-structured-to-tile", compute.detail);
    auto recursive = analyzeRecursiveDoublingAvailability(*candidate->module,
                                                          candidate->relations);
    if (recursive.kind == RecursiveDoublingAvailabilityKind::BrokenContract)
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-movement-domain", recursive.detail);
    auto distributed = analyzeDistributedMovementAvailability(
        *candidate->module, candidate->relations);
    if (distributed.brokenContract)
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-movement-domain", distributed.detail);
    auto components = queryBoundaryMovementComponents(*candidate->module,
                                                      candidate->relations);
    if (!components.succeeded())
      return fail(components.failure == BoundaryMovementFailureKind::Unsupported
                      ? ExecutableCompilationStatus::UnsupportedFailure
                      : ExecutableCompilationStatus::CompilerFailure,
                  "search-movement-components", components.detail);
    attempt.components = std::move(components.anchors);
    attempt.transportMask.assign(attempt.components.size(), 0);
    attempt.movements.push_back({});
    attempt.algorithms.push_back({});
    if (distributed.sharedDDR) {
      MovementChoice ddr;
      ddr.options.transport = BoundaryMovementTransport::SharedDDR;
      attempt.movements.push_back(ddr);
    }
    // Typed algorithm choices only. Each clone is created when this cursor is
    // actually selected, then enters the same full downstream acceptance gate.
    for (unsigned mask = 1; mask < 8; ++mask) {
      bool gather = mask & 1, allToAll = mask & 2, reduce = mask & 4;
      if ((gather && !recursive.isAvailable()) ||
          (allToAll && !distributed.dimensionOrderedAllToAll) ||
          (reduce && !distributed.distributedReduction))
        continue;
      MovementChoice choice;
      choice.recursive = gather;
      choice.allToAll = allToAll;
      choice.reduction = reduce;
      if (gather)
        choice.options.allGather =
            CompleteAllGatherAlgorithm::RecursiveDoubling;
      if (allToAll)
        choice.options.allToAll = CompleteAllToAllAlgorithm::DimensionOrdered;
      if (reduce)
        choice.options.reduction = DistributedReductionAlgorithm::Ring;
      attempt.movements.push_back(choice);
      attempt.algorithms.push_back(choice);
    }
    attempt.lowered.emplace(std::move(*candidate));
    return RegionPrepared{};
  }

  void completeRegion(TemporalAttempt &temporal) {
    RegionAttempt &attempt = temporal.regions.front();
    attempt.lowered.reset();
    attempt.movements.clear();
    attempt.nextMovement = 0;
    attempt.components.clear();
    attempt.algorithms.clear();
    attempt.transportMask.clear();
    attempt.nextAlgorithm = 0;
    attempt.hasMixedMask = false;
    attempt.mixedExhausted = false;
    if (attempt.layoutQuery &&
        (attempt.firstLayout || attempt.localPlacement ||
         attempt.nextLayout < attempt.layoutConstraints.size()))
      return;
    temporal.regions.pop_front();
    if (temporal.regions.empty())
      pending.pop_front();
  }

  StructuralCandidateEvaluation finish(ExecutableCompilationResult compiled) {
    const auto status = classifyActualStatus(compiled.status);
    hasAcceptedCandidate |= compiled.isAccepted();
    if (status == ActualCandidateStatus::CompilerBug ||
        compiled.failureScope ==
            ExecutableFailureScope::StructuralChoiceInvariant ||
        !structural)
      exhausted = true;
    if (statistics) {
      statistics->acceptedCandidates += compiled.isAccepted();
      statistics->exactRejectedCandidates += compiled.isProvenExactRejection();
      statistics->unsupportedCandidates +=
          status == ActualCandidateStatus::Unsupported;
      statistics->indeterminateCandidates +=
          status == ActualCandidateStatus::Indeterminate;
    }
    if (!compiled.isAccepted() && support::getActiveCompileTimingSession())
      diagnostics << "wafer-compile: rejected-candidate gate=" << compiled.gate
                  << " detail=" << compiled.detail << '\n';
    ActualCandidateResult result;
    result.status = status;
    result.detail = compiled.detail;
    if (compiled.isAccepted() || compiled.isProvenExactRejection())
      result.compilation.emplace(std::move(compiled));
    TemporalWork next = nextTemporalWork();
    return {std::move(result), 1,
            exhausted                      ? CandidateContinuation::Exhausted
            : next == TemporalWork::Repair ? CandidateContinuation::Repair
            : next == TemporalWork::Resume && hasAcceptedCandidate
                ? CandidateContinuation::Improve
                : CandidateContinuation::Explore,
            repairs.empty() ? CandidateRetention::Replaceable
                            : CandidateRetention::PendingCapacityRepair};
  }

  RegionState state;
  mlir::ModuleOp tensorProgram;
  const StructuredProgramAnalysis &analysis;
  PhysicalDataflowPlanningSession &planning;
  const frontend::FrontendProgramVerificationResult &program;
  const ExecutionConfig &executionConfig;
  llvm::raw_ostream &diagnostics;
  ProgramDataHandoff &programData;
  const SearchCurrentIROptions &options;
  SearchCurrentIRStatistics *statistics;
  ExecutableLoweringStatistics *executableStatistics;
  std::optional<CurrentCandidate> structural;
  std::vector<TemporalAxis> axes;
  std::vector<std::vector<TemporalChoice>> visited;
  std::deque<std::vector<TemporalChoice>> proposals;
  std::deque<std::vector<TemporalChoice>> repairs;
  std::deque<TemporalAttempt> pending;
  unsigned temporalPhase = 0;
  bool exhausted = false;
  bool hasAcceptedCandidate = false;
};

class CurrentIRStructuralEvaluator final : public StructuralCandidateEvaluator {
public:
  CurrentIRStructuralEvaluator(
      mlir::ModuleOp tensorProgram, const StructuredProgramAnalysis &analysis,
      PhysicalDataflowPlanningSession &planning,
      const frontend::FrontendProgramVerificationResult &program,
      const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
      ProgramDataHandoff &programData, const SearchCurrentIROptions &options,
      SearchCurrentIRStatistics *statistics,
      ExecutableLoweringStatistics *executableStatistics)
      : tensorProgram(tensorProgram), analysis(analysis), planning(planning),
        program(program), executionConfig(executionConfig),
        diagnostics(diagnostics), programData(programData), options(options),
        statistics(statistics), executableStatistics(executableStatistics) {}

  std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &state) override {
    return std::make_unique<CurrentIRCandidateSession>(
        state, tensorProgram, analysis, planning, program, executionConfig,
        diagnostics, programData, options, statistics, executableStatistics);
  }

private:
  mlir::ModuleOp tensorProgram;
  const StructuredProgramAnalysis &analysis;
  PhysicalDataflowPlanningSession &planning;
  const frontend::FrontendProgramVerificationResult &program;
  const ExecutionConfig &executionConfig;
  llvm::raw_ostream &diagnostics;
  ProgramDataHandoff &programData;
  const SearchCurrentIROptions &options;
  SearchCurrentIRStatistics *statistics;
  ExecutableLoweringStatistics *executableStatistics;
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
      options.limits.width == 0 || options.limits.trials == 0)
    return fail(ExecutableCompilationStatus::CompilerFailure, "search-input",
                "search requires current TensorProgram and positive work "
                "limits");
  if (options.deadline && std::chrono::steady_clock::now() >= *options.deadline)
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
  if (mlir::failed(promoteContractionAccumulation(*function)))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "contraction-accumulation",
                "cannot materialize F32 accumulators");
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
  auto admission = PhysicalDataflowPlanningProblem::create(
      **structured, CardId(0), analysis::IndexRelationLimits());
  auto *problem = std::get_if<PhysicalDataflowPlanningProblem>(&admission);
  if (!problem) {
    const auto &failure = std::get<SpatialDomainFailure>(admission);
    return fail(failure.kind == SpatialDomainFailureKind::UnsupportedSemantics
                    ? ExecutableCompilationStatus::UnsupportedFailure
                    : ExecutableCompilationStatus::CompilerFailure,
                "search-planning-problem", failure.detail);
  }
  PhysicalDataflowPlanningSession planning(*problem,
                                           maximumInitialRegionProposals);

  auto cohort =
      analysis::SearchCostCohort::create(analysis::SearchCostPolicy{}, &detail);
  if (mlir::failed(cohort))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "search-cost-cohort", detail);
  std::optional<analysis::SearchCostCohort> currentCohort(std::move(*cohort));
  CurrentIRStructuralEvaluator evaluator(
      tensorProgram, **structured, planning, program, executionConfig,
      diagnostics, programData, options, statistics, executableStatistics);

  UnifiedSearchOptions traversal;
  traversal.planningCredits = options.planningCredits;
  traversal.retainedBranches = options.limits.width;
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
  searchCounter("peak-retained-branches", searched.work.peakRetainedBranches);
  searchCounter("resumed-candidates", searched.work.resumedCandidates);
  searchCounter("retired-branches", searched.work.retiredBranches);
  searchCounter("candidate-actualizations",
                searched.work.candidateActualizations);
  searchCounter("width", options.limits.width);
  searchCounter("trials", options.limits.trials);
  searchCounter("trials-used", searched.work.candidateActualizations);
  searchCounter("trials-remaining",
                options.limits.trials - searched.work.candidateActualizations);
  searchCounter("incomplete-inner-domains",
                searched.work.incompleteInnerDomains);
  searchCounter("accepted-candidates", searched.control.statistics.accepted);
  searchCounter("exact-rejected-candidates",
                searched.control.statistics.exactRejected);
  searchCounter("unsupported-candidates",
                searched.control.statistics.unsupported);
  searchCounter("indeterminate-candidates",
                searched.control.statistics.indeterminate);
  if (statistics) {
    searchCounter("temporal-domains", statistics->temporalDomainsBuilt);
    searchCounter("temporal-applications", statistics->temporalApplications);
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
    searchCounter("dimension-ordered-all-to-all-candidates",
                  statistics->dimensionOrderedAllToAllCandidates);
    searchCounter("dimension-ordered-all-to-all-accepted",
                  statistics->dimensionOrderedAllToAllAccepted);
    searchCounter("distributed-ring-candidates",
                  statistics->distributedRingCandidates);
    searchCounter("distributed-ring-accepted",
                  statistics->distributedRingAccepted);
    searchCounter("region-preserving-candidates",
                  statistics->regionPreservingCandidates);
    searchCounter("region-preserving-accepted",
                  statistics->regionPreservingAccepted);
    searchCounter("merged-region-candidates",
                  statistics->mergedRegionCandidates);
    searchCounter("merged-region-accepted", statistics->mergedRegionAccepted);
    searchCounter("shared-ddr-candidates", statistics->sharedDDRCandidates);
    searchCounter("shared-ddr-accepted", statistics->sharedDDRAccepted);
    searchCounter("input-sharing-candidates",
                  statistics->inputSharingCandidates);
    searchCounter("input-sharing-accepted", statistics->inputSharingAccepted);
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
    diagnostics << " accepted=" << statistics->acceptedCandidates
                << " exact_rejected=" << statistics->exactRejectedCandidates
                << " capacity_refinements="
                << statistics->actualCapacityRefinements
                << " unavailable_refinements="
                << statistics->unavailableCapacityRefinements
                << " unsupported=" << statistics->unsupportedCandidates
                << " indeterminate=" << statistics->indeterminateCandidates;
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
