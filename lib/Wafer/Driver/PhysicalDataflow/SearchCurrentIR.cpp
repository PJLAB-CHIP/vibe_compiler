//===- SearchCurrentIR.cpp - Current-IR physical search ----------------===//

#include "SearchCurrentIR.h"

#include "CapacityFeedback.h"
#include "PhysicalDataflowInstrumentation.h"
#include "StructuredProgramAnalysis.h"
#include "TemporalProposals.h"
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
#include "Wafer/Transforms/Tile/AccessReuse.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
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
  support::ScopedCompileTimingSpan timing("search-phase", "current-ir",
                                          "clone-candidate");
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
  std::optional<AccessReuseChoice> reuse;
  // Owned, immutable actual IR after boundary closure. Profitable siblings
  // share this prefix; it contains no predicted loads or storage.
  std::shared_ptr<const CurrentCandidate> input;
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
      ExecutableLoweringStatistics *executableStatistics,
      const std::optional<analysis::SearchCostCohort> &costCohort,
      size_t explorationStratum)
      : state(std::move(state)), tensorProgram(tensorProgram),
        analysis(analysis), planning(planning), program(program),
        executionConfig(executionConfig), diagnostics(diagnostics),
        programData(programData), options(options), statistics(statistics),
        executableStatistics(executableStatistics), costCohort(costCohort),
        explorationStratum(explorationStratum) {}

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
        return yield();
      }
      if (stage == Stage::SelectTemporal) {
        const bool continuing = !pending.empty();
        if (statistics && continuing)
          ++statistics->temporalBackpressureTurns;
        TemporalWork work = nextTemporalWork();
        if (!continuing) {
          auto order = temporalWorkOrder();
          for (unsigned offset = 0; offset < order.size(); ++offset)
            if (order[(temporalPhase + offset) % order.size()] == work) {
              temporalPhase = (temporalPhase + offset + 1) % order.size();
              break;
            }
          currentContinuation =
              work == TemporalWork::Repair ? CandidateContinuation::Repair
              : work == TemporalWork::Improve || work == TemporalWork::Resume
                  ? CandidateContinuation::Improve
                  : CandidateContinuation::Explore;
        }
        if (work == TemporalWork::Resume && pending.empty()) {
          pending.push_back(std::move(*realization));
          realization.reset();
        }
        if (work != TemporalWork::Resume) {
          auto failure = startTemporal(work);
          if (failure)
            return finish(std::move(*failure));
          if (pending.empty()) {
            exhausted = true;
            return {{}, 0, CandidateContinuation::Exhausted};
          }
        }
        stage = Stage::PrepareRegion;
        return yield();
      }
      TemporalAttempt &temporal = pending.front();
      RegionAttempt &attempt = temporal.regions.front();
      if (stage == Stage::PrepareRegion) {
        if (!attempt.lowered) {
          auto prepared = prepareRegion(temporal, attempt);
          if (std::holds_alternative<RegionPreparationYielded>(prepared))
            return yield();
          if (auto *failure =
                  std::get_if<ExecutableCompilationResult>(&prepared)) {
            completeRegion(temporal);
            stage = Stage::SelectTemporal;
            return finish(std::move(*failure));
          }
        }
        stage = Stage::EvaluateMovement;
        return yield();
      }
      if (attempt.nextMovement == attempt.movements.size()) {
        completeRegion(temporal);
        stage = Stage::SelectTemporal;
        return yield();
      }
      MovementChoice choice =
          std::move(attempt.movements[attempt.nextMovement++]);
      std::string detail;
      mlir::IRMapping mapping;
      auto candidate = cloneCandidate(
          choice.input ? *choice.input : *attempt.lowered, mapping, detail);
      if (mlir::failed(candidate))
        return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                           "search-movement-clone", detail));
      std::optional<AccessReuseChoice> mappedReuse;
      if (choice.reuse) {
        auto mapped = mapAccessReuseChoice(*choice.reuse, mapping);
        if (mlir::failed(mapped))
          return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                             "search-access-reuse-clone",
                             "clone omitted a selected read or scope"));
        mappedReuse = std::move(*mapped);
      }
      BoundaryMovementOptions movementOptions = choice.options;
      for (auto &component : movementOptions.components) {
        if (choice.input)
          break;
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
      BoundaryMovementResult movement;
      if (!choice.input) {
        movement = materializeTileBoundaryMovement(
            *candidate->module, candidate->relations, movementOptions);
        recordMovementInstrumentation(movement.statistics);
      }
      ExecutableCompilationResult compiled;
      std::optional<analysis::SearchObjective> evaluatedObjective;
      InputCapacityFeedback capacityFeedback;
      std::mutex feedbackMutex;
      if (!movement.succeeded()) {
        compiled =
            fail(movement.failure == BoundaryMovementFailureKind::Unsupported
                     ? ExecutableCompilationStatus::UnsupportedFailure
                     : ExecutableCompilationStatus::CompilerFailure,
                 "search-boundary-movement", movement.detail);
      } else if (!choice.input &&
                 ((choice.recursive &&
                   !movement.statistics.recursiveDoublingComponents) ||
                  (choice.allToAll &&
                   !movement.statistics.dimensionOrderedAllToAllComponents) ||
                  (choice.reduction &&
                   !movement.statistics.ringReduceScatterComponents &&
                   !movement.statistics.ringAllReduceComponents))) {
        compiled = fail(ExecutableCompilationStatus::UnsupportedFailure,
                        "search-boundary-movement",
                        "selected algorithm has no current component");
      } else {
        if (!choice.input && !choice.pipeline) {
          // BoundaryMovement owns function-boundary bufferization and load
          // creation. Analyze its actual output, never anticipated loads in
          // the open physical prefix.
          auto facts = analysis::analyzeAccessReuse(*candidate->module);
          auto reuse = proposeAccessReuse(
              facts, costCohort ? costCohort->getPolicy()
                                : analysis::SearchCostPolicy{});
          if (statistics) {
            ++statistics->accessReuseQueries;
            statistics->accessReuseEligible += reuse.opportunities;
            statistics->accessReuseQueued += reuse.choices.size();
            statistics->accessReuseLowBenefit += reuse.lowBenefit;
            statistics->accessReuseUnknownBenefit += reuse.unknownBenefit;
          }
          support::addCompileCounter("access-reuse", "scope-queries",
                                     facts.scopeQueries);
          support::addCompileCounter("access-reuse", "indeterminate-scopes",
                                     facts.indeterminateScopes);
          if (!reuse.choices.empty()) {
            auto prefix =
                std::make_shared<CurrentCandidate>(std::move(*candidate));
            auto insert = attempt.movements.begin() +
                          std::max(attempt.nextMovement, attempt.baseMovements);
            for (auto &selection : reuse.choices) {
              MovementChoice selected = choice;
              selected.input = prefix;
              selected.reuse = std::move(selection);
              insert = std::next(
                  attempt.movements.insert(insert, std::move(selected)));
            }
            // Keep the real boundary output as the siblings' input. The base
            // candidate uses the same ordinary clone operation; no trial
            // lowering is performed for profitability or memory prediction.
            mlir::IRMapping baseMapping;
            candidate = cloneCandidate(*prefix, baseMapping, detail);
            if (mlir::failed(candidate))
              return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                                 "search-access-reuse-prefix", detail));
            support::addCompileCounter("access-reuse", "retained-prefixes", 1);
            if (temporal.repairReuse) {
              auto selected = std::find_if(
                  attempt.movements.begin() + attempt.nextMovement,
                  attempt.movements.end(), [&](const auto &movement) {
                    return movement.input == prefix && movement.reuse &&
                           llvm::any_of(
                               movement.reuse->actions, [](const auto &action) {
                                 return action.kind != AccessReuseKind::Peer;
                               });
                  });
              if (selected != attempt.movements.end()) {
                MovementChoice base = choice;
                base.input = prefix;
                choice = std::move(*selected);
                *selected = std::move(base);
                auto mapped = mapAccessReuseChoice(*choice.reuse, baseMapping);
                if (mlir::failed(mapped))
                  return finish(
                      fail(ExecutableCompilationStatus::CompilerFailure,
                           "search-reuse-repair",
                           "clone omitted reuse repair anchors"));
                mappedReuse = std::move(*mapped);
                temporal.repairReuse = false;
              }
            }
          }
        }
        if (mappedReuse) {
          if (statistics)
            ++statistics->accessReuseCandidates;
          auto shared = materializeAccessReuse(
              *candidate->module, candidate->relations, *mappedReuse);
          if (!shared.succeeded())
            return finish(fail(
                shared.failure == AccessReuseFailureKind::Unsupported
                    ? ExecutableCompilationStatus::UnsupportedFailure
                    : (shared.failure == AccessReuseFailureKind::Indeterminate
                           ? ExecutableCompilationStatus::IndeterminateFailure
                           : ExecutableCompilationStatus::CompilerFailure),
                "search-access-reuse", shared.detail));
          recordMovementInstrumentation(shared.movement);
          support::addCompileCounter("access-reuse", "peer-receives",
                                     shared.movement.peerReceives);
          support::addCompileCounter("access-reuse", "resident-windows",
                                     shared.residentWindows);
          support::addCompileCounter("access-reuse", "sliding-windows",
                                     shared.slidingWindows);
          support::addCompileCounter("access-reuse", "two-level-windows",
                                     shared.twoLevelWindows);
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
        llvm::SmallVector<const TemporalDomain *> currentDomains;
        for (const auto &axis : axes)
          currentDomains.push_back(&axis.domain);
        auto observeCapacity = [&](CardId card, TileId tile,
                                   const SPMMemoryPlanningFailure &failure,
                                   const StructuredMaterializationRelations
                                       &relations) {
          if (options.downstream.capacityObserver)
            options.downstream.capacityObserver(card, tile, failure, relations);
          auto inputFeedback = deriveInputCapacityFeedback(
              card, tile, failure, currentDomains, temporal.choices);
          support::addCompileCounter("capacity-feedback", "input-coordinates",
                                     inputFeedback.coordinates.size());
          support::addCompileCounter("capacity-feedback",
                                     "unavailable-input-demands",
                                     inputFeedback.unavailableDemands);
          support::addCompileCounter("capacity-feedback", "ambiguous-inputs",
                                     inputFeedback.ambiguousInputs);
          support::addCompileCounter("capacity-feedback",
                                     "shared-input-demands",
                                     inputFeedback.sharedInputDemands);
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
          // Input-map evidence in one domain must not hide independently
          // proven owner/body evidence in other domains of this failure.
          for (const auto &demand : failure.individuallyOversizedDemands)
            observe(demand);
          for (const auto &demand : failure.capacityConflictDemands)
            observe(demand);
          for (size_t axis : localAxes) {
            // A witnessed owner in this actual Region is relevant even
            // when input provenance supplied only a subset of axes.
            // Fused producer/consumer scopes must be refined together;
            // dropping this evidence for multi-scope domains strands
            // output/psum conflicts behind the input-only feedback.
            auto descriptors = axes[axis].domain.getScopeDescriptors(
                temporal.choices[axis].kind);
            for (auto [scope, descriptor] : llvm::enumerate(descriptors))
              for (auto [iterator, capability] :
                   llvm::enumerate(descriptor.iteratorCapabilities))
                if (capability == IteratorTilingCapability::Tileable)
                  inputFeedback.coordinates.insert({axis, scope, iterator});
          }
          std::lock_guard<std::mutex> lock(feedbackMutex);
          capacityFeedback.coordinates.insert(inputFeedback.coordinates.begin(),
                                              inputFeedback.coordinates.end());
          if (inputFeedback.status == CapacityFeedbackStatus::BrokenContract &&
              (capacityFeedback.status !=
                   CapacityFeedbackStatus::BrokenContract ||
               inputFeedback.detail < capacityFeedback.detail)) {
            capacityFeedback.status = CapacityFeedbackStatus::BrokenContract;
            capacityFeedback.detail = std::move(inputFeedback.detail);
          }
        };
        CurrentIRDownstreamOptions downstreamOptions = options.downstream;
        downstreamOptions.communication =
            CommunicationProposalPolicy::DependencyOrdered;
        downstreamOptions.distanceOneLoadPipeline = choice.pipeline;
        if (choice.pipeline)
          support::addCompileCounter("search", "pipeline-candidates", 1);
        downstreamOptions.capacityObserver = observeCapacity;
        compiled = compileCurrentIRCandidateToExecutable(
            std::move(candidate->module), std::move(candidate->relations),
            planning.getProblem().getCardId(), analysis.availableTileIds,
            program, executionConfig, diagnostics, programData,
            downstreamOptions, &downstream, executableStatistics);
        if (capacityFeedback.status == CapacityFeedbackStatus::BrokenContract)
          return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                             "search-capacity-feedback",
                             capacityFeedback.detail));
        if (choice.pipeline && compiled.isAccepted())
          support::addCompileCounter("search", "pipeline-accepted", 1);
        if (statistics && choice.reuse && compiled.isAccepted()) {
          ++statistics->accessReuseAccepted;
          if (llvm::any_of(choice.reuse->actions, [](const auto &action) {
                return action.kind != AccessReuseKind::Peer;
              })) {
            ++statistics->residentReuseAccepted;
            const auto &reads =
                compiled.executable->resourceCost.aggregateDDRReadBytes;
            if (reads.isKnown())
              statistics->minimumResidentDDRReadBytes =
                  statistics->minimumResidentDDRReadBytes
                      ? std::min(*statistics->minimumResidentDDRReadBytes,
                                 reads.value)
                      : reads.value;
          }
        }
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
      if (choice.options.components.empty())
        temporal.observedTransports.insert(choice.options.transport);
      bool repairQueued = false;
      if (hasActualSPMCapacityRejection(compiled)) {
        if (statistics && choice.reuse)
          ++statistics->accessReuseCapacityRejected;
        temporal.capacityObserved |= !capacityFeedback.coordinates.empty();
        bool temporalReuse =
            choice.reuse &&
            llvm::any_of(choice.reuse->actions, [](const auto &action) {
              return action.kind != AccessReuseKind::Peer;
            });
        bool refined = proposals->observeCapacity(
            temporal.choices, capacityFeedback.coordinates, temporalReuse);
        repairQueued = refined;
        if (choice.reuse && refined &&
            llvm::any_of(choice.reuse->actions, [](const auto &action) {
              return action.kind != AccessReuseKind::Peer;
            }))
          repairReuse = true;
        if (statistics) {
          statistics->actualCapacityRefinements += refined;
          statistics->unavailableCapacityRefinements += !refined;
        }
      } else if (compiled.isAccepted()) {
        auto objective =
            deriveExecutableSearchObjective(*compiled.executable, costCohort);
        evaluatedObjective = objective;
        if (auto *known =
                std::get_if<analysis::KnownSearchObjective>(&objective)) {
          uint64_t duration = known->estimatedDurationPicoseconds;
          temporal.bestDuration =
              temporal.bestDuration ? std::min(*temporal.bestDuration, duration)
                                    : duration;
          attempt.bestDuration = attempt.bestDuration
                                     ? std::min(*attempt.bestDuration, duration)
                                     : duration;
          proposals->observeAccepted(temporal.choices, duration);
        }
      }
      if (!hasAcceptedCandidate && repairQueued && !realization &&
          proposals->prepareNext(TemporalProposalKind::Repair)) {
        // Keep the same actual prefix and all remaining alternatives, but
        // serve certified capacity repair before exploring those siblings.
        temporal.realizationOnly = true;
        realization.emplace(std::move(temporal));
        pending.pop_front();
      } else if (!temporal.realizationOnly && compiled.isAccepted() &&
                 (!realization || !realization->bestDuration ||
                  (temporal.bestDuration &&
                   *temporal.bestDuration < *realization->bestDuration))) {
        // Expose improvement immediately. The unvisited transport/closure
        // siblings retain this very same actual prefix in the one realization
        // slot; they no longer block another Temporal point. The executable
        // itself is handed to the controller below, never reconstructed.
        RegionAttempt invariant;
        invariant.layoutInput = attempt.layoutInput;
        invariant.placement = LayoutMaterializationPlacement::LoopInvariant;
        invariant.merged = attempt.merged;
        temporal.regions.push_back(std::move(invariant));
        temporal.realizationOnly = true;
        realization.emplace(std::move(temporal));
        pending.pop_front();
      } else if (temporal.realizationOnly && choice.reuse &&
                 compiled.isAccepted() &&
                 llvm::all_of(choice.reuse->actions,
                              [](const auto &action) {
                                return action.kind == AccessReuseKind::Peer;
                              }) &&
                 attempt.nextMovement < attempt.movements.size() &&
                 attempt.movements[attempt.nextMovement].reuse &&
                 llvm::any_of(
                     attempt.movements[attempt.nextMovement].reuse->actions,
                     [](const auto &action) {
                       return action.kind != AccessReuseKind::Peer;
                     })) {
        // Keep this actual prefix for its first temporal/peer combination.
        // A layout-placement sibling must not consume its turn first.
      } else if (temporal.realizationOnly) {
        if (temporal.bestDuration)
          proposals->observeAccepted(temporal.choices, *temporal.bestDuration);
        auto visited = std::move(temporal.regions.front());
        temporal.regions.pop_front();
        temporal.regions.push_back(std::move(visited));
        if (llvm::all_of(temporal.regions, [](const auto &region) {
              return bool(region.layoutInput);
            }))
          temporal.tiled = {};
        realization.emplace(std::move(temporal));
        pending.pop_front();
      } else {
        finishBasePoint(temporal);
      }
      return finish(std::move(compiled), std::move(evaluatedObjective));
    }
  }

private:
  enum class Stage { SelectTemporal, PrepareRegion, EvaluateMovement };
  enum class TemporalWork { Proposal, Repair, Improve, Resume };

  std::array<TemporalWork, 4> temporalWorkOrder() const {
    return hasAcceptedCandidate
               ? std::array{TemporalWork::Improve, TemporalWork::Resume,
                            TemporalWork::Repair, TemporalWork::Proposal}
               : std::array{TemporalWork::Repair, TemporalWork::Repair,
                            TemporalWork::Proposal, TemporalWork::Resume};
  }

  TemporalWork nextTemporalWork() {
    // Finish the current actualization before starting another Temporal IR.
    if (!pending.empty())
      return TemporalWork::Resume;
    if (repairReuse && proposals &&
        proposals->prepareNext(TemporalProposalKind::Repair))
      return TemporalWork::Repair;
    if (!hasAcceptedCandidate && proposals &&
        proposals->prepareNext(TemporalProposalKind::Repair))
      return TemporalWork::Repair;
    for (unsigned offset = 0; offset < 4; ++offset) {
      auto work = temporalWorkOrder()[(temporalPhase + offset) % 4];
      if ((work == TemporalWork::Proposal && proposals &&
           proposals->prepareNext(TemporalProposalKind::Explore)) ||
          (work == TemporalWork::Repair && proposals &&
           proposals->prepareNext(TemporalProposalKind::Repair)) ||
          (work == TemporalWork::Improve && proposals &&
           proposals->prepareNext(TemporalProposalKind::Improve)) ||
          (work == TemporalWork::Resume && realization))
        return work;
    }
    return TemporalWork::Proposal;
  }

  struct RegionPrepared {};
  struct RegionPreparationYielded {};
  using RegionPreparation =
      std::variant<RegionPrepared, RegionPreparationYielded,
                   ExecutableCompilationResult>;
  struct LayoutInput {
    CurrentCandidate candidate;
    std::unique_ptr<LayoutAssignmentQuery> query;
    ExactPBQPResult assignment;
  };
  struct RegionAttempt {
    std::optional<CurrentCandidate> lowered;
    // Both placements refer to one unchanged, owned input and one PBQP solve.
    // Applying either choice mutates only its exact IRMapping clone.
    std::shared_ptr<const LayoutInput> layoutInput;
    LayoutMaterializationPlacement placement =
        LayoutMaterializationPlacement::FirstUse;
    std::vector<MovementChoice> movements;
    size_t nextMovement = 0;
    size_t baseMovements = 0;
    std::optional<uint64_t> bestDuration;
    bool merged = false;
    bool baseComplete() const {
      return lowered && nextMovement >= baseMovements;
    }
  };
  struct TemporalAttempt {
    std::vector<TemporalChoice> choices;
    CurrentCandidate tiled;
    std::deque<RegionAttempt> regions;
    std::set<BoundaryMovementTransport> observedTransports;
    std::optional<uint64_t> bestDuration;
    bool realizationOnly = false;
    bool capacityObserved = false;
    bool repairReuse = false;
  };

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
    std::vector<const TemporalDomain *> domains;
    for (const auto &axis : axes)
      domains.push_back(&axis.domain);
    proposals.emplace(std::move(domains));
    proposals->seed(joint);
    if (hasFusion)
      proposals->seed(independent);
    return std::nullopt;
  }

  std::optional<ExecutableCompilationResult> startTemporal(TemporalWork work) {
    support::ScopedCompileTimingSpan timing("search-phase", "current-ir",
                                            "start-temporal");
    std::vector<TemporalChoice> choices;
    std::string detail;
    auto kind = work == TemporalWork::Repair    ? TemporalProposalKind::Repair
                : work == TemporalWork::Improve ? TemporalProposalKind::Improve
                                                : TemporalProposalKind::Explore;
    if (proposals->prepareNext(kind)) {
      choices = proposals->take(kind);
      support::addCompileCounter(
          "search",
          work == TemporalWork::Repair    ? "integer-repair-proposals"
          : work == TemporalWork::Improve ? "integer-improve-proposals"
                                          : "integer-explore-proposals",
          1);
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
        if (proposals->visitRaw(choices))
          break;
      }
    }
    if (statistics) {
      const auto number = statistics->temporalCandidateActualizations++;
      support::addCompileCounter(
          "search-temporal",
          llvm::formatv("choice-{0}-structural-stratum", number).str(),
          explorationStratum);
      for (auto [axis, choice] : llvm::enumerate(choices))
        for (auto [scope, selected] : llvm::enumerate(choice.scopes))
          for (auto [dimension, size] :
               llvm::enumerate(selected.iteratorTileSizes))
            support::addCompileCounter(
                "search-temporal",
                llvm::formatv("choice-{0}-domain-{1}-scope-{2}-dim-{3}", number,
                              axis, scope, dimension)
                    .str(),
                size);
    }
    mlir::IRMapping mapping;
    auto candidate = cloneCandidate(*structural, mapping, detail);
    if (mlir::failed(candidate))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-temporal-clone", detail);
    std::vector<TemporalDomain> remappedDomains;
    std::vector<TemporalChoice> remappedChoices;
    remappedDomains.reserve(axes.size());
    remappedChoices.reserve(axes.size());
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
      remappedDomains.push_back(std::move(*domain));
      remappedChoices.push_back(std::move(*remapped));
    }
    llvm::SmallVector<TemporalTilingRequest, 32> requests;
    for (auto [domain, choice] :
         llvm::zip_equal(remappedDomains, remappedChoices))
      requests.push_back({domain, choice});
    TemporalTilingFailure temporalFailure;
    if (mlir::failed(applyTemporalTiling(requests, candidate->relations,
                                         &temporalFailure)))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-temporal-apply", temporalFailure.detail);
    if (statistics)
      statistics->temporalApplications += requests.size();
    SpatialRegionMaterializationFailure failure;
    auto availability = analyzeCommunicationRegionClosure(
        *candidate->module, candidate->relations, &failure);
    if (mlir::failed(availability))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-communication-region-availability", failure.detail);
    const bool canMerge =
        *availability == CommunicationRegionClosureAvailability::Available;
    TemporalAttempt attempt;
    attempt.repairReuse =
        work == TemporalWork::Repair && std::exchange(repairReuse, false);
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
    support::ScopedCompileTimingSpan timing("search-phase", "current-ir",
                                            "prepare-region");
    std::string detail;
    if (!attempt.layoutInput) {
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
      support::addCompileCounter("search", "layout-pbqp-solves", 1);
      auto first = query.query->solve(options.layoutWorkLimit);
      if (first.status != ExactPBQPStatus::Optimal &&
          first.status != ExactPBQPStatus::Feasible)
        return fail(classifyLayoutFailure(first.status), "search-layout-query",
                    "current layout PBQP has no complete assignment");
      attempt.layoutInput = std::make_shared<LayoutInput>(LayoutInput{
          std::move(*candidate), std::move(query.query), std::move(first)});
      return RegionPreparationYielded{};
    }
    const auto &input = *attempt.layoutInput;
    const auto placement = attempt.placement;
    support::addCompileCounter("search",
                               placement ==
                                       LayoutMaterializationPlacement::FirstUse
                                   ? "first-use-placement-candidates"
                                   : "invariant-placement-candidates",
                               1);
    mlir::IRMapping mapping;
    auto candidate = cloneCandidate(input.candidate, mapping, detail);
    if (mlir::failed(candidate))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-layout-clone", detail);
    LayoutOptimizationResult layout;
    {
      support::ScopedCompileTimingSpan timing("stage", "current-ir-physical",
                                              "layout-and-bufferization");
      layout = input.query->apply(*candidate->module, candidate->relations,
                                  input.assignment, &mapping, placement);
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
    auto physicalPlacement = optimizePhysicalMovementPlacement(
        candidate->module->getOperation(), candidate->relations, placement);
    if (mlir::failed(physicalPlacement))
      return fail(ExecutableCompilationStatus::CompilerFailure,
                  "search-physical-movement-placement",
                  "physical movement placement produced invalid current IR");
    support::addCompileCounter("movement", "invariant-physical-copies",
                               *physicalPlacement);
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
    // Transport preferences seed actual construction. Component combinations
    // are selected only by the unified ready frontier, not a Cartesian mask.
    attempt.movements.push_back({});
    if (distributed.sharedDDR) {
      MovementChoice ddr;
      ddr.options.transport = BoundaryMovementTransport::SharedDDR;
      attempt.movements.push_back(ddr);
    }
    attempt.baseMovements = attempt.movements.size();
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
    }
    attempt.lowered.emplace(std::move(*candidate));
    return RegionPrepared{};
  }

  void completeRegion(TemporalAttempt &temporal) {
    support::ScopedCompileTimingSpan timing("search-phase", "current-ir",
                                            "complete-region");
    temporal.regions.pop_front();
    if (temporal.regions.empty())
      pending.pop_front();
    else if (!temporal.realizationOnly)
      finishBasePoint(temporal);
  }

  void finishBasePoint(TemporalAttempt &temporal) {
    if (!llvm::all_of(temporal.regions, [](const auto &region) {
          return region.baseComplete();
        })) {
      while (temporal.regions.front().baseComplete()) {
        auto region = std::move(temporal.regions.front());
        temporal.regions.pop_front();
        temporal.regions.push_back(std::move(region));
      }
      return;
    }
    if (temporal.bestDuration)
      proposals->observeAccepted(temporal.choices, *temporal.bestDuration);
    // Keep one actual local realization anchor, never every failed prefix.
    // Its component cursors keep their owning current post-layout IR alive.
    for (auto &region : temporal.regions) {
      // Physical copies can become invariant during StructuredToTile even
      // when the layout query itself has no invariant materialization.
      // Keep this actual placement alternative for every retained anchor.
      if (realization && realization->bestDuration &&
          (!region.bestDuration ||
           *realization->bestDuration <= *region.bestDuration))
        continue;
      TemporalAttempt anchor;
      anchor.choices = temporal.choices;
      anchor.bestDuration = region.bestDuration;
      anchor.realizationOnly = true;
      RegionAttempt invariant;
      invariant.layoutInput = region.layoutInput;
      invariant.placement = LayoutMaterializationPlacement::LoopInvariant;
      invariant.merged = region.merged;
      const bool sharingQueued = llvm::any_of(
          llvm::ArrayRef(region.movements).drop_front(region.nextMovement),
          [](const auto &movement) { return movement.reuse.has_value(); });
      anchor.regions.push_back(std::move(region));
      if (sharingQueued)
        anchor.regions.push_back(std::move(invariant));
      else
        anchor.regions.push_front(std::move(invariant));
      realization.emplace(std::move(anchor));
    }
    pending.pop_front();
  }

  void recordOwnership() {
    if (statistics) {
      uint64_t modules = bool(structural);
      std::set<const LayoutInput *> inputs;
      std::set<const CurrentCandidate *> reuseInputs;
      auto count = [&](const TemporalAttempt &temporal) {
        modules += bool(temporal.tiled.module);
        for (const auto &region : temporal.regions) {
          modules += bool(region.lowered);
          if (region.layoutInput)
            inputs.insert(region.layoutInput.get());
          for (const auto &movement : region.movements)
            if (movement.input)
              reuseInputs.insert(movement.input.get());
        }
      };
      for (const auto &temporal : pending)
        count(temporal);
      if (realization)
        count(*realization);
      modules += inputs.size() + reuseInputs.size();
      statistics->peakSessionTemporalPrefixes = std::max<uint64_t>(
          statistics->peakSessionTemporalPrefixes, pending.size());
      statistics->peakSessionIRModules =
          std::max(statistics->peakSessionIRModules, modules);
    }
  }

  StructuralCandidateEvaluation yield() {
    recordOwnership();
    return {{},
            0,
            currentContinuation,
            CandidateRetention::UnfinishedActualization};
  }

  StructuralCandidateEvaluation finish(
      ExecutableCompilationResult compiled,
      std::optional<analysis::SearchObjective> objective = std::nullopt) {
    recordOwnership();
    stage = Stage::SelectTemporal;
    const auto status = classifyActualStatus(compiled.status);
    if (!hasAcceptedCandidate && compiled.isAccepted()) {
      hasAcceptedCandidate = true;
      temporalPhase = 0;
    }
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
    if (objective)
      result.objective = ActualCandidateResult::EvaluatedObjective{
          costCohort, std::move(*objective)};
    if (compiled.isAccepted() || compiled.isProvenExactRejection())
      result.compilation.emplace(std::move(compiled));
    TemporalWork next = nextTemporalWork();
    return {std::move(result), 1,
            exhausted ? CandidateContinuation::Exhausted
            : next == TemporalWork::Repair ||
                    (next == TemporalWork::Resume && !pending.empty() &&
                     !hasAcceptedCandidate && pending.front().capacityObserved)
                ? CandidateContinuation::Repair
            : next == TemporalWork::Improve ||
                    (next == TemporalWork::Resume && hasAcceptedCandidate)
                ? CandidateContinuation::Improve
                : CandidateContinuation::Explore,
            !pending.empty() ? CandidateRetention::UnfinishedActualization
            : proposals && proposals->hasCapacityRoundInProgress()
                ? CandidateRetention::PendingCapacityRepair
                : CandidateRetention::Replaceable};
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
  const std::optional<analysis::SearchCostCohort> &costCohort;
  std::optional<CurrentCandidate> structural;
  std::vector<TemporalAxis> axes;
  std::optional<TemporalProposals> proposals;
  std::deque<TemporalAttempt> pending;
  std::optional<TemporalAttempt> realization;
  const size_t explorationStratum;
  unsigned temporalPhase = 0;
  bool repairReuse = false;
  Stage stage = Stage::SelectTemporal;
  CandidateContinuation currentContinuation = CandidateContinuation::Explore;
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
      ExecutableLoweringStatistics *executableStatistics,
      const std::optional<analysis::SearchCostCohort> &costCohort)
      : tensorProgram(tensorProgram), analysis(analysis), planning(planning),
        program(program), executionConfig(executionConfig),
        diagnostics(diagnostics), programData(programData), options(options),
        statistics(statistics), executableStatistics(executableStatistics),
        costCohort(costCohort) {}

  std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &state) override {
    return std::make_unique<CurrentIRCandidateSession>(
        state, tensorProgram, analysis, planning, program, executionConfig,
        diagnostics, programData, options, statistics, executableStatistics,
        costCohort, nextExplorationStratum++);
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
  const std::optional<analysis::SearchCostCohort> &costCohort;
  size_t nextExplorationStratum = 0;
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
  if (mlir::failed(foldContractionInitializers(*function)))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "contraction-initializer", "cannot fold constant initializer");
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
  CurrentIRStructuralEvaluator evaluator(tensorProgram, **structured, planning,
                                         program, executionConfig, diagnostics,
                                         programData, options, statistics,
                                         executableStatistics, currentCohort);

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
  searchCounter("stage-yields", searched.work.stageYields);
  searchCounter("retired-branches", searched.work.retiredBranches);
  searchCounter("local-region-refinements",
                searched.work.localRegionRefinements);
  searchCounter("non-incumbent-region-refinements",
                searched.work.nonIncumbentRegionRefinements);
  searchCounter("candidate-actualizations",
                searched.work.candidateActualizations);
  searchCounter("width", options.limits.width);
  searchCounter("trials", options.limits.trials);
  searchCounter("trials-used", searched.work.candidateActualizations);
  if (statistics) {
    searchCounter("peak-session-temporal-prefixes",
                  statistics->peakSessionTemporalPrefixes);
    searchCounter("peak-session-ir-modules", statistics->peakSessionIRModules);
    searchCounter("temporal-backpressure-turns",
                  statistics->temporalBackpressureTurns);
  }
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
    searchCounter("access-reuse-candidates", statistics->accessReuseCandidates);
    searchCounter("access-reuse-queries", statistics->accessReuseQueries);
    searchCounter("access-reuse-low-benefit",
                  statistics->accessReuseLowBenefit);
    searchCounter("access-reuse-unknown-benefit",
                  statistics->accessReuseUnknownBenefit);
    searchCounter("access-reuse-eligible", statistics->accessReuseEligible);
    searchCounter("access-reuse-queued", statistics->accessReuseQueued);
    searchCounter("access-reuse-accepted", statistics->accessReuseAccepted);
    searchCounter("resident-reuse-accepted", statistics->residentReuseAccepted);
    if (statistics->minimumResidentDDRReadBytes)
      searchCounter("resident-minimum-ddr-read-bytes",
                    *statistics->minimumResidentDDRReadBytes);
    searchCounter("access-reuse-capacity-rejected",
                  statistics->accessReuseCapacityRejected);
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
