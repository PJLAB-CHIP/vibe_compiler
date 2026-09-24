//===- SearchCurrentIR.cpp - Current-IR physical search ----------------===//

#include "SearchCurrentIR.h"

#include "CapacityFeedback.h"
#include "PhysicalDataflowInstrumentation.h"
#include "StructuredProgramAnalysis.h"
#include "TemporalProposals.h"
#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningProblem.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Planning/PhysicalDataflow/TensorAssemblySelection.h"
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
#include "Wafer/Transforms/Tile/ScalarExecution.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <deque>
#include <iterator>
#include <list>
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
  TensorChoiceBindings tensorChoices;
};

struct TemporalAxis {
  TemporalDomain domain;
  TemporalSuccessor current;
};

struct LayoutInput {
  CurrentCandidate candidate;
  std::unique_ptr<LayoutAssignmentQuery> query;
  ExactPBQPResult assignment;
};

struct LayoutPrefix {
  TensorAssemblyIntent assembly;
  std::shared_ptr<const LayoutInput> input;
  llvm::SmallVector<TensorAssemblySelection, 4> opportunities;
  std::array<std::shared_ptr<const CurrentCandidate>, 2> prepared;
};

/// Only already-materialized owners live here. Children are filled once and
/// never mutated; each transformation clones its nearest actual ancestor.
struct TemporalPrefix : llvm::FoldingSetNode {
  mlir::Operation *parent = nullptr;
  std::vector<TemporalChoice> choices;
  CurrentCandidate tiled;
  bool canMerge = false;
  // One optional slot per closure/scalar combination. A different assembly
  // intent replaces the slot, so arbitrary group combinations do not multiply
  // the bounded prefix-owner count.
  std::array<LayoutPrefix, 4> layouts;

  static void profile(llvm::FoldingSetNodeID &id, mlir::Operation *parent,
                      const std::vector<TemporalChoice> &choices) {
    id.AddPointer(parent);
    id.AddInteger(choices.size());
    for (const auto &choice : choices) {
      id.AddInteger(static_cast<unsigned>(choice.kind));
      id.AddInteger(choice.scopes.size());
      for (const auto &scope : choice.scopes) {
        id.AddPointer(scope.operation);
        id.AddInteger(scope.iteratorTileSizes.size());
        for (int64_t size : scope.iteratorTileSizes)
          id.AddInteger(size);
        id.AddInteger(scope.loopOrder.size());
        for (unsigned axis : scope.loopOrder)
          id.AddInteger(axis);
      }
    }
  }
  void Profile(llvm::FoldingSetNodeID &id) const {
    profile(id, parent, choices);
  }

  void collectModules(llvm::SmallPtrSetImpl<mlir::Operation *> &owners) const {
    owners.insert(tiled.module.get().getOperation());
    for (const auto &layout : layouts) {
      if (layout.input)
        owners.insert(layout.input->candidate.module.get().getOperation());
      for (const auto &candidate : layout.prepared)
        if (candidate)
          owners.insert(candidate->module.get().getOperation());
    }
  }
};

/// One cache for the whole search, not width entries in each structural
/// session. An in-flight attempt pins its entry across eviction. Parent
/// sessions remove their keys before destroying the immutable structural IR.
class CurrentIRPrefixCache {
public:
  CurrentIRPrefixCache(uint64_t capacity, SearchCurrentIRStatistics *statistics)
      : capacity(capacity), statistics(statistics) {}

  std::shared_ptr<TemporalPrefix>
  find(mlir::Operation *parent, const std::vector<TemporalChoice> &choices) {
    llvm::FoldingSetNodeID id;
    TemporalPrefix::profile(id, parent, choices);
    void *position = nullptr;
    auto *found = index.FindNodeOrInsertPos(id, position);
    if (!found)
      return {};
    auto it = llvm::find_if(
        entries, [&](const auto &entry) { return entry.get() == found; });
    auto result = *it;
    entries.splice(entries.end(), entries, it);
    return result;
  }

  void retain(std::shared_ptr<TemporalPrefix> prefix) {
    if (!capacity)
      return;
    if (entries.size() == capacity) {
      index.RemoveNode(entries.front().get());
      entries.pop_front();
      if (statistics)
        ++statistics->prefixEvictions;
    }
    index.InsertNode(prefix.get());
    entries.push_back(std::move(prefix));
    if (statistics)
      statistics->peakCachedPrefixes =
          std::max<uint64_t>(statistics->peakCachedPrefixes, entries.size());
  }

  void erase(mlir::Operation *parent) {
    for (auto it = entries.begin(); it != entries.end();) {
      if ((*it)->parent != parent) {
        ++it;
        continue;
      }
      index.RemoveNode(it->get());
      it = entries.erase(it);
    }
  }

  void collectModules(mlir::Operation *parent,
                      llvm::SmallPtrSetImpl<mlir::Operation *> &owners) const {
    for (const auto &entry : entries)
      if (entry->parent == parent)
        entry->collectModules(owners);
  }

private:
  const uint64_t capacity;
  SearchCurrentIRStatistics *statistics;
  llvm::FoldingSet<TemporalPrefix> index;
  std::list<std::shared_ptr<TemporalPrefix>> entries;
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
  auto tensorChoices = source.tensorChoices.clone(mapping);
  if (mlir::failed(tensorChoices)) {
    detail = "candidate clone omitted a current Tensor choice binding";
    return mlir::failure();
  }
  result->tensorChoices = std::move(*tensorChoices);
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

static bool advanceTemporalAxes(const std::vector<TemporalAxis> &axes,
                                std::vector<TemporalSuccessor> &current,
                                std::string &detail, bool &compilerBug) {
  for (size_t offset = 0; offset < axes.size(); ++offset) {
    const size_t index = axes.size() - offset - 1;
    const TemporalAxis &axis = axes[index];
    const TemporalCursor *cursor = current[index].getCursor();
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
      current[index] = std::move(next);
      for (size_t reset = index + 1; reset < axes.size(); ++reset) {
        current[reset] = axes[reset].domain.getFirstChoice();
        if (current[reset].getKind() != TemporalSuccessorKind::Choice) {
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
  total.pipelinedLoops += value.pipelinedLoops;
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

struct PipelineScope {
  uint64_t card, tile;
  IterationCoordinatesAttr coordinates;
  friend bool operator==(const PipelineScope &a, const PipelineScope &b) {
    return a.card == b.card && a.tile == b.tile &&
           a.coordinates == b.coordinates;
  }
};

struct ImplementationChoice {
  BoundaryMovementOptions options;
  bool recursive = false;
  bool allToAll = false;
  bool reduction = false;
  bool pipeline = false;
  llvm::SmallVector<PipelineScope, 4> pipelineScopes;
  bool merged = false;
  bool cpuScalars = false;
  TensorAssemblyIntent assembly;
  LayoutMaterializationPlacement placement =
      LayoutMaterializationPlacement::FirstUse;
  std::optional<AccessReuseIntent> reuse;

  friend bool operator==(const ImplementationChoice &a,
                         const ImplementationChoice &b) {
    return a.options.allGather == b.options.allGather &&
           a.options.allToAll == b.options.allToAll &&
           a.options.reduction == b.options.reduction &&
           a.options.transport == b.options.transport &&
           a.recursive == b.recursive && a.allToAll == b.allToAll &&
           a.reduction == b.reduction && a.pipeline == b.pipeline &&
           a.pipelineScopes.size() == b.pipelineScopes.size() &&
           llvm::all_of(a.pipelineScopes,
                        [&](const auto &scope) {
                          return llvm::is_contained(b.pipelineScopes, scope);
                        }) &&
           a.merged == b.merged && a.cpuScalars == b.cpuScalars &&
           a.assembly == b.assembly && a.placement == b.placement &&
           a.reuse == b.reuse;
  }
};

/// TemporalTiling is the only producer of iteration-coordinate annotations.
/// Later stages preserve or erase these loop identities; they cannot restore
/// a selected identity absent from the actual tiled owner. This is only a
/// necessary condition: read windows, effects and participants are still
/// checked after their actual materialization.
bool hasImplementationScopes(mlir::ModuleOp module,
                             const ImplementationChoice &choice) {
  if (!choice.pipeline && !choice.reuse)
    return true;
  llvm::SmallVector<PipelineScope, 16> scopes;
  module.walk([&](mlir::scf::ForOp loop) {
    auto coordinates = getIterationCoordinates(loop);
    auto tile = loop->getParentOfType<TileModuleOp>();
    if (coordinates && tile)
      scopes.push_back({tile.getCardId(), tile.getTileId(), coordinates});
  });
  for (const auto &scope : choice.pipelineScopes)
    if (!llvm::is_contained(scopes, scope))
      return false;
  if (choice.reuse)
    for (const auto &selection : choice.reuse->selections) {
      if (selection.kind == AccessReuseKind::Peer)
        continue;
      auto contains = [&](IterationCoordinatesAttr coordinates) {
        return llvm::any_of(scopes, [&](const auto &scope) {
          return selection.card >= 0 && selection.tile >= 0 &&
                 scope.card == static_cast<uint64_t>(selection.card) &&
                 scope.tile == static_cast<uint64_t>(selection.tile) &&
                 scope.coordinates == coordinates;
        });
      };
      if (!contains(selection.scope) ||
          (selection.innerScope && !contains(selection.innerScope)))
        return false;
    }
  return true;
}

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
      size_t explorationStratum, CurrentIRPrefixCache &prefixCache)
      : state(std::move(state)), tensorProgram(tensorProgram),
        analysis(analysis), planning(planning), program(program),
        executionConfig(executionConfig), diagnostics(diagnostics),
        programData(programData), options(options), statistics(statistics),
        executableStatistics(executableStatistics), costCohort(costCohort),
        explorationStratum(explorationStratum), prefixCache(prefixCache) {}

  ~CurrentIRCandidateSession() override {
    if (structural)
      prefixCache.erase(structural->module->getOperation());
  }

  StructuralCandidateEvaluation
  advance(CandidateAdvanceLimits limits) override {
    stepSchemesStarted = 0;
    stepSchemesCompleted = 0;
    retentionLimit = limits.retainedBranches;
    if (exhausted)
      return closed();
    if (options.deadline &&
        std::chrono::steady_clock::now() >= *options.deadline) {
      exhausted = true;
      auto result = StructuralCandidateEvaluation{
          actualFailure(ActualCandidateStatus::Indeterminate,
                        "search wall-time budget exhausted"),
          0};
      result.completeDomain = false;
      return result;
    }
    if (!structural) {
      if (!limits.mayStartScheme)
        return blocked();
      implementations.push_back(std::make_unique<ImplementationBranch>());
      active = 0;
      implementations.front()->started = true;
      stepSchemesStarted = 1;
      auto failure = initialize();
      if (failure)
        return finish(std::move(*failure));
      activate(0);
      return yield();
    }
    if (!pending) {
      if (!selectImplementation(limits.mayStartScheme)) {
        if (budgetBlocked)
          return blocked();
        exhausted = true;
        return closed();
      }
      auto failure = startTemporal();
      if (failure)
        return finish(std::move(*failure));
      if (!pending)
        return yield();
      return yield();
    }
    auto &temporal = *pending;
    auto &attempt = temporal.region;
    if (!attempt.lowered) {
      if (attempt.layoutInput)
        discoverLocalAssembly(temporal);
      auto prepared = prepareRegion(temporal, attempt);
      if (auto *failure = std::get_if<ExecutableCompilationResult>(&prepared))
        return finish(std::move(*failure));
      return yield();
    }
    const auto choice = temporal.choice;
    std::string detail;
    mlir::IRMapping mapping;
    auto candidate = cloneCandidate(*attempt.lowered, mapping, detail);
    if (mlir::failed(candidate))
      return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                         "search-movement-clone", detail));
    auto movement = materializeTileBoundaryMovement(
        *candidate->module, candidate->relations, choice.options);
    recordMovementInstrumentation(movement.statistics);
    if (!movement.succeeded())
      return finish(
          fail(movement.failure == BoundaryMovementFailureKind::Unsupported
                   ? ExecutableCompilationStatus::UnsupportedFailure
                   : ExecutableCompilationStatus::CompilerFailure,
               "search-boundary-movement", movement.detail));
    if (attempt.placement == LayoutMaterializationPlacement::FirstUse &&
        hasInvariantPhysicalMovement(candidate->module.get().getOperation())) {
      auto sibling = choice;
      sibling.placement = LayoutMaterializationPlacement::LoopInvariant;
      discover(std::move(sibling), temporal.choices);
    }
    // Boundary movement has now materialized the actual DPS loads. Apply the
    // same placement choice to them and their dependent conversions.
    auto physicalPlacement = optimizePhysicalMovementPlacement(
        candidate->module.get().getOperation(), candidate->relations,
        attempt.placement);
    if (mlir::failed(physicalPlacement))
      return finish(
          fail(ExecutableCompilationStatus::CompilerFailure,
               "search-physical-movement-placement",
               "physical movement placement produced invalid current IR"));
    support::addCompileCounter("movement", "invariant-physical-copies",
                               *physicalPlacement);
    if ((choice.recursive &&
         !movement.statistics.recursiveDoublingComponents) ||
        (choice.allToAll &&
         !movement.statistics.dimensionOrderedAllToAllComponents) ||
        (choice.reduction && !movement.statistics.ringReduceScatterComponents &&
         !movement.statistics.ringAllReduceComponents))
      return finish(fail(ExecutableCompilationStatus::UnsupportedFailure,
                         "search-boundary-movement",
                         "selected algorithm has no current component"));

    // Discovery happens on actual loads, before any SPM/target result. The
    // logical selection is retained independently of this base point's cost.
    auto facts = analysis::analyzeAccessReuse(*candidate->module);
    if (!choice.reuse && !choice.pipeline) {
      auto reuse =
          proposeAccessReuse(facts, costCohort ? costCohort->getPolicy()
                                               : analysis::SearchCostPolicy{});
      if (statistics) {
        ++statistics->accessReuseQueries;
        statistics->accessReuseEligible += reuse.opportunities;
        statistics->accessReuseLowBenefit += reuse.lowBenefit;
        statistics->accessReuseUnknownBenefit += reuse.unknownBenefit;
      }
      support::addCompileCounter("access-reuse", "scope-queries",
                                 facts.scopeQueries);
      support::addCompileCounter("access-reuse", "indeterminate-scopes",
                                 facts.indeterminateScopes);
      for (const auto &selection : reuse.choices) {
        auto captured = captureAccessReuse(selection.choice, facts);
        if (auto *failure = std::get_if<AccessReuseBindingFailure>(&captured)) {
          if (failure->status == analysis::IndexRelationStatus::Invalid)
            return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                               "search-reuse-selection", failure->detail));
          support::addCompileCounter("access-reuse", "unbound-selections", 1);
          continue;
        }
        auto sibling = choice;
        sibling.reuse = std::get<AccessReuseIntent>(std::move(captured));
        discover(std::move(sibling), temporal.choices,
                 selection.estimatedBenefitPicoseconds);
      }
    }
    if (choice.reuse) {
      auto bound = bindAccessReuse(*choice.reuse, facts);
      if (auto *failure = std::get_if<AccessReuseBindingFailure>(&bound))
        return finish(
            fail(failure->status == analysis::IndexRelationStatus::Invalid
                     ? ExecutableCompilationStatus::CompilerFailure
                 : failure->status ==
                         analysis::IndexRelationStatus::ResourceExhausted
                     ? ExecutableCompilationStatus::IndeterminateFailure
                     : ExecutableCompilationStatus::UnsupportedFailure,
                 "search-reuse-binding", failure->detail));
      auto shared =
          materializeAccessReuse(*candidate->module, candidate->relations,
                                 std::get<AccessReuseChoice>(bound));
      if (!shared.succeeded())
        return finish(
            fail(shared.failure == AccessReuseFailureKind::Unsupported
                     ? ExecutableCompilationStatus::UnsupportedFailure
                 : shared.failure == AccessReuseFailureKind::Indeterminate
                     ? ExecutableCompilationStatus::IndeterminateFailure
                     : ExecutableCompilationStatus::CompilerFailure,
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
    if (mlir::failed(
            optimizeCurrentIRStorage(*candidate->module, candidate->relations)))
      return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                         "storage-optimization",
                         "common storage optimization failed"));
    auto pipelineQuery = queryDistanceOneLoadPipelines(*candidate->module);
    if (pipelineQuery.failure &&
        pipelineQuery.failure->kind != LoopPipeliningFailureKind::Unsupported)
      return finish(fail(pipelineQuery.failure->kind ==
                                 LoopPipeliningFailureKind::Indeterminate
                             ? ExecutableCompilationStatus::IndeterminateFailure
                             : ExecutableCompilationStatus::CompilerFailure,
                         "search-pipeline", pipelineQuery.failure->detail));
    auto &pipelineLoops = pipelineQuery.loops;
    llvm::SmallVector<PipelineScope, 4> pipelineScopes;
    bool completePipelineScopes = true;
    for (auto loop : pipelineLoops) {
      auto tile = loop->getParentOfType<TileModuleOp>();
      auto coordinates = getIterationCoordinates(loop);
      if (!tile || !coordinates) {
        completePipelineScopes = false;
        continue;
      }
      PipelineScope scope{tile.getCardId(), tile.getTileId(), coordinates};
      if (!llvm::is_contained(pipelineScopes, scope))
        pipelineScopes.push_back(scope);
    }
    const bool pipelineAvailable =
        completePipelineScopes && !pipelineScopes.empty();
    if (choice.pipeline &&
        (!pipelineAvailable ||
         pipelineScopes.size() != choice.pipelineScopes.size() ||
         !llvm::all_of(pipelineScopes, [&](const auto &scope) {
           return llvm::is_contained(choice.pipelineScopes, scope);
         })))
      return finish(fail(
          ExecutableCompilationStatus::UnsupportedFailure, "search-pipeline",
          "selected pipeline is unavailable at this tile point"));
    if (!choice.pipeline && pipelineAvailable) {
      auto sibling = choice;
      sibling.pipeline = true;
      sibling.pipelineScopes = std::move(pipelineScopes);
      discover(std::move(sibling), temporal.choices);
    }
    if (statistics) {
      statistics->recursiveDoublingCandidates += choice.recursive;
      statistics->dimensionOrderedAllToAllCandidates += choice.allToAll;
      statistics->distributedRingCandidates += choice.reduction;
      statistics->sharedDDRCandidates +=
          choice.options.transport == BoundaryMovementTransport::SharedDDR;
    }
    ExecutableCompilationResult compiled;
    std::optional<analysis::SearchObjective> evaluatedObjective;
    InputCapacityFeedback capacityFeedback;
    std::mutex feedbackMutex;
    CurrentIRDownstreamStatistics downstream;
    // SCF pipelining replaces loop bodies. Their earlier capacity-owner
    // scope handles do not survive this actual transformation.
    if (choice.pipeline)
      candidate->temporalBodies.clear();
    const auto temporalBodies = candidate->temporalBodies;
    llvm::SmallVector<const TemporalDomain *> currentDomains;
    for (const auto &axis : axes)
      currentDomains.push_back(&axis.domain);
    auto observeCapacity =
        [&](CardId card, TileId tile, const SPMMemoryPlanningFailure &failure,
            const StructuredMaterializationRelations &relations) {
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
    downstreamOptions.movementPlacement = attempt.placement;
    downstreamOptions.communication =
        CommunicationProposalPolicy::DependencyOrdered;
    downstreamOptions.distanceOneLoadPipeline = choice.pipeline;
    if (choice.pipeline)
      support::addCompileCounter("search", "pipeline-candidates", 1);
    downstreamOptions.capacityObserver = observeCapacity;
    compiled = compileCurrentIRCandidateToExecutable(
        std::move(candidate->module), std::move(candidate->relations),
        planning.getProblem().getCardId(), analysis.availableTileIds, program,
        executionConfig, diagnostics, programData, downstreamOptions,
        &downstream, executableStatistics);
    if (capacityFeedback.status == CapacityFeedbackStatus::BrokenContract)
      return finish(fail(ExecutableCompilationStatus::CompilerFailure,
                         "search-capacity-feedback", capacityFeedback.detail));
    if (choice.pipeline && compiled.isAccepted())
      support::addCompileCounter("search", "pipeline-accepted", 1);
    if (!choice.assembly.selections.empty() && compiled.isAccepted())
      support::addCompileCounter("search", "local-assembly-accepted", 1);
    if (statistics && !choice.assembly.selections.empty() &&
        compiled.isAccepted())
      ++statistics->assemblyAccepted;
    if (statistics && compiled.isAccepted() &&
        !choice.assembly.selections.empty() &&
        choice.assembly.selections.size() <
            attempt.assemblyOpportunities.size())
      ++statistics->assemblyMixedAccepted;
    if (statistics && choice.reuse && compiled.isAccepted()) {
      ++statistics->accessReuseAccepted;
      if (llvm::any_of(choice.reuse->selections, [](const auto &action) {
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
            choice.options.transport == BoundaryMovementTransport::SharedDDR;
        statistics->mergedRegionAccepted += attempt.merged;
        statistics->regionPreservingAccepted += !attempt.merged;
      }
    }

    if (hasActualSPMCapacityRejection(compiled)) {
      if (statistics && choice.reuse)
        ++statistics->accessReuseCapacityRejected;
      bool refined = current().proposals->observeCapacity(
          temporal.choices, capacityFeedback.coordinates, true);
      if (statistics) {
        statistics->actualCapacityRefinements += refined;
        statistics->unavailableCapacityRefinements += !refined;
        if (!choice.assembly.selections.empty()) {
          ++statistics->localAssemblyCapacityRejected;
          statistics->localAssemblyCapacityRefinements += refined;
        } else if (!attempt.assemblyOpportunities.empty()) {
          ++statistics->sharedAssemblyCapacityRejected;
          statistics->sharedAssemblyCapacityRefinements += refined;
        }
      }
      if (support::getActiveCompileTimingSession())
        diagnostics << "wafer-compile: capacity-feedback structural="
                    << explorationStratum << " implementation=" << *active
                    << " local_assembly_groups="
                    << choice.assembly.selections.size()
                    << " refined=" << refined << '\n';
    } else if (compiled.isAccepted()) {
      auto objective =
          deriveExecutableSearchObjective(*compiled.executable, costCohort);
      evaluatedObjective = objective;
      if (auto *known =
              std::get_if<analysis::KnownSearchObjective>(&objective)) {
        auto duration = known->estimatedDurationPicoseconds;
        if (support::getActiveCompileTimingSession())
          diagnostics << "wafer-compile: accepted-candidate pipeline="
                      << choice.pipeline << " local_assembly_groups="
                      << choice.assembly.selections.size()
                      << " estimated_picoseconds=" << duration << '\n';
        current().bestDuration =
            current().bestDuration ? std::min(*current().bestDuration, duration)
                                   : duration;
        current().proposals->observeAccepted(temporal.choices, *known);
        // Preserve a qualified source point for each unstarted sibling. A
        // later discovery has not passed this sibling's actual leaf and must
        // not displace the point solely because it was discovered last.
        for (auto &branch : implementations) {
          if (branch->state != BranchState::Waiting ||
              (branch->discoveryDuration &&
               *branch->discoveryDuration <= duration))
            continue;
          auto point = llvm::find(branch->discovery, temporal.choices);
          if (point == branch->discovery.end())
            continue;
          std::rotate(branch->discovery.begin(), point, std::next(point));
          branch->discoveryDuration = duration;
        }
      }
    }
    return finish(std::move(compiled), std::move(evaluatedObjective));
  }

  uint64_t getRetainedBranchCount() const override {
    return std::max<uint64_t>(
        1, llvm::count_if(implementations, [](const auto &branch) {
          return bool(branch->proposals);
        }));
  }

  bool hasOpenTrial() const override {
    return options.mode == SearchMode::Deep && !exhausted &&
           llvm::any_of(implementations, [](const auto &branch) {
             return branch->started && branch->proposals;
           });
  }

private:
  enum class BranchState { Waiting, Active, Complete, Retired };
  struct ImplementationBranch {
    ImplementationChoice choice;
    std::vector<std::vector<TemporalChoice>> discovery;
    std::optional<uint64_t> discoveryDuration;
    long double priority = 0;
    std::unique_ptr<TemporalProposals> proposals;
    std::vector<TemporalSuccessor> raw;
    std::optional<uint64_t> bestDuration;
    BranchState state = BranchState::Waiting;
    unsigned phase = 0;
    uint64_t lastVisit = 0;
    bool started = false;
  };
  struct RegionPrepared {};
  struct RegionPreparationYielded {};
  using RegionPreparation =
      std::variant<RegionPrepared, RegionPreparationYielded,
                   ExecutableCompilationResult>;
  struct RegionAttempt {
    std::shared_ptr<const CurrentCandidate> lowered;
    std::shared_ptr<const LayoutInput> layoutInput;
    LayoutMaterializationPlacement placement =
        LayoutMaterializationPlacement::FirstUse;
    bool merged = false;
    bool cpuScalars = false;
    TensorAssemblyIntent assembly;
    llvm::SmallVector<TensorAssemblySelection, 4> assemblyOpportunities;
    unsigned layoutIndex() const { return 2 * merged + cpuScalars; }
  };
  struct TemporalAttempt {
    std::vector<TemporalChoice> choices;
    std::shared_ptr<TemporalPrefix> prefix;
    ImplementationChoice choice;
    RegionAttempt region;
  };

  ImplementationBranch &current() { return *implementations[*active]; }

  void discoverLocalAssembly(const TemporalAttempt &temporal) {
    for (auto intent : extendTensorAssemblyIntent(
             temporal.choice.assembly, temporal.region.assemblyOpportunities)) {
      auto sibling = temporal.choice;
      sibling.assembly = std::move(intent);
      // The complete sibling retains the parent's selected reuse choice.
      // Preserve its ranking hint too; resetting it to zero systematically
      // postpones a composed implementation behind every reuse-only sibling.
      // This hint affects visitation only. The new branch owns fresh proposals
      // and must materialize and pass the same actual capacity/cost gates.
      discover(std::move(sibling), temporal.choices, current().priority);
    }
  }

  bool discover(ImplementationChoice choice,
                const std::vector<TemporalChoice> &point,
                long double priority = 0) {
    if (support::getActiveCompileTimingSession())
      diagnostics
          << "wafer-compile: discovered-implementation structural="
          << explorationStratum << " from_implementation=" << *active
          << " attempt="
          << (statistics ? statistics->temporalCandidateActualizations : 0)
          << " pipeline=" << choice.pipeline << " invariant="
          << (choice.placement == LayoutMaterializationPlacement::LoopInvariant)
          << " local_assembly_groups=" << choice.assembly.selections.size()
          << '\n';
    for (const auto &branch : implementations)
      if (branch->choice == choice) {
        if (branch->state == BranchState::Waiting) {
          auto existing = llvm::find(branch->discovery, point);
          if (!branch->discoveryDuration && priority >= branch->priority) {
            if (existing != branch->discovery.end())
              branch->discovery.erase(existing);
            branch->discovery.insert(branch->discovery.begin(), point);
            branch->priority = priority;
          } else if (existing == branch->discovery.end()) {
            branch->discovery.push_back(point);
          }
        } else if (branch->proposals) {
          branch->proposals->startAt(point);
        }
        return false;
      }
    auto branch = std::make_unique<ImplementationBranch>();
    branch->choice = std::move(choice);
    branch->discovery.push_back(point);
    branch->priority = priority;
    implementations.push_back(std::move(branch));
    const auto id = implementations.size() - 1;
    const auto &selected = implementations.back()->choice;
    // Placement/pipeline siblings can inherit reuse too. Count each actual
    // new branch here, independently of which discovery path produced it.
    if (selected.reuse && statistics)
      ++statistics->accessReuseBranchesDiscovered;
    if (!selected.assembly.selections.empty() && statistics)
      ++statistics->assemblyBranchesDiscovered;
    auto count = [&](llvm::StringRef field, uint64_t value) {
      support::addCompileCounter("search-implementation",
                                 llvm::formatv("structural-{0}-choice-{1}-{2}",
                                               explorationStratum, id, field)
                                     .str(),
                                 value);
    };
    count("structural-stratum", explorationStratum);
    count("merged", selected.merged);
    count("cpu-scalars", selected.cpuScalars);
    count("local-assembly-groups", selected.assembly.selections.size());
    count("invariant",
          selected.placement == LayoutMaterializationPlacement::LoopInvariant);
    count("pipeline", selected.pipeline);
    count("reuse", bool(selected.reuse));
    if (selected.reuse)
      count("resident",
            llvm::any_of(selected.reuse->selections, [](const auto &selection) {
              return selection.kind != AccessReuseKind::Peer;
            }));
    support::addCompileCounter("search", "implementation-choices", 1);
    return true;
  }

  void activate(size_t index) {
    auto &branch = *implementations[index];
    std::vector<const TemporalDomain *> domains;
    for (const auto &axis : axes) {
      domains.push_back(&axis.domain);
      branch.raw.push_back(axis.domain.getFirstChoice());
    }
    branch.proposals = std::make_unique<TemporalProposals>(std::move(domains));
    for (const auto &point : branch.discovery)
      branch.proposals->startAt(point);
    for (const auto &seed : seeds)
      branch.proposals->seed(seed);
    branch.state = BranchState::Active;
    if (branch.choice.reuse && statistics)
      ++statistics->accessReuseBranchesStarted;
    if (!branch.choice.assembly.selections.empty())
      support::addCompileCounter("search", "local-assembly-branches-started",
                                 1);
    if (!branch.choice.assembly.selections.empty() && statistics)
      ++statistics->assemblyBranchesStarted;
    active = index;
  }

  std::optional<TemporalProposalKind> nextWork(ImplementationBranch &branch) {
    if (!branch.bestDuration &&
        branch.proposals->prepareNext(TemporalProposalKind::Repair))
      return TemporalProposalKind::Repair;
    const std::array order = branch.bestDuration
                                 ? std::array{TemporalProposalKind::Improve,
                                              TemporalProposalKind::Repair,
                                              TemporalProposalKind::Explore}
                                 : std::array{TemporalProposalKind::Repair,
                                              TemporalProposalKind::Explore,
                                              TemporalProposalKind::Improve};
    for (unsigned offset = 0; offset < order.size(); ++offset) {
      auto kind = order[(branch.phase + offset) % order.size()];
      if (branch.proposals->prepareNext(kind))
        return kind;
    }
    return std::nullopt;
  }

  bool selectImplementation(bool mayStart) {
    budgetBlocked = false;
    std::optional<size_t> waiting;
    for (size_t i = 0; i < implementations.size(); ++i)
      if (implementations[i]->state == BranchState::Waiting &&
          (!waiting ||
           implementations[i]->priority > implementations[*waiting]->priority))
        waiting = i;
    if (waiting && mayStart && (visit % 3 == 1 || !active)) {
      if (getRetainedBranchCount() >= retentionLimit &&
          llvm::any_of(implementations, [](const auto &branch) {
            return bool(branch->proposals);
          })) {
        // A lazy implementation descriptor does not own another IR tree. Make
        // room only by retiring evaluated, unprotected heuristic work.
        std::optional<size_t> retire;
        for (size_t i = 0; i < implementations.size(); ++i) {
          const auto &branch = *implementations[i];
          if (!branch.proposals ||
              branch.proposals->hasCapacityRoundInProgress())
            continue;
          if (!retire || !branch.bestDuration ||
              (implementations[*retire]->bestDuration &&
               *branch.bestDuration > *implementations[*retire]->bestDuration))
            retire = i;
        }
        if (retire && options.mode == SearchMode::Standard) {
          implementations[*retire]->proposals.reset();
          implementations[*retire]->state = BranchState::Retired;
          support::addCompileCounter("search", "retired-implementations", 1);
          if (active == retire)
            active.reset();
        }
      }
      if (getRetainedBranchCount() < retentionLimit ||
          !llvm::any_of(implementations, [](const auto &branch) {
            return bool(branch->proposals);
          })) {
        activate(*waiting);
        current().started = true;
        stepSchemesStarted = 1;
        return true;
      }
    }
    // Repair priority belongs inside each implementation's proposal process.
    // Across implementations, resume the least recently evaluated owner in
    // both modes. Draining one capacity chain here can starve an already
    // started implementation for the entire standard evaluation budget.
    std::optional<size_t> selected;
    for (size_t i = 0; i < implementations.size(); ++i) {
      auto &branch = *implementations[i];
      if (!branch.proposals)
        continue;
      if (!selected ||
          std::tie(branch.lastVisit, i) <
              std::tie(implementations[*selected]->lastVisit, *selected))
        selected = i;
    }
    active = selected;
    if (active)
      return true;
    if (waiting) {
      if (!mayStart) {
        budgetBlocked = true;
        return false;
      }
      activate(*waiting);
      current().started = true;
      stepSchemesStarted = 1;
      return true;
    }
    return false;
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
    structural->tensorChoices =
        TensorChoiceBindings::capture(structural->module->getOperation());
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
    seeds.push_back(std::move(joint));
    if (hasFusion)
      seeds.push_back(std::move(independent));
    return std::nullopt;
  }

  std::optional<ExecutableCompilationResult> startTemporal() {
    support::ScopedCompileTimingSpan timing("search-phase", "current-ir",
                                            "start-temporal");
    std::vector<TemporalChoice> choices;
    std::string detail;
    auto work = nextWork(current());
    if (work) {
      choices = current().proposals->take(*work);
      current().phase = (current().phase + 1) % 3;
      support::addCompileCounter(
          "search",
          *work == TemporalProposalKind::Repair    ? "integer-repair-proposals"
          : *work == TemporalProposalKind::Improve ? "integer-improve-proposals"
                                                : "integer-explore-proposals",
          1);
    } else if (options.mode == SearchMode::Deep) {
      completeImplementation();
      return std::nullopt;
    } else {
      bool bug = false;
      while (true) {
        if (!advanceTemporalAxes(axes, current().raw, detail, bug)) {
          if (bug)
            return fail(ExecutableCompilationStatus::CompilerFailure,
                        "search-temporal-next", detail);
          completeImplementation();
          return std::nullopt;
        }
        choices.clear();
        for (const auto &point : current().raw)
          choices.push_back(*point.getChoice());
        if (current().proposals->visitRaw(choices))
          break;
      }
    }
    if (statistics) {
      statistics->accessReuseCandidates += bool(current().choice.reuse);
      statistics->assemblyCandidates +=
          !current().choice.assembly.selections.empty();
      const auto number = statistics->temporalCandidateActualizations++;
      support::addCompileCounter(
          "search-temporal",
          llvm::formatv("choice-{0}-implementation", number).str(), *active);
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
    auto prefix = prefixCache.find(structural->module->getOperation(), choices);
    if (prefix) {
      if (statistics)
        ++statistics->temporalPrefixHits;
    } else {
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
        auto domain =
            remapTemporalDomain(axis.domain, region, mapping, &detail);
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
                                           &temporalFailure,
                                           &candidate->tensorChoices)))
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
      prefix = std::make_shared<TemporalPrefix>();
      prefix->parent = structural->module->getOperation();
      prefix->choices = choices;
      prefix->tiled = std::move(*candidate);
      prefix->canMerge =
          *availability == CommunicationRegionClosureAvailability::Available;
      prefixCache.retain(prefix);
    }
    const bool canMerge = prefix->canMerge;
    if (canMerge && !current().choice.merged) {
      auto sibling = current().choice;
      sibling.merged = true;
      discover(std::move(sibling), choices);
    }
    if (current().choice.merged && !canMerge)
      return fail(ExecutableCompilationStatus::UnsupportedFailure,
                  "search-communication-region-availability",
                  "selected closure is unavailable at this tile point");
    if (!hasImplementationScopes(*prefix->tiled.module, current().choice)) {
      support::addCompileCounter("search", "missing-implementation-scopes", 1);
      return fail(ExecutableCompilationStatus::UnsupportedFailure,
                  "search-implementation-scope",
                  "selected iteration scope is absent at this tile point");
    }
    TemporalAttempt attempt;
    attempt.choices = std::move(choices);
    attempt.prefix = std::move(prefix);
    attempt.choice = current().choice;
    attempt.region.merged = current().choice.merged;
    attempt.region.cpuScalars = current().choice.cpuScalars;
    attempt.region.assembly = current().choice.assembly;
    attempt.region.placement = current().choice.placement;
    if (statistics) {
      statistics->mergedRegionCandidates += attempt.region.merged;
      statistics->regionPreservingCandidates += !attempt.region.merged;
    }
    auto &layout = attempt.prefix->layouts[attempt.region.layoutIndex()];
    if (layout.assembly == attempt.region.assembly) {
      attempt.region.layoutInput = layout.input;
      attempt.region.assemblyOpportunities = layout.opportunities;
    }
    if (attempt.region.layoutInput && statistics)
      ++statistics->layoutPrefixHits;
    pending.emplace(std::move(attempt));
    return std::nullopt;
  }

  RegionPreparation prepareRegion(TemporalAttempt &temporal,
                                  RegionAttempt &attempt) {
    support::ScopedCompileTimingSpan timing("search-phase", "current-ir",
                                            "prepare-region");
    std::string detail;
    if (!attempt.layoutInput) {
      mlir::IRMapping mapping;
      auto candidate = cloneCandidate(temporal.prefix->tiled, mapping, detail);
      if (mlir::failed(candidate))
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-region-clone", detail);
      if (attempt.merged) {
        // Closure replaces these actual Region bodies. No source identity is
        // reconstructed from the merged IR for capacity feedback.
        candidate->temporalBodies.clear();
        SpatialRegionMaterializationFailure failure;
        CommunicationRegionClosureStatistics closure;
        mlir::IRMapping closureMapping;
        if (mlir::failed(closeCrossTileCommunicationRegions(
                *candidate->module, candidate->relations, &closure, &failure,
                &closureMapping)))
          return fail(ExecutableCompilationStatus::CompilerFailure,
                      "search-communication-region-closure", failure.detail);
        candidate->tensorChoices.remap(closureMapping);
        if (statistics)
          statistics->communicationRegionClosures +=
              closure.closedExchangeComponents;
      }
      OnlineAttentionDecompositionFailure attentionFailure;
      if (mlir::failed(decomposeOnlineAttention(
              *candidate->module, candidate->relations, &attentionFailure,
              &candidate->tensorChoices)))
        return fail(attentionFailure.kind ==
                            OnlineAttentionDecompositionFailureKind::
                                UnsupportedSemantics
                        ? ExecutableCompilationStatus::UnsupportedFailure
                        : ExecutableCompilationStatus::CompilerFailure,
                    "search-attention-decomposition", attentionFailure.detail);
      llvm::SmallVector<mlir::tensor::ExtractSliceOp> reads;
      candidate->module->walk([&](mlir::tensor::ExtractSliceOp read) {
        if (read->getParentOfType<TileRegionOp>())
          reads.push_back(read);
      });
      llvm::DenseMap<mlir::Operation *, TensorAssemblyOpportunity> readOutcomes;
      for (auto read : reads) {
        auto outcome = queryLocalTensorAssemblyRead(read);
        if (outcome.kind == TensorAssemblyOpportunityKind::BrokenContract)
          return fail(ExecutableCompilationStatus::CompilerFailure,
                      "search-tensor-assembly-query", outcome.detail);
        readOutcomes.try_emplace(read, std::move(outcome));
      }
      auto families = groupTensorAssemblyReads(reads, candidate->tensorChoices);
      support::addCompileCounter("search", "unbound-tensor-read-identities",
                                 families.unavailable.size());
      llvm::SmallVector<TensorAssemblyOpportunity, 4> outcomes;
      for (const auto &family : families.families) {
        TensorAssemblyOpportunity outcome{
            TensorAssemblyOpportunityKind::Available, {}};
        for (auto read : family.reads) {
          const auto &current = readOutcomes.find(read)->second;
          if (current.kind ==
                  TensorAssemblyOpportunityKind::ResourceExhausted ||
              (outcome.kind !=
                   TensorAssemblyOpportunityKind::ResourceExhausted &&
               current.kind != TensorAssemblyOpportunityKind::Available))
            outcome = current;
        }
        if (outcome.kind == TensorAssemblyOpportunityKind::Available)
          attempt.assemblyOpportunities.push_back(family.selection);
        outcomes.push_back(std::move(outcome));
      }
      discoverLocalAssembly(temporal);
      llvm::SmallVector<
          std::pair<TileRegionOp,
                    llvm::SmallVector<mlir::tensor::ExtractSliceOp>>,
          4>
          selectedRegions;
      for (const auto &selection : attempt.assembly.selections) {
        auto found = llvm::find_if(families.families, [&](const auto &family) {
          return family.selection == selection;
        });
        if (found == families.families.end()) {
          if (statistics)
            ++statistics->assemblyBindingRejected;
          return fail(
              ExecutableCompilationStatus::UnsupportedFailure,
              "search-tensor-assembly-binding",
              "selected assembly read family is absent at this tile point");
        }
        const auto &outcome = outcomes[found - families.families.begin()];
        if (outcome.kind != TensorAssemblyOpportunityKind::Available)
          return fail(outcome.kind ==
                              TensorAssemblyOpportunityKind::ResourceExhausted
                          ? ExecutableCompilationStatus::IndeterminateFailure
                          : ExecutableCompilationStatus::UnsupportedFailure,
                      "search-tensor-assembly-query", outcome.detail);
        for (auto read : found->reads) {
          auto region = read->getParentOfType<TileRegionOp>();
          auto group = llvm::find_if(selectedRegions, [&](const auto &entry) {
            return entry.first == region;
          });
          if (group == selectedRegions.end())
            selectedRegions.push_back({region, {read}});
          else
            group->second.push_back(read);
        }
      }
      // The selection is now an explicit list of current reads. No parent
      // correspondence survives mutation into layout/bufferization or memory.
      candidate->tensorChoices.clear();
      for (const auto &[region, selectedReads] : selectedRegions) {
        TemporalTilingFailure assemblyFailure;
        auto localized = materializeLocalTensorAssemblyReads(
            region, selectedReads, candidate->relations, &assemblyFailure);
        if (mlir::failed(localized))
          return fail(
              assemblyFailure.kind ==
                      TemporalTilingFailureKind::ResourceExhausted
                  ? ExecutableCompilationStatus::IndeterminateFailure
              : assemblyFailure.kind == TemporalTilingFailureKind::Unsupported
                  ? ExecutableCompilationStatus::UnsupportedFailure
                  : ExecutableCompilationStatus::CompilerFailure,
              "search-tensor-assembly-materialization", assemblyFailure.detail);
        support::addCompileCounter("search", "local-assembly-reads",
                                   localized->tileLocalAssemblies);
      }
      if (statistics && !attempt.assembly.selections.empty() &&
          attempt.assembly.selections.size() <
              attempt.assemblyOpportunities.size())
        ++statistics->assemblyMixedMaterializations;
      ExternalBufferLayout external;
      external.layout = options.externalLayout;
      for (const auto &input : program.distributedInputs)
        external.inputArguments.push_back(input.index);
      auto prepared = prepareCurrentLayoutInput(*candidate->module,
                                                candidate->relations, external);
      prepared.statistics.invocations = 0;
      recordLayoutInstrumentation(prepared.statistics);
      if (!prepared.succeeded())
        return fail(classifyLayoutFailure(prepared.status),
                    "search-layout-input", prepared.detail);
      const bool cpuAvailable = hasCPUScalarAlternative(*candidate->module);
      if (attempt.cpuScalars) {
        if (!cpuAvailable)
          return fail(ExecutableCompilationStatus::UnsupportedFailure,
                      "search-scalar-execution",
                      "selected CPU arithmetic has no current register scope");
        auto materialized = materializeCPUScalarAlternative(
            *candidate->module, candidate->relations);
        if (mlir::failed(materialized) || !*materialized)
          return fail(
              ExecutableCompilationStatus::CompilerFailure,
              "search-scalar-execution",
              "CPU scalar materialization failed its current-IR contract");
        support::addCompileCounter("search", "cpu-scalar-materializations",
                                   *materialized);
      } else if (cpuAvailable) {
        auto sibling = temporal.choice;
        sibling.cpuScalars = true;
        discover(std::move(sibling), temporal.choices);
      }
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
      temporal.prefix->layouts[attempt.layoutIndex()] =
          LayoutPrefix{attempt.assembly,
                       attempt.layoutInput,
                       attempt.assemblyOpportunities,
                       {}};
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
    auto &prepared = temporal.prefix->layouts[attempt.layoutIndex()]
                         .prepared[static_cast<unsigned>(placement)];
    if (prepared) {
      if (statistics)
        ++statistics->preparedPrefixHits;
    } else {
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
          candidate->module.get().getOperation(), candidate->relations,
          placement);
      if (mlir::failed(physicalPlacement))
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "search-physical-movement-placement",
                    "physical movement placement produced invalid current IR");
      support::addCompileCounter("movement", "invariant-physical-copies",
                                 *physicalPlacement);
      prepared = std::make_shared<CurrentCandidate>(std::move(*candidate));
    }
    const auto &candidate = prepared;
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
    const auto &selected = temporal.choice;
    if ((selected.recursive && !recursive.isAvailable()) ||
        (selected.allToAll && !distributed.dimensionOrderedAllToAll) ||
        (selected.reduction && !distributed.distributedReduction))
      return fail(
          ExecutableCompilationStatus::UnsupportedFailure,
          "search-movement-domain",
          "selected collective has no current component at this tile point");
    if (placement == LayoutMaterializationPlacement::FirstUse &&
        (input.query->hasLoopInvariantPlacement(input.assignment) ||
         hasInvariantPhysicalMovement(
             candidate->module.get().getOperation()))) {
      auto sibling = temporal.choice;
      sibling.placement = LayoutMaterializationPlacement::LoopInvariant;
      discover(std::move(sibling), temporal.choices);
    }
    if (distributed.sharedDDR && temporal.choice.options.transport !=
                                     BoundaryMovementTransport::SharedDDR) {
      auto sibling = temporal.choice;
      sibling.options.transport = BoundaryMovementTransport::SharedDDR;
      sibling.options.allGather = CompleteAllGatherAlgorithm::Ring;
      sibling.options.allToAll = CompleteAllToAllAlgorithm::Direct;
      sibling.options.reduction = DistributedReductionAlgorithm::Centralized;
      sibling.recursive = sibling.allToAll = sibling.reduction = false;
      discover(std::move(sibling), temporal.choices);
    }
    for (unsigned mask = 1; mask < 8; ++mask) {
      bool gather = mask & 1, allToAll = mask & 2, reduce = mask & 4;
      if ((gather && !recursive.isAvailable()) ||
          (allToAll && !distributed.dimensionOrderedAllToAll) ||
          (reduce && !distributed.distributedReduction))
        continue;
      auto sibling = temporal.choice;
      sibling.options.transport = BoundaryMovementTransport::Peer;
      sibling.recursive = gather;
      sibling.allToAll = allToAll;
      sibling.reduction = reduce;
      sibling.options.allGather =
          gather ? CompleteAllGatherAlgorithm::RecursiveDoubling
                 : CompleteAllGatherAlgorithm::Ring;
      sibling.options.allToAll =
          allToAll ? CompleteAllToAllAlgorithm::DimensionOrdered
                   : CompleteAllToAllAlgorithm::Direct;
      sibling.options.reduction =
          reduce ? DistributedReductionAlgorithm::Ring
                 : DistributedReductionAlgorithm::Centralized;
      discover(std::move(sibling), temporal.choices);
    }
    attempt.lowered = prepared;
    return RegionPrepared{};
  }

  void recordOwnership() {
    if (!statistics)
      return;
    llvm::SmallPtrSet<mlir::Operation *, 16> owners;
    if (structural) {
      owners.insert(structural->module->getOperation());
      prefixCache.collectModules(structural->module->getOperation(), owners);
    }
    if (pending)
      pending->prefix->collectModules(owners);
    statistics->peakSessionTemporalPrefixes = std::max<uint64_t>(
        statistics->peakSessionTemporalPrefixes, bool(pending));
    statistics->peakSessionIRModules =
        std::max<uint64_t>(statistics->peakSessionIRModules, owners.size());
  }

  StructuralCandidateEvaluation closed() {
    StructuralCandidateEvaluation result;
    result.completeDomain = false;
    return result;
  }
  StructuralCandidateEvaluation blocked() {
    auto result = closed();
    result.budgetBlocked = true;
    return result;
  }
  StructuralCandidateEvaluation yield() {
    recordOwnership();
    StructuralCandidateEvaluation result{
        {},
        0,
        CandidateContinuation::Explore,
        CandidateRetention::UnfinishedActualization};
    result.schemesStarted = stepSchemesStarted;
    result.schemesCompleted = stepSchemesCompleted;
    result.completeDomain = false;
    return result;
  }

  void completeImplementation() {
    current().proposals.reset();
    current().state = BranchState::Complete;
    ++stepSchemesCompleted;
    active.reset();
  }

  StructuralCandidateEvaluation
  finish(ExecutableCompilationResult compiled,
         std::optional<analysis::SearchObjective> objective = std::nullopt) {
    recordOwnership();
    const auto completedChoice = pending ? pending->choice : current().choice;
    pending.reset();
    const auto status = classifyActualStatus(compiled.status);
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
                  << " local_assembly_groups="
                  << completedChoice.assembly.selections.size()
                  << " detail=" << compiled.detail << '\n';
    ActualCandidateResult result;
    result.status = status;
    result.detail = compiled.detail;
    if (objective)
      result.objective = ActualCandidateResult::EvaluatedObjective{
          costCohort, std::move(*objective)};
    if (compiled.isAccepted() || compiled.isProvenExactRejection())
      result.compilation.emplace(std::move(compiled));
    CandidateContinuation next = CandidateContinuation::Explore;
    if (exhausted && active && status != ActualCandidateStatus::CompilerBug)
      completeImplementation();
    if (active && current().proposals) {
      current().lastVisit = ++visit;
      auto work = nextWork(current());
      if (work)
        next = *work == TemporalProposalKind::Repair
                   ? CandidateContinuation::Repair
               : *work == TemporalProposalKind::Improve
                   ? CandidateContinuation::Improve
                   : CandidateContinuation::Explore;
      else if (options.mode == SearchMode::Deep) {
        completeImplementation();
      }
    }
    CandidateRetention retention = CandidateRetention::Replaceable;
    if (llvm::any_of(implementations, [](const auto &branch) {
          return branch->proposals &&
                 branch->proposals->hasCapacityRoundInProgress();
        }))
      retention = CandidateRetention::PendingCapacityRepair;
    else if (llvm::any_of(implementations, [](const auto &branch) {
               return branch->state == BranchState::Waiting;
             }))
      retention = CandidateRetention::PendingImplementation;
    StructuralCandidateEvaluation evaluation{
        std::move(result), 1,
        exhausted ? CandidateContinuation::Exhausted : next, retention};
    evaluation.schemesStarted = stepSchemesStarted;
    evaluation.schemesCompleted = stepSchemesCompleted;
    evaluation.completeDomain = false;
    return evaluation;
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
  std::vector<std::vector<TemporalChoice>> seeds;
  std::vector<std::unique_ptr<ImplementationBranch>> implementations;
  std::optional<size_t> active;
  std::optional<TemporalAttempt> pending;
  const size_t explorationStratum;
  CurrentIRPrefixCache &prefixCache;
  uint64_t visit = 0;
  uint64_t retentionLimit = 1;
  uint64_t stepSchemesStarted = 0;
  uint64_t stepSchemesCompleted = 0;
  bool exhausted = false;
  bool budgetBlocked = false;
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
        costCohort(costCohort),
        prefixCache(std::min(options.prefixCacheEntries, options.limits.width),
                    statistics) {}

  std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &state) override {
    return std::make_unique<CurrentIRCandidateSession>(
        state, tensorProgram, analysis, planning, program, executionConfig,
        diagnostics, programData, options, statistics, executableStatistics,
        costCohort, nextExplorationStratum++, prefixCache);
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
  CurrentIRPrefixCache prefixCache;
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
  traversal.trialCredits = options.limits.trials;
  traversal.mode = options.mode;
  traversal.maximumRegionRefinementCandidates =
      maximumRegionRefinementCandidates;
  traversal.termination = options.termination;
  traversal.candidateObserver = options.candidateObserver;
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
  searchCounter("trials-used", searched.work.trialsUsed);
  searchCounter("schemes-started", searched.work.schemesStarted);
  searchCounter("schemes-completed", searched.work.schemesCompleted);
  searchCounter("schemes-unfinished",
                searched.work.schemesStarted - searched.work.schemesCompleted);
  searchCounter("deep-mode", options.mode == SearchMode::Deep);
  searchCounter("incumbent-updates",
                searched.control.statistics.incumbentUpdates);
  if (searched.work.firstFeasible) {
    searchCounter("first-feasible-candidate",
                  searched.work.firstFeasible->candidateIndex);
    searchCounter("first-feasible-microseconds",
                  searched.work.firstFeasible->elapsedMicroseconds);
  }
  if (searched.work.incumbentSelected) {
    searchCounter("winner-first-candidate",
                  searched.work.incumbentSelected->candidateIndex);
    searchCounter("winner-first-microseconds",
                  searched.work.incumbentSelected->elapsedMicroseconds);
  }
  if (statistics) {
    searchCounter("peak-session-temporal-prefixes",
                  statistics->peakSessionTemporalPrefixes);
    searchCounter("peak-session-ir-modules", statistics->peakSessionIRModules);
    searchCounter("temporal-backpressure-turns",
                  statistics->temporalBackpressureTurns);
  }
  searchCounter("trials-remaining",
                options.limits.trials - searched.work.trialsUsed);
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
    searchCounter("temporal-prefix-hits", statistics->temporalPrefixHits);
    searchCounter("layout-prefix-hits", statistics->layoutPrefixHits);
    searchCounter("prepared-prefix-hits", statistics->preparedPrefixHits);
    searchCounter("prefix-evictions", statistics->prefixEvictions);
    searchCounter("peak-cached-prefixes", statistics->peakCachedPrefixes);
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
    searchCounter("access-reuse-branches-started",
                  statistics->accessReuseBranchesStarted);
    searchCounter("access-reuse-queries", statistics->accessReuseQueries);
    searchCounter("access-reuse-low-benefit",
                  statistics->accessReuseLowBenefit);
    searchCounter("access-reuse-unknown-benefit",
                  statistics->accessReuseUnknownBenefit);
    searchCounter("access-reuse-eligible", statistics->accessReuseEligible);
    searchCounter("access-reuse-branches-discovered",
                  statistics->accessReuseBranchesDiscovered);
    searchCounter("access-reuse-accepted", statistics->accessReuseAccepted);
    searchCounter("resident-reuse-accepted", statistics->residentReuseAccepted);
    if (statistics->minimumResidentDDRReadBytes)
      searchCounter("resident-minimum-ddr-read-bytes",
                    *statistics->minimumResidentDDRReadBytes);
    searchCounter("access-reuse-capacity-rejected",
                  statistics->accessReuseCapacityRejected);
    searchCounter("assembly-shared-capacity-rejected",
                  statistics->sharedAssemblyCapacityRejected);
    searchCounter("assembly-local-capacity-rejected",
                  statistics->localAssemblyCapacityRejected);
    searchCounter("assembly-shared-capacity-refinements",
                  statistics->sharedAssemblyCapacityRefinements);
    searchCounter("assembly-local-capacity-refinements",
                  statistics->localAssemblyCapacityRefinements);
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
  if (options.candidateObserver) {
    if (!searched.inspectedCandidate)
      return fail(searched.control.coverage == SearchControllerCoverage::Failed
                      ? ExecutableCompilationStatus::CompilerFailure
                      : ExecutableCompilationStatus::IndeterminateFailure,
                  "search-inspection", "requested candidate was not reached");
    auto &inspected = *searched.inspectedCandidate;
    if (inspected.compilation)
      return std::move(*inspected.compilation);
    return fail(inspected.status == ActualCandidateStatus::CompilerBug
                    ? ExecutableCompilationStatus::CompilerFailure
                : inspected.status == ActualCandidateStatus::Unsupported
                    ? ExecutableCompilationStatus::UnsupportedFailure
                    : ExecutableCompilationStatus::IndeterminateFailure,
                "search-inspection", inspected.detail);
  }
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
