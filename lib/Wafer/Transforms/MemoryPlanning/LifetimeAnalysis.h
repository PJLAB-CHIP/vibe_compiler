//===- LifetimeAnalysis.h - Structured memory lifetime analysis -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_MEMORYPLANNING_LIFETIMEANALYSIS_H
#define WAFER_TRANSFORMS_MEMORYPLANNING_LIFETIMEANALYSIS_H

#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cassert>
#include <functional>
#include <optional>
#include <utility>

namespace wafer::memory_planning::detail {

/// A conjunction of structured control-flow decisions. Incompatible
/// conditions describe mutually exclusive paths and therefore cannot make two
/// live segments overlap.
class PathCondition {
public:
  static PathCondition root() { return PathCondition{}; }

  bool compatibleWith(const PathCondition &other) const;
  /// Packing compatibility is deliberately weaker than execution-path
  /// compatibility: branch decisions nested in a loop may be selected again
  /// on every dynamic iteration and therefore cannot prove that two static
  /// buffers are globally exclusive.
  bool compatibleForPacking(const PathCondition &other) const;
  PathCondition withoutRepeatableDecisions() const;
  std::optional<PathCondition> intersect(const PathCondition &other) const;
  bool implies(const PathCondition &other) const;
  std::optional<PathCondition> withDecision(uint64_t decision, bool selected,
                                            bool repeatable = false) const;
  void subtract(const PathCondition &covered,
                llvm::SmallVectorImpl<PathCondition> &remaining) const;

  bool operator==(const PathCondition &other) const {
    return trueDecisions == other.trueDecisions &&
           falseDecisions == other.falseDecisions &&
           repeatableDecisions == other.repeatableDecisions;
  }
  bool operator!=(const PathCondition &other) const {
    return !(*this == other);
  }

private:
  using DecisionSet = llvm::SmallVector<uint64_t, 4>;

  PathCondition() = default;
  PathCondition(DecisionSet trueDecisions, DecisionSet falseDecisions,
                DecisionSet repeatableDecisions)
      : trueDecisions(std::move(trueDecisions)),
        falseDecisions(std::move(falseDecisions)),
        repeatableDecisions(std::move(repeatableDecisions)) {}

  DecisionSet trueDecisions;
  DecisionSet falseDecisions;
  DecisionSet repeatableDecisions;
};

struct ProgramPoint {
  int64_t event = 0;
  PathCondition path = PathCondition::root();
};

enum class TimelineFailureKind {
  DecisionDomainExhausted,
  InconsistentPathCondition,
  UnsupportedRegionControlFlow,
};

struct TimelineFailure {
  TimelineFailureKind kind = TimelineFailureKind::UnsupportedRegionControlFlow;
  mlir::Operation *origin = nullptr;
};

/// A deterministic preorder timeline for structured IR. `scf.if` successors
/// are mutually exclusive for one execution of their parent region and every
/// `scf.for` body is modeled as an optional path because the loop may execute
/// zero times. Decision conditions use a sparse identifier set rather than a
/// machine-word mask. Decisions inside a loop are marked repeatable so packing
/// does not mistake per-iteration exclusivity for whole-execution exclusivity.
class StructuredTimeline {
public:
  static mlir::FailureOr<StructuredTimeline>
  build(mlir::Operation *scope, TimelineFailure *failure = nullptr);

  std::optional<ProgramPoint> lookup(mlir::Operation *op) const;
  std::optional<int64_t> lookupSubtreeEnd(mlir::Operation *op) const;

private:
  llvm::DenseMap<mlir::Operation *, ProgramPoint> points;
  llvm::DenseMap<mlir::Operation *, int64_t> subtreeEnds;
};

/// Operation-anchored, recomputable timeline facts for MLIR pass pipelines.
/// The analysis owns no target policy and writes nothing to IR. Passes that do
/// not mutate structured control flow may explicitly preserve it; all other
/// transformations use the AnalysisManager's default invalidation.
class StructuredTimelineAnalysis {
public:
  explicit StructuredTimelineAnalysis(mlir::Operation *scope);

  bool isValid() const { return timeline.has_value(); }
  const StructuredTimeline &getTimeline() const {
    assert(timeline && "requested an invalid structured timeline");
    return *timeline;
  }
  const TimelineFailure &getFailure() const { return failure; }

private:
  std::optional<StructuredTimeline> timeline;
  TimelineFailure failure;
};

struct LiveSegment {
  int64_t beginEvent = 0;
  int64_t endEvent = 0;
  PathCondition path = PathCondition::root();
};

struct LifetimeDemand {
  mlir::memref::AllocOp allocation;
  int64_t sizeBytes = 0;
  int64_t alignmentBytes = 0;
  unsigned stableOrdinal = 0;
  ProgramPoint allocationPoint;
  llvm::SmallVector<LiveSegment, 4> segments;
};

struct RootRef {
  unsigned demandIndex = 0;
  PathCondition path = PathCondition::root();
};

/// Path-qualified semantic origin for a tracked memref value. Unlike RootRef,
/// this also represents caller-owned/external roots that have no packing
/// demand in the current planner.
struct ValueOriginRef {
  mlir::Value root;
  PathCondition path = PathCondition::root();
};

enum class LifetimeFailureKind {
  MissingAllocationEvent,
  UnsupportedTrackedValueProducer,
  UnsupportedTrackedValueEscape,
  LoopCarriedAllocationInstance,
  MissingAsyncCompletion,
  UnsupportedAsyncCompletionFlow,
  MissingLocalCompletion,
  LoopBackedgeCompletion,
  InconsistentCompletionState,
};

struct LifetimeFailure {
  LifetimeFailureKind kind = LifetimeFailureKind::MissingAllocationEvent;
  mlir::Operation *origin = nullptr;
};

using TrackedTypePredicate = std::function<bool(mlir::Type)>;
using ValueResolver = std::function<mlir::Value(mlir::Value)>;
using ExplicitRootPredicate = std::function<bool(mlir::Value)>;

/// Returns true when a direct call is a closed, side-effect-free alias helper.
/// Every tensor/memref result must resolve through supported alias/control-flow
/// operations to one or more statically tracked caller operands. Operations
/// that touch storage-shaped values must themselves have recognized alias
/// semantics; only unrelated pure scalar operations are otherwise admitted.
/// Nested calls, storage ownership and memory access are rejected, while
/// callers still own target-resource checks.
bool isSupportedDirectAliasCall(mlir::func::CallOp call,
                                const TrackedTypePredicate &isTrackedType);

class LocalCompletionTracker;

/// Recomputes allocation-root aliases and live segments from the current IR.
/// The caller supplies the tracked memref predicate and an optional boundary
/// resolver; no result is written to IR or retained across transformations.
class LifetimeDataflow {
public:
  LifetimeDataflow(const StructuredTimeline &timeline,
                   llvm::MutableArrayRef<LifetimeDemand> demands,
                   TrackedTypePredicate isTrackedType,
                   ValueResolver resolveValue = {},
                   ExplicitRootPredicate isExplicitRoot = {});

  mlir::LogicalResult run(mlir::Operation *scope,
                          LocalCompletionTracker *localCompletion = nullptr,
                          LifetimeFailure *failure = nullptr);

  llvm::SmallVector<RootRef, 2> rootsAt(mlir::Value value,
                                        PathCondition usePath) const;
  llvm::SmallVector<ValueOriginRef, 2> originsAt(mlir::Value value,
                                                 PathCondition usePath) const;
  llvm::SmallVector<RootRef, 2> asyncRootsAt(mlir::Value handle,
                                             PathCondition usePath) const;
  void extendTo(RootRef ref, ProgramPoint completionPoint);

private:
  friend class LocalCompletionTracker;

  struct AsyncTaskRef {
    unsigned taskIndex = 0;
    PathCondition path = PathCondition::root();
  };
  struct AsyncTaskState {
    mlir::Operation *origin = nullptr;
    llvm::SmallVector<PathCondition, 2> pendingPaths;
  };

  mlir::LogicalResult initialize(LifetimeFailure *failure);
  mlir::LogicalResult processRegion(mlir::Region &region,
                                    LocalCompletionTracker *localCompletion,
                                    LifetimeFailure *failure);
  mlir::LogicalResult processBlock(mlir::Block &block,
                                   LocalCompletionTracker *localCompletion,
                                   LifetimeFailure *failure);
  void recordOperands(mlir::Operation *op);
  void mapViewLikeResults(mlir::Operation *op);
  mlir::LogicalResult mapSelectLikeResult(mlir::Operation *op,
                                          LifetimeFailure *failure);
  mlir::LogicalResult mapDirectCallResults(mlir::Operation *op,
                                           LifetimeFailure *failure);
  mlir::LogicalResult mapAsyncDependencyResults(mlir::Operation *op,
                                                LifetimeFailure *failure);
  void mapIfResults(mlir::Operation *op);
  void mapForRegionIterArgs(mlir::Operation *op);
  mlir::LogicalResult mapForResultsAndBackedge(mlir::Operation *op,
                                               LifetimeFailure *failure);
  void mapSingleExecutionRegionBlockArgs(mlir::Operation *op);
  void mapSingleExecutionRegionResults(mlir::Operation *op);
  mlir::Value normalize(mlir::Value value) const;
  void recordUse(RootRef ref, int64_t event);
  llvm::SmallVector<AsyncTaskRef, 2> asyncTasksAt(mlir::Value handle,
                                                  PathCondition usePath) const;
  static void
  appendUniqueAsyncTaskRefs(llvm::SmallVectorImpl<AsyncTaskRef> &destination,
                            llvm::ArrayRef<AsyncTaskRef> source);
  static bool sameAsyncTaskRefs(llvm::ArrayRef<AsyncTaskRef> lhs,
                                llvm::ArrayRef<AsyncTaskRef> rhs);
  void completeAsyncTasks(mlir::Operation *op);
  mlir::LogicalResult finishAsyncTasks(LifetimeFailure *failure) const;

  const StructuredTimeline &timeline;
  llvm::MutableArrayRef<LifetimeDemand> demands;
  TrackedTypePredicate isTrackedType;
  ValueResolver resolveValue;
  ExplicitRootPredicate isExplicitRoot;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> valueRefs;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<ValueOriginRef, 2>>
      valueOrigins;
  // A loop backedge is discovered after its body has been visited. Cache
  // entries produced in that body must be recomputed against the completed
  // recurrence union, while entries created afterwards can stop recursive
  // alias walks immediately.
  uint64_t provenanceRevision = 1;
  llvm::DenseMap<mlir::Value, uint64_t> valueRefRevisions;
  llvm::DenseMap<mlir::Value, uint64_t> valueOriginRevisions;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> asyncRefs;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<AsyncTaskRef, 2>> asyncTaskRefs;
  llvm::SmallVector<AsyncTaskState, 4> asyncTasks;
};

/// Extends tracked local-engine accesses through path-covering completion
/// barriers. An asynchronous issue is tracked only when its centralized local
/// completion contract is OrderedAsynchronousIssue and it has a value-associated
/// storage effect on a tracked root. While such an access is pending,
/// deallocation and operations without a complete effect contract fail closed.
class LocalCompletionTracker {
public:
  LocalCompletionTracker() = default;

  mlir::LogicalResult observe(mlir::Operation *op, LifetimeDataflow &dataflow,
                              LifetimeFailure *failure = nullptr);
  mlir::LogicalResult
  verifyLoopBackedge(mlir::Operation *loop, LifetimeDataflow &dataflow,
                     LifetimeFailure *failure = nullptr) const;
  mlir::LogicalResult finish(mlir::Operation *scope,
                             LifetimeFailure *failure = nullptr) const;

private:
  struct PendingIssue {
    mlir::Operation *origin = nullptr;
    PathCondition path = PathCondition::root();
    uint32_t workerMask = 0;
    bool accessOrderResolved = false;
    bool hasWrite = false;
  };
  struct PendingAccess {
    mlir::Operation *origin = nullptr;
    RootRef root;
    uint32_t workerMask = 0;
    mlir::Value logicalRoot;
    mlir::Value accessIdentity;
    bool write = false;
  };
  struct AccessCollection {
    llvm::SmallVector<PendingAccess, 4> accesses;
    bool hasTrackedEffect = false;
    bool hasUnknownObserverEffect = false;
    bool allResolved = true;
    bool hasWrite = false;
  };

  AccessCollection collectAccesses(mlir::Operation *op, ProgramPoint point,
                                   uint32_t workerMask,
                                   LifetimeDataflow &dataflow) const;
  mlir::LogicalResult
  verifyPendingObservers(mlir::Operation *op, ProgramPoint point,
                         const NCCSynchronizationContract &contract,
                         const AccessCollection &current,
                         LifetimeFailure *failure) const;
  bool provesLoopBackedgeOrder(const PendingIssue &issue, mlir::Operation *loop,
                               LifetimeDataflow &dataflow) const;
  void appendPendingAccess(PendingAccess access);
  void refreshPendingAccessSummary();
  void processFence(ProgramPoint fencePoint, uint32_t participantMask,
                    LifetimeDataflow &dataflow);

  llvm::SmallVector<PendingIssue, 8> pendingIssues;
  llvm::SmallVector<PendingAccess, 8> pendingAccesses;
  uint32_t commonPendingWorkerMask = 0;
  bool pendingWorkerMasksAgree = true;
  bool pendingAllHaveResolvedRoots = true;
  bool pendingAllHaveLogicalRoots = true;
};

/// Combines two independent byte-alignment divisibility requirements.
/// Returns nullopt for non-positive inputs or int64 overflow.
std::optional<int64_t> combineAlignmentRequirements(int64_t lhs, int64_t rhs);

bool lifetimesOverlap(const LifetimeDemand &lhs, const LifetimeDemand &rhs);

} // namespace wafer::memory_planning::detail

#endif // WAFER_TRANSFORMS_MEMORYPLANNING_LIFETIMEANALYSIS_H
