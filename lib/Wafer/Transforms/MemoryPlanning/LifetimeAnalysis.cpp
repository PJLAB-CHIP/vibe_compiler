//===- LifetimeAnalysis.cpp - Structured memory lifetime analysis --------===//

#include "MemoryPlanning/LifetimeAnalysis.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

namespace wafer::memory_planning::detail {

namespace {

static void setTimelineFailure(TimelineFailure *failure,
                               TimelineFailureKind kind,
                               mlir::Operation *origin) {
  if (failure)
    *failure = TimelineFailure{kind, origin};
}

static void setLifetimeFailure(LifetimeFailure *failure,
                               LifetimeFailureKind kind,
                               mlir::Operation *origin) {
  if (failure)
    *failure = LifetimeFailure{kind, origin};
}

static bool hasAsyncDependencyType(mlir::Value value) {
  return mlir::isa<mlir::async::TokenType, mlir::async::ValueType,
                   mlir::async::GroupType>(value.getType());
}

static bool isSupportedAsyncTaskProducer(mlir::Operation *op) {
  if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(op))
    return true;
  auto call = mlir::dyn_cast<mlir::async::CallOp>(op);
  if (!call)
    return false;
  auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::async::FuncOp>(
      call, call.getCalleeAttr());
  return callee && !callee.isExternal();
}

static mlir::scf::ForOp getEnclosingFor(mlir::Operation *op) {
  return op ? op->getParentOfType<mlir::scf::ForOp>() : mlir::scf::ForOp{};
}

static bool isStaticallyNonEmpty(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getLowerBound()));
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getUpperBound()));
  std::optional<int64_t> step =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getStep()));
  return lower && upper && step && *step > 0 && *lower < *upper;
}

static mlir::scf::YieldOp getSingleBlockYield(mlir::Region &region) {
  if (region.empty())
    return {};
  return mlir::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
}

static bool isNestedIn(mlir::Operation *operation, mlir::Operation *ancestor) {
  for (mlir::Operation *current = operation; current;
       current = current->getParentOp())
    if (current == ancestor)
      return true;
  return false;
}

static bool
collectCalleeFormalAliases(mlir::Value value, mlir::func::FuncOp callee,
                           llvm::SmallVectorImpl<unsigned> &formalIndices,
                           llvm::DenseSet<mlir::Value> &active) {
  if (!value || !active.insert(value).second)
    return true;

  auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (blockArg.getOwner() == &callee.getBody().front()) {
      unsigned index = blockArg.getArgNumber();
      if (!llvm::is_contained(formalIndices, index))
        formalIndices.push_back(index);
      return finish(true);
    }
    auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
        blockArg.getOwner() ? blockArg.getOwner()->getParentOp() : nullptr);
    if (!forOp || blockArg.getOwner() != forOp.getBody() ||
        blockArg.getArgNumber() == 0)
      return finish(false);
    unsigned index = blockArg.getArgNumber() - 1;
    if (index >= forOp.getInitArgs().size())
      return finish(false);
    if (!collectCalleeFormalAliases(forOp.getInitArgs()[index], callee,
                                    formalIndices, active))
      return finish(false);
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    return finish(yield && index < yield.getResults().size() &&
                  collectCalleeFormalAliases(yield.getResults()[index], callee,
                                             formalIndices, active));
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return finish(false);
  mlir::Operation *def = result.getOwner();
  auto collect = [&](mlir::Value source) {
    return collectCalleeFormalAliases(source, callee, formalIndices, active);
  };
  if (auto toMemref = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(def))
    return finish(collect(toMemref.getTensor()));
  if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(def))
    return finish(collect(toTensor.getMemref()));
  if (auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(def))
    return finish(collect(viewLike.getViewSource()));
  if (auto select = mlir::dyn_cast<mlir::SelectLikeOpInterface>(def))
    return finish(collect(select.getTrueValue()) &&
                  collect(select.getFalseValue()));
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
    unsigned index = result.getResultNumber();
    for (mlir::scf::YieldOp yield : {getSingleBlockYield(ifOp.getThenRegion()),
                                     getSingleBlockYield(ifOp.getElseRegion())})
      if (!yield || index >= yield.getResults().size() ||
          !collect(yield.getResults()[index]))
        return finish(false);
    return finish(true);
  }
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
    unsigned index = result.getResultNumber();
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    if (index >= forOp.getInitArgs().size() || !yield ||
        index >= yield.getResults().size())
      return finish(false);
    return finish(collect(forOp.getInitArgs()[index]) &&
                  collect(yield.getResults()[index]));
  }
  return finish(false);
}

static bool getDirectCallResultFormalAliases(
    mlir::func::CallOp call, unsigned resultIndex,
    llvm::SmallVectorImpl<unsigned> &formalIndices) {
  mlir::func::FuncOp callee =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
          call, call.getCalleeAttr());
  if (!callee || callee.isExternal() || !callee.getBody().hasOneBlock())
    return false;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      callee.getBody().front().getTerminator());
  if (!returnOp || resultIndex >= returnOp.getNumOperands())
    return false;
  llvm::DenseSet<mlir::Value> active;
  return collectCalleeFormalAliases(returnOp.getOperand(resultIndex), callee,
                                    formalIndices, active) &&
         !formalIndices.empty();
}

static void appendUniqueRootRefs(llvm::SmallVectorImpl<RootRef> &destination,
                                 llvm::ArrayRef<RootRef> source) {
  for (RootRef ref : source) {
    if (llvm::any_of(destination, [&](RootRef existing) {
          return existing.demandIndex == ref.demandIndex &&
                 existing.path == ref.path;
        }))
      continue;
    destination.push_back(ref);
  }
}

static void
appendUniqueOriginRefs(llvm::SmallVectorImpl<ValueOriginRef> &destination,
                       llvm::ArrayRef<ValueOriginRef> source) {
  for (ValueOriginRef ref : source) {
    if (llvm::any_of(destination, [&](ValueOriginRef existing) {
          return existing.root == ref.root && existing.path == ref.path;
        }))
      continue;
    destination.push_back(ref);
  }
}

static void forgetRepeatableDecisions(llvm::SmallVectorImpl<RootRef> &refs) {
  llvm::SmallVector<RootRef, 4> relaxed;
  for (RootRef ref : refs) {
    ref.path = ref.path.withoutRepeatableDecisions();
    appendUniqueRootRefs(relaxed, llvm::ArrayRef<RootRef>{ref});
  }
  refs.assign(relaxed.begin(), relaxed.end());
}

static void
forgetRepeatableDecisions(llvm::SmallVectorImpl<ValueOriginRef> &refs) {
  llvm::SmallVector<ValueOriginRef, 4> relaxed;
  for (ValueOriginRef ref : refs) {
    ref.path = ref.path.withoutRepeatableDecisions();
    appendUniqueOriginRefs(relaxed, llvm::ArrayRef<ValueOriginRef>{ref});
  }
  refs.assign(relaxed.begin(), relaxed.end());
}

} // namespace

bool isSupportedDirectAliasCall(mlir::func::CallOp call,
                                const TrackedTypePredicate &isTrackedType) {
  if (!call || !isTrackedType)
    return false;
  mlir::func::FuncOp callee =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
          call, call.getCalleeAttr());
  if (!callee || callee.isExternal() || !callee.isPrivate() ||
      !callee.getBody().hasOneBlock() ||
      call.getNumOperands() != callee.getNumArguments() ||
      call.getNumResults() != callee.getNumResults())
    return false;

  bool supportedBody = true;
  callee.getBody().walk([&](mlir::Operation *op) {
    if (!supportedBody)
      return;
    if (mlir::isa<mlir::CallOpInterface>(op)) {
      supportedBody = false;
      return;
    }
    if (mlir::isa<mlir::memref::ExtractAlignedPointerAsIndexOp,
                  mlir::memref::ExtractStridedMetadataOp>(op)) {
      supportedBody = false;
      return;
    }
    bool carriesTrackedValue =
        llvm::any_of(op->getOperandTypes(), isTrackedType) ||
        llvm::any_of(op->getResultTypes(), isTrackedType);
    auto isStorageShapedType = [](mlir::Type type) {
      return mlir::isa<mlir::TensorType, mlir::BaseMemRefType>(type);
    };
    bool touchesStorageShapedValue =
        llvm::any_of(op->getOperandTypes(), isStorageShapedType) ||
        llvm::any_of(op->getResultTypes(), isStorageShapedType);
    bool supportedAliasSemantics =
        mlir::isa<mlir::func::ReturnOp, mlir::scf::YieldOp, mlir::scf::IfOp,
                  mlir::scf::ForOp, mlir::bufferization::ToMemrefOp,
                  mlir::bufferization::ToTensorOp>(op) ||
        mlir::isa<mlir::ViewLikeOpInterface, mlir::SelectLikeOpInterface>(op);
    if ((!supportedAliasSemantics && !mlir::isPure(op)) ||
        ((carriesTrackedValue || touchesStorageShapedValue) &&
         !supportedAliasSemantics))
      supportedBody = false;
  });
  if (!supportedBody)
    return false;

  for (auto [index, result] : llvm::enumerate(call.getResults())) {
    llvm::SmallVector<unsigned, 2> formalIndices;
    bool hasFormalAliases =
        getDirectCallResultFormalAliases(call, index, formalIndices);
    bool hasStorageResult =
        mlir::isa<mlir::TensorType, mlir::BaseMemRefType>(result.getType());
    if (((isTrackedType(result.getType()) || hasStorageResult) &&
         !hasFormalAliases) ||
        (hasFormalAliases &&
         llvm::any_of(formalIndices, [&](unsigned formalIndex) {
           return formalIndex >= call.getNumOperands() ||
                  !isTrackedType(call.getOperand(formalIndex).getType());
         })))
      return false;
  }
  return true;
}

static bool containsDecision(llvm::ArrayRef<uint64_t> decisions,
                             uint64_t decision) {
  return std::binary_search(decisions.begin(), decisions.end(), decision);
}

static void insertDecision(llvm::SmallVectorImpl<uint64_t> &decisions,
                           uint64_t decision) {
  auto position =
      std::lower_bound(decisions.begin(), decisions.end(), decision);
  if (position == decisions.end() || *position != decision)
    decisions.insert(position, decision);
}

static llvm::SmallVector<uint64_t, 4>
unionDecisions(llvm::ArrayRef<uint64_t> lhs, llvm::ArrayRef<uint64_t> rhs) {
  llvm::SmallVector<uint64_t, 4> result;
  result.reserve(lhs.size() + rhs.size());
  size_t lhsIndex = 0;
  size_t rhsIndex = 0;
  while (lhsIndex < lhs.size() || rhsIndex < rhs.size()) {
    if (rhsIndex == rhs.size() ||
        (lhsIndex < lhs.size() && lhs[lhsIndex] < rhs[rhsIndex])) {
      result.push_back(lhs[lhsIndex++]);
      continue;
    }
    if (lhsIndex == lhs.size() || rhs[rhsIndex] < lhs[lhsIndex]) {
      result.push_back(rhs[rhsIndex++]);
      continue;
    }
    result.push_back(lhs[lhsIndex]);
    ++lhsIndex;
    ++rhsIndex;
  }
  return result;
}

static llvm::SmallVector<uint64_t, 4>
subtractDecisions(llvm::ArrayRef<uint64_t> decisions,
                  llvm::ArrayRef<uint64_t> removed) {
  llvm::SmallVector<uint64_t, 4> result;
  result.reserve(decisions.size());
  for (uint64_t decision : decisions)
    if (!containsDecision(removed, decision))
      result.push_back(decision);
  return result;
}

static bool decisionsIntersect(llvm::ArrayRef<uint64_t> lhs,
                               llvm::ArrayRef<uint64_t> rhs) {
  size_t lhsIndex = 0;
  size_t rhsIndex = 0;
  while (lhsIndex < lhs.size() && rhsIndex < rhs.size()) {
    if (lhs[lhsIndex] == rhs[rhsIndex])
      return true;
    if (lhs[lhsIndex] < rhs[rhsIndex])
      ++lhsIndex;
    else
      ++rhsIndex;
  }
  return false;
}

static bool
hasNonRepeatableConflict(llvm::ArrayRef<uint64_t> selected,
                         llvm::ArrayRef<uint64_t> otherRejected,
                         llvm::ArrayRef<uint64_t> selectedRepeatable,
                         llvm::ArrayRef<uint64_t> otherRepeatable) {
  size_t selectedIndex = 0;
  size_t rejectedIndex = 0;
  while (selectedIndex < selected.size() &&
         rejectedIndex < otherRejected.size()) {
    uint64_t selectedDecision = selected[selectedIndex];
    uint64_t rejectedDecision = otherRejected[rejectedIndex];
    if (selectedDecision < rejectedDecision) {
      ++selectedIndex;
      continue;
    }
    if (rejectedDecision < selectedDecision) {
      ++rejectedIndex;
      continue;
    }
    if (!containsDecision(selectedRepeatable, selectedDecision) &&
        !containsDecision(otherRepeatable, selectedDecision))
      return true;
    ++selectedIndex;
    ++rejectedIndex;
  }
  return false;
}

static bool isDecisionSubset(llvm::ArrayRef<uint64_t> subset,
                             llvm::ArrayRef<uint64_t> superset) {
  return llvm::all_of(subset, [&](uint64_t decision) {
    return containsDecision(superset, decision);
  });
}

bool PathCondition::compatibleWith(const PathCondition &other) const {
  return !decisionsIntersect(trueDecisions, other.falseDecisions) &&
         !decisionsIntersect(falseDecisions, other.trueDecisions);
}

bool PathCondition::compatibleForPacking(const PathCondition &other) const {
  return !hasNonRepeatableConflict(trueDecisions, other.falseDecisions,
                                   repeatableDecisions,
                                   other.repeatableDecisions) &&
         !hasNonRepeatableConflict(falseDecisions, other.trueDecisions,
                                   repeatableDecisions,
                                   other.repeatableDecisions);
}

PathCondition PathCondition::withoutRepeatableDecisions() const {
  return PathCondition{subtractDecisions(trueDecisions, repeatableDecisions),
                       subtractDecisions(falseDecisions, repeatableDecisions),
                       DecisionSet{}};
}

std::optional<PathCondition>
PathCondition::intersect(const PathCondition &other) const {
  if (!compatibleWith(other))
    return std::nullopt;
  return PathCondition{
      unionDecisions(trueDecisions, other.trueDecisions),
      unionDecisions(falseDecisions, other.falseDecisions),
      unionDecisions(repeatableDecisions, other.repeatableDecisions)};
}

bool PathCondition::implies(const PathCondition &other) const {
  return isDecisionSubset(other.trueDecisions, trueDecisions) &&
         isDecisionSubset(other.falseDecisions, falseDecisions);
}

std::optional<PathCondition>
PathCondition::withDecision(uint64_t decision, bool selected,
                            bool repeatable) const {
  PathCondition result = *this;
  DecisionSet &selectedDecisions =
      selected ? result.trueDecisions : result.falseDecisions;
  const DecisionSet &rejectedDecisions =
      selected ? result.falseDecisions : result.trueDecisions;
  if (containsDecision(rejectedDecisions, decision))
    return std::nullopt;
  insertDecision(selectedDecisions, decision);
  if (repeatable)
    insertDecision(result.repeatableDecisions, decision);
  return result;
}

void PathCondition::subtract(
    const PathCondition &covered,
    llvm::SmallVectorImpl<PathCondition> &remaining) const {
  if (!compatibleWith(covered)) {
    remaining.push_back(*this);
    return;
  }
  if (implies(covered))
    return;

  PathCondition prefix = *this;
  auto splitDecision = [&](uint64_t decision, bool coveredSelected) {
    bool repeatable = containsDecision(covered.repeatableDecisions, decision);
    if (std::optional<PathCondition> outside =
            prefix.withDecision(decision, !coveredSelected, repeatable))
      remaining.push_back(*outside);
    if (std::optional<PathCondition> inside =
            prefix.withDecision(decision, coveredSelected, repeatable))
      prefix = *inside;
  };

  for (uint64_t decision : covered.trueDecisions) {
    if (containsDecision(trueDecisions, decision))
      continue;
    splitDecision(decision, /*coveredSelected=*/true);
  }
  for (uint64_t decision : covered.falseDecisions) {
    if (containsDecision(falseDecisions, decision))
      continue;
    splitDecision(decision, /*coveredSelected=*/false);
  }
}

mlir::FailureOr<StructuredTimeline>
StructuredTimeline::build(mlir::Operation *scope, TimelineFailure *failure) {
  if (!scope)
    return mlir::failure();

  StructuredTimeline timeline;
  int64_t nextEvent = 0;
  std::optional<uint64_t> nextDecision = 0;
  bool failed = false;

  auto takeDecision = [&]() -> std::optional<uint64_t> {
    if (!nextDecision)
      return std::nullopt;
    uint64_t decision = *nextDecision;
    if (decision == std::numeric_limits<uint64_t>::max())
      nextDecision.reset();
    else
      ++*nextDecision;
    return decision;
  };

  std::function<void(mlir::Region &, PathCondition, unsigned)> assignRegion;
  std::function<void(mlir::Block &, PathCondition, unsigned)> assignBlock;
  assignBlock = [&](mlir::Block &block, PathCondition path,
                    unsigned loopDepth) {
    for (mlir::Operation &op : block) {
      if (failed)
        return;
      timeline.points[&op] = ProgramPoint{nextEvent++, path};

      // A single-block region can still contain a self-looping cf branch.
      // BranchOpInterface successors are not represented by this structured
      // path domain, so reject them instead of linearizing the CFG.
      if (mlir::isa<mlir::BranchOpInterface>(op)) {
        failed = true;
        setTimelineFailure(
            failure, TimelineFailureKind::UnsupportedRegionControlFlow, &op);
        return;
      }

      if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        std::optional<uint64_t> decision = takeDecision();
        if (!decision) {
          failed = true;
          setTimelineFailure(failure,
                             TimelineFailureKind::DecisionDomainExhausted, &op);
          return;
        }
        std::optional<PathCondition> thenPath =
            path.withDecision(*decision, /*selected=*/true,
                              /*repeatable=*/loopDepth != 0);
        std::optional<PathCondition> elsePath =
            path.withDecision(*decision, /*selected=*/false,
                              /*repeatable=*/loopDepth != 0);
        if (!thenPath || !elsePath) {
          failed = true;
          setTimelineFailure(
              failure, TimelineFailureKind::InconsistentPathCondition, &op);
          return;
        }
        assignRegion(ifOp.getThenRegion(), *thenPath, loopDepth);
        assignRegion(ifOp.getElseRegion(), *elsePath, loopDepth);
      } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        std::optional<uint64_t> decision = takeDecision();
        if (!decision) {
          failed = true;
          setTimelineFailure(failure,
                             TimelineFailureKind::DecisionDomainExhausted, &op);
          return;
        }
        std::optional<PathCondition> bodyPath =
            path.withDecision(*decision, /*selected=*/true,
                              /*repeatable=*/loopDepth != 0);
        if (!bodyPath) {
          failed = true;
          setTimelineFailure(
              failure, TimelineFailureKind::InconsistentPathCondition, &op);
          return;
        }
        assignRegion(forOp.getRegion(), *bodyPath, loopDepth + 1);
      } else if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(op)) {
        assignRegion(tileRegion.getBody(), path, loopDepth);
      } else {
        if (op.getNumRegions() != 0) {
          failed = true;
          setTimelineFailure(
              failure, TimelineFailureKind::UnsupportedRegionControlFlow, &op);
          return;
        }
      }

      timeline.subtreeEnds[&op] = nextEvent - 1;
    }
  };
  assignRegion = [&](mlir::Region &region, PathCondition path,
                     unsigned loopDepth) {
    if (!region.empty() && !region.hasOneBlock()) {
      failed = true;
      setTimelineFailure(failure,
                         TimelineFailureKind::UnsupportedRegionControlFlow,
                         region.getParentOp());
      return;
    }
    for (mlir::Block &block : region)
      assignBlock(block, path, loopDepth);
  };

  for (mlir::Region &region : scope->getRegions())
    assignRegion(region, PathCondition::root(), /*loopDepth=*/0);
  if (failed)
    return mlir::failure();
  return timeline;
}

std::optional<ProgramPoint>
StructuredTimeline::lookup(mlir::Operation *op) const {
  auto it = points.find(op);
  if (it == points.end())
    return std::nullopt;
  return it->second;
}

std::optional<int64_t>
StructuredTimeline::lookupSubtreeEnd(mlir::Operation *op) const {
  auto it = subtreeEnds.find(op);
  if (it == subtreeEnds.end())
    return std::nullopt;
  return it->second;
}

LifetimeDataflow::LifetimeDataflow(
    const StructuredTimeline &timeline,
    llvm::MutableArrayRef<LifetimeDemand> demands,
    TrackedTypePredicate isTrackedType, ValueResolver resolveValue,
    ExplicitRootPredicate isExplicitRoot)
    : timeline(timeline), demands(demands),
      isTrackedType(std::move(isTrackedType)),
      resolveValue(std::move(resolveValue)),
      isExplicitRoot(std::move(isExplicitRoot)) {}

mlir::Value LifetimeDataflow::normalize(mlir::Value value) const {
  if (!value || !resolveValue)
    return value;

  llvm::DenseSet<mlir::Value> seen;
  while (seen.insert(value).second) {
    mlir::Value resolved = resolveValue(value);
    if (!resolved || resolved == value)
      return value;
    value = resolved;
  }
  return value;
}

mlir::LogicalResult LifetimeDataflow::initialize(LifetimeFailure *failure) {
  valueRefs.clear();
  valueOrigins.clear();
  asyncRefs.clear();
  asyncTaskRefs.clear();
  asyncTasks.clear();
  for (auto [index, demand] : llvm::enumerate(demands)) {
    demand.segments.clear();
    if (!demand.allocation) {
      setLifetimeFailure(failure, LifetimeFailureKind::MissingAllocationEvent,
                         nullptr);
      return mlir::failure();
    }
    std::optional<ProgramPoint> point =
        timeline.lookup(demand.allocation.getOperation());
    if (!point) {
      setLifetimeFailure(failure, LifetimeFailureKind::MissingAllocationEvent,
                         demand.allocation.getOperation());
      return mlir::failure();
    }
    demand.allocationPoint = *point;
    valueRefs[normalize(demand.allocation.getMemref())] =
        llvm::SmallVector<RootRef, 2>{
            RootRef{static_cast<unsigned>(index), point->path}};
    valueOrigins[normalize(demand.allocation.getMemref())] =
        llvm::SmallVector<ValueOriginRef, 2>{
            ValueOriginRef{demand.allocation.getMemref(), point->path}};
  }
  return mlir::success();
}

llvm::SmallVector<RootRef, 2>
LifetimeDataflow::rootsAt(mlir::Value value, PathCondition usePath) const {
  llvm::DenseSet<mlir::Value> active;
  std::function<llvm::SmallVector<RootRef, 2>(mlir::Value, PathCondition)>
      collect = [&](mlir::Value current,
                    PathCondition queryPath) -> llvm::SmallVector<RootRef, 2> {
    llvm::SmallVector<RootRef, 2> refs;
    current = normalize(current);
    if (!current)
      return refs;

    auto mapped = valueRefs.find(current);
    if (mapped != valueRefs.end()) {
      for (RootRef ref : mapped->second)
        if (std::optional<PathCondition> path = ref.path.intersect(queryPath))
          appendUniqueRootRefs(
              refs, llvm::ArrayRef<RootRef>{RootRef{ref.demandIndex, *path}});
    }
    if (!active.insert(current).second)
      return refs;

    auto appendValue = [&](mlir::Value source, PathCondition sourcePath) {
      llvm::SmallVector<RootRef, 2> sourceRefs = collect(source, sourcePath);
      appendUniqueRootRefs(refs, sourceRefs);
    };
    auto appendAt = [&](mlir::Value source, std::optional<ProgramPoint> point) {
      if (!point)
        return;
      std::optional<PathCondition> path = queryPath.intersect(point->path);
      if (path)
        appendValue(source, *path);
    };

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          blockArg.getOwner() ? blockArg.getOwner()->getParentOp() : nullptr);
      if (forOp && blockArg.getOwner() == forOp.getBody() &&
          blockArg.getArgNumber() > 0) {
        unsigned index = blockArg.getArgNumber() - 1;
        if (index < forOp.getInitArgs().size())
          appendAt(forOp.getInitArgs()[index], timeline.lookup(forOp));
        mlir::scf::YieldOp yield = mlir::dyn_cast<mlir::scf::YieldOp>(
            forOp.getBody()->getTerminator());
        if (yield && index < yield.getResults().size())
          appendAt(yield.getResults()[index], timeline.lookup(yield));
      } else if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(
                     blockArg.getOwner() ? blockArg.getOwner()->getParentOp()
                                         : nullptr)) {
        unsigned index = blockArg.getArgNumber();
        if (blockArg.getOwner() == &tileRegion.getBody().front() &&
            index < tileRegion.getInputs().size()) {
          if (std::optional<ProgramPoint> point = timeline.lookup(tileRegion))
            appendAt(tileRegion.getInputs()[index], point);
          else
            appendValue(tileRegion.getInputs()[index], queryPath);
        }
      }
    } else if (auto result = mlir::dyn_cast<mlir::OpResult>(current)) {
      mlir::Operation *def = result.getOwner();
      if (auto toMemref =
              mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(def)) {
        appendAt(toMemref.getTensor(), timeline.lookup(def));
      } else if (auto toTensor =
                     mlir::dyn_cast<mlir::bufferization::ToTensorOp>(def)) {
        appendAt(toTensor.getMemref(), timeline.lookup(def));
      } else if (auto reshape = mlir::dyn_cast<ViewReshapeOp>(def)) {
        appendAt(reshape.getSource(), timeline.lookup(def));
      } else if (auto viewLike =
                     mlir::dyn_cast<mlir::ViewLikeOpInterface>(def)) {
        appendAt(viewLike.getViewSource(), timeline.lookup(def));
      } else if (auto select =
                     mlir::dyn_cast<mlir::SelectLikeOpInterface>(def)) {
        appendAt(select.getTrueValue(), timeline.lookup(def));
        appendAt(select.getFalseValue(), timeline.lookup(def));
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
        unsigned index = result.getResultNumber();
        for (mlir::scf::YieldOp yield :
             {getSingleBlockYield(ifOp.getThenRegion()),
              getSingleBlockYield(ifOp.getElseRegion())})
          if (yield && index < yield.getResults().size())
            appendAt(yield.getResults()[index], timeline.lookup(yield));
      } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
        unsigned index = result.getResultNumber();
        if (!isStaticallyNonEmpty(forOp) &&
            index < forOp.getInitArgs().size())
          appendAt(forOp.getInitArgs()[index], timeline.lookup(forOp));
        mlir::scf::YieldOp yield = mlir::dyn_cast<mlir::scf::YieldOp>(
            forOp.getBody()->getTerminator());
        if (yield && index < yield.getResults().size())
          appendAt(yield.getResults()[index], timeline.lookup(yield));
      } else if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(def)) {
        unsigned index = result.getResultNumber();
        auto yield = mlir::dyn_cast<TileYieldOp>(
            tileRegion.getBody().front().getTerminator());
        if (yield && index < yield.getValues().size())
          appendAt(yield.getValues()[index], timeline.lookup(yield));
      }
    }

    active.erase(current);
    return refs;
  };
  return collect(value, usePath);
}

llvm::SmallVector<ValueOriginRef, 2>
LifetimeDataflow::originsAt(mlir::Value value, PathCondition usePath) const {
  llvm::DenseSet<mlir::Value> active;
  std::function<llvm::SmallVector<ValueOriginRef, 2>(mlir::Value,
                                                     PathCondition)>
      collect =
          [&](mlir::Value current,
              PathCondition queryPath) -> llvm::SmallVector<ValueOriginRef, 2> {
    llvm::SmallVector<ValueOriginRef, 2> refs;
    current = normalize(current);
    if (!current)
      return refs;

    auto mapped = valueOrigins.find(current);
    if (mapped != valueOrigins.end()) {
      for (ValueOriginRef ref : mapped->second)
        if (std::optional<PathCondition> path = ref.path.intersect(queryPath))
          appendUniqueOriginRefs(refs, llvm::ArrayRef<ValueOriginRef>{
                                           ValueOriginRef{ref.root, *path}});
    }
    if (!active.insert(current).second)
      return refs;

    bool hasAliasSemantics = false;
    auto appendValue = [&](mlir::Value source, PathCondition sourcePath) {
      llvm::SmallVector<ValueOriginRef, 2> sourceRefs =
          collect(source, sourcePath);
      appendUniqueOriginRefs(refs, sourceRefs);
    };
    auto appendAt = [&](mlir::Value source, std::optional<ProgramPoint> point) {
      if (!point)
        return;
      std::optional<PathCondition> path = queryPath.intersect(point->path);
      if (path)
        appendValue(source, *path);
    };

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          blockArg.getOwner() ? blockArg.getOwner()->getParentOp() : nullptr);
      if (forOp && blockArg.getOwner() == forOp.getBody() &&
          blockArg.getArgNumber() > 0) {
        hasAliasSemantics = true;
        unsigned index = blockArg.getArgNumber() - 1;
        if (index < forOp.getInitArgs().size())
          appendAt(forOp.getInitArgs()[index], timeline.lookup(forOp));
        mlir::scf::YieldOp yield = mlir::dyn_cast<mlir::scf::YieldOp>(
            forOp.getBody()->getTerminator());
        if (yield && index < yield.getResults().size())
          appendAt(yield.getResults()[index], timeline.lookup(yield));
      } else if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(
                     blockArg.getOwner() ? blockArg.getOwner()->getParentOp()
                                         : nullptr)) {
        hasAliasSemantics = true;
        unsigned index = blockArg.getArgNumber();
        if (blockArg.getOwner() == &tileRegion.getBody().front() &&
            index < tileRegion.getInputs().size()) {
          if (std::optional<ProgramPoint> point = timeline.lookup(tileRegion))
            appendAt(tileRegion.getInputs()[index], point);
          else
            appendValue(tileRegion.getInputs()[index], queryPath);
        }
      }
    } else if (auto result = mlir::dyn_cast<mlir::OpResult>(current)) {
      mlir::Operation *def = result.getOwner();
      if (auto toMemref =
              mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(def)) {
        appendAt(toMemref.getTensor(), timeline.lookup(def));
        hasAliasSemantics = true;
        if (refs.empty() && isExplicitRoot && isExplicitRoot(current))
          appendUniqueOriginRefs(refs, llvm::ArrayRef<ValueOriginRef>{
                                           ValueOriginRef{current, queryPath}});
      } else if (auto toTensor =
                     mlir::dyn_cast<mlir::bufferization::ToTensorOp>(def)) {
        hasAliasSemantics = true;
        appendAt(toTensor.getMemref(), timeline.lookup(def));
      } else if (auto reshape = mlir::dyn_cast<ViewReshapeOp>(def)) {
        hasAliasSemantics = true;
        appendAt(reshape.getSource(), timeline.lookup(def));
      } else if (auto viewLike =
                     mlir::dyn_cast<mlir::ViewLikeOpInterface>(def)) {
        hasAliasSemantics = true;
        appendAt(viewLike.getViewSource(), timeline.lookup(def));
      } else if (auto select =
                     mlir::dyn_cast<mlir::SelectLikeOpInterface>(def)) {
        hasAliasSemantics = true;
        appendAt(select.getTrueValue(), timeline.lookup(def));
        appendAt(select.getFalseValue(), timeline.lookup(def));
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
        hasAliasSemantics = true;
        unsigned index = result.getResultNumber();
        for (mlir::scf::YieldOp yield :
             {getSingleBlockYield(ifOp.getThenRegion()),
              getSingleBlockYield(ifOp.getElseRegion())})
          if (yield && index < yield.getResults().size())
            appendAt(yield.getResults()[index], timeline.lookup(yield));
      } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
        hasAliasSemantics = true;
        unsigned index = result.getResultNumber();
        if (!isStaticallyNonEmpty(forOp) &&
            index < forOp.getInitArgs().size())
          appendAt(forOp.getInitArgs()[index], timeline.lookup(forOp));
        mlir::scf::YieldOp yield = mlir::dyn_cast<mlir::scf::YieldOp>(
            forOp.getBody()->getTerminator());
        if (yield && index < yield.getResults().size())
          appendAt(yield.getResults()[index], timeline.lookup(yield));
      } else if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(def)) {
        unsigned index = result.getResultNumber();
        auto yield = mlir::dyn_cast<TileYieldOp>(
            tileRegion.getBody().front().getTerminator());
        std::optional<ProgramPoint> point =
            yield ? timeline.lookup(yield.getOperation()) : std::nullopt;
        hasAliasSemantics = point.has_value();
        if (yield && point && index < yield.getValues().size())
          appendAt(yield.getValues()[index], point);
      }
    }

    llvm::SmallVector<PathCondition, 2> uncoveredPaths{queryPath};
    for (ValueOriginRef ref : refs) {
      llvm::SmallVector<PathCondition, 2> nextUncovered;
      for (PathCondition path : uncoveredPaths)
        path.subtract(ref.path, nextUncovered);
      uncoveredPaths = std::move(nextUncovered);
    }
    if (!hasAliasSemantics && isTrackedType && isTrackedType(current.getType()))
      for (PathCondition path : uncoveredPaths)
        appendUniqueOriginRefs(refs, llvm::ArrayRef<ValueOriginRef>{
                                         ValueOriginRef{current, path}});

    active.erase(current);
    return refs;
  };
  return collect(value, usePath);
}

llvm::SmallVector<RootRef, 2>
LifetimeDataflow::asyncRootsAt(mlir::Value handle,
                               PathCondition usePath) const {
  llvm::SmallVector<RootRef, 2> refs;
  auto it = asyncRefs.find(handle);
  if (it == asyncRefs.end())
    return refs;
  for (RootRef ref : it->second)
    if (std::optional<PathCondition> path = ref.path.intersect(usePath))
      refs.push_back(RootRef{ref.demandIndex, *path});
  return refs;
}

llvm::SmallVector<LifetimeDataflow::AsyncTaskRef, 2>
LifetimeDataflow::asyncTasksAt(mlir::Value handle,
                               PathCondition usePath) const {
  llvm::SmallVector<AsyncTaskRef, 2> refs;
  auto it = asyncTaskRefs.find(handle);
  if (it == asyncTaskRefs.end())
    return refs;
  for (AsyncTaskRef ref : it->second)
    if (std::optional<PathCondition> path = ref.path.intersect(usePath))
      appendUniqueAsyncTaskRefs(refs, llvm::ArrayRef<AsyncTaskRef>{
                                          AsyncTaskRef{ref.taskIndex, *path}});
  return refs;
}

void LifetimeDataflow::appendUniqueAsyncTaskRefs(
    llvm::SmallVectorImpl<AsyncTaskRef> &destination,
    llvm::ArrayRef<AsyncTaskRef> source) {
  for (AsyncTaskRef ref : source) {
    if (llvm::any_of(destination, [&](AsyncTaskRef existing) {
          return existing.taskIndex == ref.taskIndex &&
                 existing.path == ref.path;
        }))
      continue;
    destination.push_back(ref);
  }
}

bool LifetimeDataflow::sameAsyncTaskRefs(llvm::ArrayRef<AsyncTaskRef> lhs,
                                         llvm::ArrayRef<AsyncTaskRef> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(lhs, [&](AsyncTaskRef ref) {
    return llvm::any_of(rhs, [&](AsyncTaskRef other) {
      return ref.taskIndex == other.taskIndex && ref.path == other.path;
    });
  });
}

bool LifetimeDataflow::asyncTaskPathsCover(llvm::ArrayRef<AsyncTaskRef> refs,
                                           unsigned taskIndex,
                                           PathCondition requiredPath) {
  llvm::SmallVector<PathCondition, 2> remaining{requiredPath};
  for (AsyncTaskRef ref : refs) {
    if (ref.taskIndex != taskIndex)
      continue;
    llvm::SmallVector<PathCondition, 2> next;
    for (PathCondition path : remaining)
      path.subtract(ref.path, next);
    remaining = std::move(next);
  }
  return remaining.empty();
}

void LifetimeDataflow::recordUse(RootRef ref, int64_t event) {
  if (ref.demandIndex >= demands.size())
    return;
  LifetimeDemand &demand = demands[ref.demandIndex];
  if (event < demand.allocationPoint.event)
    return;
  // Every segment for an allocation starts at the same allocation event.
  // Repeated instruction uses on the same structured path therefore only
  // extend that path's live interval; retaining one segment per use makes
  // packing quadratic in the instruction count without adding information.
  for (LiveSegment &segment : demand.segments) {
    if (segment.beginEvent != demand.allocationPoint.event ||
        segment.path != ref.path)
      continue;
    segment.endEvent = std::max(segment.endEvent, event);
    return;
  }
  demand.segments.push_back(
      LiveSegment{demand.allocationPoint.event, event, ref.path});
}

void LifetimeDataflow::extendTo(RootRef ref, ProgramPoint completionPoint) {
  std::optional<PathCondition> path = ref.path.intersect(completionPoint.path);
  if (!path)
    return;
  recordUse(RootRef{ref.demandIndex, *path}, completionPoint.event);
}

void LifetimeDataflow::recordOperands(mlir::Operation *op) {
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!point)
    return;
  for (mlir::Value operand : op->getOperands()) {
    llvm::SmallVector<RootRef, 2> refs =
        hasAsyncDependencyType(operand) ? asyncRootsAt(operand, point->path)
                                        : rootsAt(operand, point->path);
    for (RootRef ref : refs)
      recordUse(ref, point->event);
  }
}

void LifetimeDataflow::mapViewLikeResults(mlir::Operation *op) {
  auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op);
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!point)
    return;
  mlir::Value source;
  if (viewLike)
    source = viewLike.getViewSource();
  else if (auto reshape = mlir::dyn_cast<ViewReshapeOp>(op))
    source = reshape.getSource();
  if (!source)
    return;
  llvm::SmallVector<RootRef, 2> refs = rootsAt(source, point->path);
  llvm::SmallVector<ValueOriginRef, 2> origins = originsAt(source, point->path);
  for (mlir::Value result : op->getResults()) {
    if (!refs.empty())
      valueRefs[normalize(result)] = refs;
    if (!origins.empty())
      valueOrigins[normalize(result)] = origins;
  }
}

mlir::LogicalResult
LifetimeDataflow::mapSelectLikeResult(mlir::Operation *op,
                                      LifetimeFailure *failure) {
  auto select = mlir::dyn_cast<mlir::SelectLikeOpInterface>(op);
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!select || !point || op->getNumResults() != 1)
    return mlir::success();

  mlir::Value result = op->getResult(0);
  if (!hasAsyncDependencyType(result)) {
    llvm::SmallVector<RootRef, 2> refs;
    llvm::SmallVector<ValueOriginRef, 2> origins;
    for (mlir::Value selected :
         {select.getTrueValue(), select.getFalseValue()}) {
      llvm::SmallVector<RootRef, 2> selectedRefs =
          rootsAt(selected, point->path);
      refs.append(selectedRefs.begin(), selectedRefs.end());
      llvm::SmallVector<ValueOriginRef, 2> selectedOrigins =
          originsAt(selected, point->path);
      origins.append(selectedOrigins.begin(), selectedOrigins.end());
    }
    if (!refs.empty())
      valueRefs[normalize(result)] = std::move(refs);
    if (!origins.empty())
      valueOrigins[normalize(result)] = std::move(origins);
    return mlir::success();
  }

  if (!hasAsyncDependencyType(result))
    return mlir::success();

  llvm::SmallVector<RootRef, 2> refs;
  for (mlir::Value selected : {select.getTrueValue(), select.getFalseValue()}) {
    llvm::SmallVector<RootRef, 2> selectedRefs = rootsAt(selected, point->path);
    if (hasAsyncDependencyType(selected))
      selectedRefs = asyncRootsAt(selected, point->path);
    appendUniqueRootRefs(refs, selectedRefs);
  }
  if (!refs.empty())
    asyncRefs[result] = std::move(refs);

  llvm::SmallVector<AsyncTaskRef, 2> trueTasks =
      asyncTasksAt(select.getTrueValue(), point->path);
  llvm::SmallVector<AsyncTaskRef, 2> falseTasks =
      asyncTasksAt(select.getFalseValue(), point->path);
  // SelectLike has data-dependent choice rather than a structured path. An
  // await of its result completes only the selected task, so a union of
  // distinct task identities would unsoundly discharge the unselected task.
  if (!sameAsyncTaskRefs(trueTasks, falseTasks)) {
    setLifetimeFailure(failure,
                       LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
    return mlir::failure();
  }
  if (!trueTasks.empty())
    asyncTaskRefs[result] = std::move(trueTasks);
  return mlir::success();
}

mlir::LogicalResult
LifetimeDataflow::mapDirectCallResults(mlir::Operation *op,
                                       LifetimeFailure *failure) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!call || !point || !isSupportedDirectAliasCall(call, isTrackedType))
    return mlir::success();

  for (auto [index, result] : llvm::enumerate(call.getResults())) {
    llvm::SmallVector<unsigned, 2> formalIndices;
    if (!getDirectCallResultFormalAliases(call, index, formalIndices)) {
      if (!isTrackedType(result.getType()))
        continue;
      setLifetimeFailure(
          failure, LifetimeFailureKind::UnsupportedTrackedValueProducer, op);
      return mlir::failure();
    }

    llvm::SmallVector<RootRef, 2> roots;
    llvm::SmallVector<ValueOriginRef, 2> origins;
    for (unsigned formalIndex : formalIndices) {
      if (formalIndex >= call.getNumOperands()) {
        setLifetimeFailure(
            failure, LifetimeFailureKind::UnsupportedTrackedValueProducer, op);
        return mlir::failure();
      }
      mlir::Value actual = call.getOperand(formalIndex);
      appendUniqueRootRefs(roots, rootsAt(actual, point->path));
      appendUniqueOriginRefs(origins, originsAt(actual, point->path));
    }
    if (!roots.empty())
      valueRefs[normalize(result)] = std::move(roots);
    if (!origins.empty())
      valueOrigins[normalize(result)] = std::move(origins);
  }
  return mlir::success();
}

mlir::LogicalResult
LifetimeDataflow::mapAsyncDependencyResults(mlir::Operation *op,
                                            LifetimeFailure *failure) {
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!point)
    return mlir::success();
  if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp, TileRegionOp>(op) ||
      mlir::isa<mlir::SelectLikeOpInterface>(op))
    return mlir::success();

  // async.group is a mutable completion handle: add_to_group does not return
  // a new group SSA value. Only a direct create_group handle is accepted for
  // mutation/await; aliasing a mutable group would require a memory-SSA model.
  if (auto addToGroup = mlir::dyn_cast<mlir::async::AddToGroupOp>(op)) {
    llvm::SmallVector<RootRef, 2> addedRoots =
        asyncRootsAt(addToGroup.getOperand(), point->path);
    llvm::SmallVector<AsyncTaskRef, 2> addedTasks =
        asyncTasksAt(addToGroup.getOperand(), point->path);
    if ((!addedRoots.empty() || !addedTasks.empty()) &&
        !mlir::isa_and_nonnull<mlir::async::CreateGroupOp>(
            addToGroup.getGroup().getDefiningOp())) {
      setLifetimeFailure(
          failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
      return mlir::failure();
    }
    if (mlir::scf::ForOp loop = getEnclosingFor(op)) {
      bool addsDynamicTask = llvm::any_of(addedTasks, [&](AsyncTaskRef ref) {
        return ref.taskIndex < asyncTasks.size() &&
               isNestedIn(asyncTasks[ref.taskIndex].origin, loop);
      });
      if (addsDynamicTask) {
        setLifetimeFailure(
            failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
        return mlir::failure();
      }
    }
    llvm::SmallVector<RootRef, 2> combined = asyncRefs[addToGroup.getGroup()];
    appendUniqueRootRefs(combined, addedRoots);
    asyncRefs[addToGroup.getGroup()] = std::move(combined);
    llvm::SmallVector<AsyncTaskRef, 2> combinedTasks =
        asyncTaskRefs[addToGroup.getGroup()];
    appendUniqueAsyncTaskRefs(combinedTasks, addedTasks);
    asyncTaskRefs[addToGroup.getGroup()] = std::move(combinedTasks);
    return mlir::success();
  }

  if (auto awaitAll = mlir::dyn_cast<mlir::async::AwaitAllOp>(op)) {
    llvm::SmallVector<RootRef, 2> roots =
        asyncRootsAt(awaitAll.getOperand(), point->path);
    llvm::SmallVector<AsyncTaskRef, 2> tasks =
        asyncTasksAt(awaitAll.getOperand(), point->path);
    if ((!roots.empty() || !tasks.empty()) &&
        !mlir::isa_and_nonnull<mlir::async::CreateGroupOp>(
            awaitAll.getOperand().getDefiningOp())) {
      setLifetimeFailure(
          failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
      return mlir::failure();
    }
    return mlir::success();
  }
  if (mlir::isa<mlir::async::CreateGroupOp, mlir::async::AwaitOp,
                InstrDTEWaitOp>(op))
    return mlir::success();

  llvm::SmallVector<RootRef, 2> directRoots;
  llvm::SmallVector<PathCondition, 2> directAccessPaths;
  llvm::SmallVector<AsyncTaskRef, 2> operandTasks;
  for (mlir::Value operand : op->getOperands()) {
    if (hasAsyncDependencyType(operand)) {
      llvm::SmallVector<AsyncTaskRef, 2> tasks =
          asyncTasksAt(operand, point->path);
      appendUniqueAsyncTaskRefs(operandTasks, tasks);
      continue;
    }
    llvm::SmallVector<RootRef, 2> roots = rootsAt(operand, point->path);
    appendUniqueRootRefs(directRoots, roots);
    for (ValueOriginRef origin : originsAt(operand, point->path))
      if (!llvm::is_contained(directAccessPaths, origin.path))
        directAccessPaths.push_back(origin.path);
  }

  bool hasAsyncResult = llvm::any_of(op->getResults(), [](mlir::Value result) {
    return hasAsyncDependencyType(result);
  });
  if (!hasAsyncResult)
    return mlir::success();

  if (!directAccessPaths.empty()) {
    if (auto call = mlir::dyn_cast<mlir::async::CallOp>(op)) {
      auto callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::async::FuncOp>(
              call, call.getCalleeAttr());
      bool erasedTrackedType =
          !callee || callee.isExternal() ||
          callee.getArgumentTypes().size() != call.getArgOperands().size();
      if (!erasedTrackedType) {
        for (auto [index, operand] : llvm::enumerate(call.getArgOperands())) {
          if (originsAt(operand, point->path).empty())
            continue;
          if (!isTrackedType ||
              !isTrackedType(callee.getArgumentTypes()[index])) {
            erasedTrackedType = true;
            break;
          }
        }
      }
      if (erasedTrackedType) {
        setLifetimeFailure(
            failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
        return mlir::failure();
      }
    }
  }

  if (!operandTasks.empty() ||
      (!directAccessPaths.empty() && !isSupportedAsyncTaskProducer(op))) {
    setLifetimeFailure(failure,
                       LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
    return mlir::failure();
  }
  if (directAccessPaths.empty())
    return mlir::success();

  AsyncTaskState task;
  task.origin = op;
  task.pendingPaths = directAccessPaths;
  unsigned taskIndex = asyncTasks.size();
  asyncTasks.push_back(std::move(task));

  llvm::SmallVector<AsyncTaskRef, 2> taskRefs;
  for (PathCondition path : asyncTasks.back().pendingPaths)
    taskRefs.push_back(AsyncTaskRef{taskIndex, path});
  for (mlir::Value result : op->getResults()) {
    if (!hasAsyncDependencyType(result))
      continue;
    asyncRefs[result] = directRoots;
    asyncTaskRefs[result] = taskRefs;
  }
  return mlir::success();
}

void LifetimeDataflow::completeAsyncTasks(mlir::Operation *op) {
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!point)
    return;

  llvm::SmallVector<mlir::Value, 2> handles;
  if (auto await = mlir::dyn_cast<mlir::async::AwaitOp>(op)) {
    handles.push_back(await.getOperand());
  } else if (auto awaitAll = mlir::dyn_cast<mlir::async::AwaitAllOp>(op)) {
    handles.push_back(awaitAll.getOperand());
  } else if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op)) {
    handles.append(wait.getTokens().begin(), wait.getTokens().end());
  } else {
    return;
  }

  for (mlir::Value handle : handles) {
    for (AsyncTaskRef completed : asyncTasksAt(handle, point->path)) {
      if (completed.taskIndex >= asyncTasks.size())
        continue;
      AsyncTaskState &task = asyncTasks[completed.taskIndex];
      llvm::SmallVector<PathCondition, 2> remaining;
      for (PathCondition pending : task.pendingPaths)
        pending.subtract(completed.path, remaining);
      task.pendingPaths = std::move(remaining);
    }
  }
}

mlir::LogicalResult
LifetimeDataflow::finishAsyncTasks(LifetimeFailure *failure) const {
  for (const AsyncTaskState &task : asyncTasks) {
    if (task.pendingPaths.empty())
      continue;
    setLifetimeFailure(failure, LifetimeFailureKind::MissingAsyncCompletion,
                       task.origin);
    return mlir::failure();
  }
  return mlir::success();
}

void LifetimeDataflow::mapIfResults(mlir::Operation *op) {
  auto ifOp = mlir::cast<mlir::scf::IfOp>(op);
  mlir::scf::YieldOp thenYield = getSingleBlockYield(ifOp.getThenRegion());
  mlir::scf::YieldOp elseYield = getSingleBlockYield(ifOp.getElseRegion());
  if (!thenYield || !elseYield)
    return;

  for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
    if (!hasAsyncDependencyType(result)) {
      llvm::SmallVector<RootRef, 2> refs;
      llvm::SmallVector<ValueOriginRef, 2> origins;
      for (mlir::scf::YieldOp yield : {thenYield, elseYield}) {
        if (index >= yield.getResults().size())
          continue;
        std::optional<ProgramPoint> point = timeline.lookup(yield);
        if (!point)
          continue;
        llvm::SmallVector<RootRef, 2> yielded =
            rootsAt(yield.getResults()[index], point->path);
        refs.append(yielded.begin(), yielded.end());
        llvm::SmallVector<ValueOriginRef, 2> yieldedOrigins =
            originsAt(yield.getResults()[index], point->path);
        origins.append(yieldedOrigins.begin(), yieldedOrigins.end());
      }
      if (!refs.empty())
        valueRefs[normalize(result)] = std::move(refs);
      if (!origins.empty())
        valueOrigins[normalize(result)] = std::move(origins);
      continue;
    }
    if (!hasAsyncDependencyType(result))
      continue;

    llvm::SmallVector<RootRef, 2> refs;
    llvm::SmallVector<AsyncTaskRef, 2> taskRefs;
    for (mlir::scf::YieldOp yield : {thenYield, elseYield}) {
      if (index >= yield.getResults().size())
        continue;
      std::optional<ProgramPoint> point = timeline.lookup(yield);
      if (!point)
        continue;
      llvm::SmallVector<RootRef, 2> yielded =
          asyncRootsAt(yield.getResults()[index], point->path);
      appendUniqueRootRefs(refs, yielded);
      llvm::SmallVector<AsyncTaskRef, 2> yieldedTasks =
          asyncTasksAt(yield.getResults()[index], point->path);
      appendUniqueAsyncTaskRefs(taskRefs, yieldedTasks);
    }
    if (!refs.empty())
      asyncRefs[result] = std::move(refs);
    if (!taskRefs.empty())
      asyncTaskRefs[result] = std::move(taskRefs);
  }
}

void LifetimeDataflow::mapForRegionIterArgs(mlir::Operation *op) {
  auto forOp = mlir::cast<mlir::scf::ForOp>(op);
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!point)
    return;
  for (auto [init, iterArg] :
       llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
    if (hasAsyncDependencyType(init)) {
      llvm::SmallVector<RootRef, 2> roots = asyncRootsAt(init, point->path);
      if (!roots.empty())
        asyncRefs[iterArg] = std::move(roots);
      llvm::SmallVector<AsyncTaskRef, 2> tasks =
          asyncTasksAt(init, point->path);
      if (!tasks.empty())
        asyncTaskRefs[iterArg] = std::move(tasks);
      continue;
    }
    llvm::SmallVector<RootRef, 2> refs = rootsAt(init, point->path);
    if (!refs.empty())
      valueRefs[normalize(iterArg)] = std::move(refs);
    llvm::SmallVector<ValueOriginRef, 2> origins = originsAt(init, point->path);
    if (!origins.empty())
      valueOrigins[normalize(iterArg)] = std::move(origins);
  }
}

void LifetimeDataflow::mapTileRegionBlockArgs(mlir::Operation *op) {
  auto tileRegion = mlir::cast<TileRegionOp>(op);
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!point || tileRegion.getBody().empty())
    return;

  for (auto [input, blockArg] :
       llvm::zip(tileRegion.getInputs(),
                 tileRegion.getBody().front().getArguments())) {
    if (hasAsyncDependencyType(input)) {
      llvm::SmallVector<RootRef, 2> roots = asyncRootsAt(input, point->path);
      if (!roots.empty())
        asyncRefs[blockArg] = std::move(roots);
      llvm::SmallVector<AsyncTaskRef, 2> tasks =
          asyncTasksAt(input, point->path);
      if (!tasks.empty())
        asyncTaskRefs[blockArg] = std::move(tasks);
      continue;
    }

    llvm::SmallVector<RootRef, 2> roots = rootsAt(input, point->path);
    if (!roots.empty())
      valueRefs[normalize(blockArg)] = std::move(roots);
    llvm::SmallVector<ValueOriginRef, 2> origins =
        originsAt(input, point->path);
    if (!origins.empty())
      valueOrigins[normalize(blockArg)] = std::move(origins);
  }
}

void LifetimeDataflow::mapTileRegionResults(mlir::Operation *op) {
  auto tileRegion = mlir::cast<TileRegionOp>(op);
  if (tileRegion.getBody().empty())
    return;
  auto yield =
      mlir::dyn_cast<TileYieldOp>(tileRegion.getBody().front().getTerminator());
  std::optional<ProgramPoint> point =
      yield ? timeline.lookup(yield.getOperation()) : std::nullopt;
  if (!yield || !point)
    return;

  for (auto [yielded, result] :
       llvm::zip(yield.getValues(), tileRegion.getResults())) {
    if (hasAsyncDependencyType(result)) {
      llvm::SmallVector<RootRef, 2> roots = asyncRootsAt(yielded, point->path);
      if (!roots.empty())
        asyncRefs[result] = std::move(roots);
      llvm::SmallVector<AsyncTaskRef, 2> tasks =
          asyncTasksAt(yielded, point->path);
      if (!tasks.empty())
        asyncTaskRefs[result] = std::move(tasks);
      continue;
    }

    llvm::SmallVector<RootRef, 2> roots = rootsAt(yielded, point->path);
    if (!roots.empty())
      valueRefs[normalize(result)] = std::move(roots);
    llvm::SmallVector<ValueOriginRef, 2> origins =
        originsAt(yielded, point->path);
    if (!origins.empty())
      valueOrigins[normalize(result)] = std::move(origins);
  }
}

mlir::LogicalResult
LifetimeDataflow::mapForResultsAndBackedge(mlir::Operation *op,
                                           LifetimeFailure *failure) {
  auto forOp = mlir::cast<mlir::scf::ForOp>(op);
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
  std::optional<ProgramPoint> loopPoint = timeline.lookup(op);
  std::optional<int64_t> loopEnd = timeline.lookupSubtreeEnd(op);
  if (!yield || !loopPoint || !loopEnd)
    return mlir::success();

  std::optional<ProgramPoint> yieldPoint = timeline.lookup(yield);
  for (auto [index, result] : llvm::enumerate(forOp.getResults())) {
    if (hasAsyncDependencyType(result)) {
      llvm::SmallVector<RootRef, 2> refs;
      llvm::SmallVector<AsyncTaskRef, 2> initialTasks;
      llvm::SmallVector<AsyncTaskRef, 2> backedgeTasks;
      if (index < forOp.getInitArgs().size()) {
        llvm::SmallVector<RootRef, 2> initial =
            asyncRootsAt(forOp.getInitArgs()[index], loopPoint->path);
        appendUniqueRootRefs(refs, initial);
        initialTasks =
            asyncTasksAt(forOp.getInitArgs()[index], loopPoint->path);
      }
      if (yieldPoint && index < yield.getResults().size()) {
        llvm::SmallVector<RootRef, 2> backedge =
            asyncRootsAt(yield.getResults()[index], yieldPoint->path);
        for (RootRef ref : backedge) {
          if (ref.demandIndex >= demands.size())
            continue;
          mlir::Operation *allocation =
              demands[ref.demandIndex].allocation.getOperation();
          if (!isNestedIn(allocation, op))
            continue;
          setLifetimeFailure(failure,
                             LifetimeFailureKind::LoopCarriedAllocationInstance,
                             allocation);
          return mlir::failure();
        }
        appendUniqueRootRefs(refs, backedge);
        backedgeTasks =
            asyncTasksAt(yield.getResults()[index], yieldPoint->path);
      }

      // A loop result selects the init handle on the zero-trip path and the
      // yielded handle otherwise. A union of different completion identities
      // would let one await incorrectly discharge a task that was not selected.
      // Accept only an identity-preserving recurrence whose backedge covers
      // every path through the loop body.
      for (AsyncTaskRef ref : backedgeTasks) {
        if (ref.taskIndex < asyncTasks.size() &&
            isNestedIn(asyncTasks[ref.taskIndex].origin, op)) {
          setLifetimeFailure(
              failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow,
              asyncTasks[ref.taskIndex].origin);
          return mlir::failure();
        }
        if (!llvm::any_of(initialTasks, [&](AsyncTaskRef initial) {
              return initial.taskIndex == ref.taskIndex;
            })) {
          setLifetimeFailure(
              failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
          return mlir::failure();
        }
      }
      if (yieldPoint) {
        for (AsyncTaskRef initial : initialTasks) {
          std::optional<PathCondition> required =
              initial.path.intersect(yieldPoint->path);
          if (required && !asyncTaskPathsCover(backedgeTasks, initial.taskIndex,
                                               *required)) {
            setLifetimeFailure(
                failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow,
                op);
            return mlir::failure();
          }
        }
      }
      // A loop-local branch can select differently on the next dynamic
      // iteration. Once a completion handle crosses the backedge, those
      // repeatable decisions no longer constrain which root it may refer to.
      forgetRepeatableDecisions(refs);
      for (RootRef ref : refs)
        if (ref.demandIndex < demands.size())
          demands[ref.demandIndex].segments.push_back(
              LiveSegment{loopPoint->event, *loopEnd, ref.path});
      if (!refs.empty()) {
        asyncRefs[result] = refs;
        if (index < forOp.getRegionIterArgs().size())
          asyncRefs[forOp.getRegionIterArgs()[index]] = std::move(refs);
      }
      if (!initialTasks.empty()) {
        asyncTaskRefs[result] = initialTasks;
        if (index < forOp.getRegionIterArgs().size())
          asyncTaskRefs[forOp.getRegionIterArgs()[index]] =
              std::move(initialTasks);
      }
      continue;
    }
    llvm::SmallVector<RootRef, 2> initialRefs;
    llvm::SmallVector<ValueOriginRef, 2> initialOrigins;
    if (index < forOp.getInitArgs().size()) {
      initialRefs = rootsAt(forOp.getInitArgs()[index], loopPoint->path);
      initialOrigins =
          originsAt(forOp.getInitArgs()[index], loopPoint->path);
    }
    llvm::SmallVector<RootRef, 2> backedgeRefs;
    llvm::SmallVector<ValueOriginRef, 2> backedgeOrigins;
    if (yieldPoint && index < yield.getResults().size()) {
      backedgeRefs = rootsAt(yield.getResults()[index], yieldPoint->path);
      for (RootRef ref : backedgeRefs) {
        if (ref.demandIndex >= demands.size())
          continue;
        mlir::Operation *allocation =
            demands[ref.demandIndex].allocation.getOperation();
        if (!isNestedIn(allocation, op))
          continue;
        setLifetimeFailure(failure,
                           LifetimeFailureKind::LoopCarriedAllocationInstance,
                           allocation);
        return mlir::failure();
      }
      backedgeOrigins =
          originsAt(yield.getResults()[index], yieldPoint->path);
    }

    llvm::SmallVector<RootRef, 2> recurrenceRefs = initialRefs;
    appendUniqueRootRefs(recurrenceRefs, backedgeRefs);
    llvm::SmallVector<ValueOriginRef, 2> recurrenceOrigins = initialOrigins;
    appendUniqueOriginRefs(recurrenceOrigins, backedgeOrigins);
    llvm::SmallVector<RootRef, 2> resultRefs =
        isStaticallyNonEmpty(forOp) ? backedgeRefs : recurrenceRefs;
    llvm::SmallVector<ValueOriginRef, 2> resultOrigins =
        isStaticallyNonEmpty(forOp) ? backedgeOrigins : recurrenceOrigins;

    // The same recurrence may traverse a different loop-local branch on a
    // later iteration. Preserve outer non-repeatable control conditions but
    // forget repeatable decisions before publishing the fixed-point union.
    forgetRepeatableDecisions(recurrenceRefs);
    forgetRepeatableDecisions(recurrenceOrigins);
    forgetRepeatableDecisions(resultRefs);
    forgetRepeatableDecisions(resultOrigins);

    for (RootRef ref : recurrenceRefs)
      if (ref.demandIndex < demands.size())
        demands[ref.demandIndex].segments.push_back(
            LiveSegment{loopPoint->event, *loopEnd, ref.path});
    if (!recurrenceRefs.empty() &&
        index < forOp.getRegionIterArgs().size())
      valueRefs[normalize(forOp.getRegionIterArgs()[index])] = recurrenceRefs;
    if (!resultRefs.empty())
      valueRefs[normalize(result)] = std::move(resultRefs);
    if (!recurrenceOrigins.empty() &&
        index < forOp.getRegionIterArgs().size())
      valueOrigins[normalize(forOp.getRegionIterArgs()[index])] =
          recurrenceOrigins;
    if (!resultOrigins.empty())
      valueOrigins[normalize(result)] = std::move(resultOrigins);
  }

  // A root defined outside the loop and referenced from its body may be read
  // again on every dynamic iteration. Static preorder alone would otherwise
  // let a later body-local allocation overwrite that root before the next
  // backedge. Conservatively keep every such captured root live across the
  // complete loop subtree. Body-local roots are recreated each iteration and
  // remain governed by the backedge completion proof; loop-carried ones were
  // handled above (including the dynamic-allocation rejection).
  for (LifetimeDemand &demand : demands) {
    mlir::Operation *allocation = demand.allocation.getOperation();
    if (isNestedIn(allocation, op))
      continue;
    bool referencedInBody =
        llvm::any_of(demand.segments, [&](const LiveSegment &segment) {
          return segment.endEvent > loopPoint->event &&
                 segment.endEvent <= *loopEnd;
        });
    if (referencedInBody)
      demand.segments.push_back(
          LiveSegment{loopPoint->event, *loopEnd, loopPoint->path});
  }
  return mlir::success();
}

mlir::LogicalResult
LifetimeDataflow::processRegion(mlir::Region &region,
                                LocalCompletionTracker *localCompletion,
                                LifetimeFailure *failure) {
  for (mlir::Block &block : region)
    if (mlir::failed(processBlock(block, localCompletion, failure)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult
LifetimeDataflow::processBlock(mlir::Block &block,
                               LocalCompletionTracker *localCompletion,
                               LifetimeFailure *failure) {
  for (mlir::Operation &op : block) {
    recordOperands(&op);

    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      mapForRegionIterArgs(&op);
      if (mlir::failed(
              processRegion(forOp.getRegion(), localCompletion, failure)))
        return mlir::failure();
      if (localCompletion &&
          mlir::failed(localCompletion->verifyLoopBackedge(&op, failure)))
        return mlir::failure();
      if (mlir::failed(mapForResultsAndBackedge(&op, failure)))
        return mlir::failure();
    } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      if (mlir::failed(
              processRegion(ifOp.getThenRegion(), localCompletion, failure)) ||
          mlir::failed(
              processRegion(ifOp.getElseRegion(), localCompletion, failure)))
        return mlir::failure();
      mapIfResults(&op);
    } else if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(op)) {
      mapTileRegionBlockArgs(&op);
      if (mlir::failed(
              processRegion(tileRegion.getBody(), localCompletion, failure)))
        return mlir::failure();
      mapTileRegionResults(&op);
    } else {
      for (mlir::Region &region : op.getRegions())
        if (mlir::failed(processRegion(region, localCompletion, failure)))
          return mlir::failure();
      mapViewLikeResults(&op);
      if (mlir::failed(mapSelectLikeResult(&op, failure)))
        return mlir::failure();
      if (mlir::failed(mapDirectCallResults(&op, failure)))
        return mlir::failure();
    }

    std::optional<ProgramPoint> point = timeline.lookup(&op);
    bool hasTrackedOperand =
        point && llvm::any_of(op.getOperands(), [&](mlir::Value operand) {
          return !hasAsyncDependencyType(operand) &&
                 !originsAt(operand, point->path).empty();
        });
    bool exposesRawMetadata =
        mlir::isa<mlir::memref::ExtractAlignedPointerAsIndexOp,
                  mlir::memref::ExtractStridedMetadataOp>(op);
    bool hasSupportedTrackedUse =
        mlir::isa<
            mlir::memref::LoadOp, mlir::memref::StoreOp, mlir::memref::CopyOp,
            mlir::memref::DeallocOp, mlir::memref::DimOp, mlir::memref::RankOp,
            mlir::memref::PrefetchOp, mlir::memref::AssumeAlignmentOp,
            mlir::memref::AtomicRMWOp, mlir::async::CallOp,
            mlir::bufferization::ToMemrefOp, mlir::bufferization::ToTensorOp,
            mlir::scf::IfOp, mlir::scf::ForOp, TileRegionOp>(op) ||
        mlir::isa<mlir::ViewLikeOpInterface, mlir::SelectLikeOpInterface,
                  mlir::MemoryEffectOpInterface>(op) ||
        mlir::isa<ViewReshapeOp>(op) ||
        op.hasTrait<mlir::OpTrait::IsTerminator>();
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
      hasSupportedTrackedUse = hasSupportedTrackedUse ||
                               isSupportedDirectAliasCall(call, isTrackedType);

    bool hasUnsupportedScopeEscape = false;
    if (point && mlir::isa<mlir::func::ReturnOp, mlir::async::ReturnOp>(op)) {
      for (mlir::Value operand : op.getOperands()) {
        if (hasAsyncDependencyType(operand))
          continue;
        llvm::SmallVector<ValueOriginRef, 2> origins =
            originsAt(operand, point->path);
        if (origins.empty())
          continue;
        if (mlir::isa<mlir::async::ReturnOp>(op) ||
            (mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
             (!isTrackedType || !isTrackedType(operand.getType())))) {
          hasUnsupportedScopeEscape = true;
          break;
        }
      }
    }
    if (hasTrackedOperand && (exposesRawMetadata || !hasSupportedTrackedUse)) {
      setLifetimeFailure(
          failure, LifetimeFailureKind::UnsupportedTrackedValueEscape, &op);
      return mlir::failure();
    }
    if (hasUnsupportedScopeEscape) {
      setLifetimeFailure(
          failure, LifetimeFailureKind::UnsupportedTrackedValueEscape, &op);
      return mlir::failure();
    }

    bool hasTrackedResult =
        isTrackedType && llvm::any_of(op.getResultTypes(), isTrackedType);
    bool hasKnownTrackedSemantics = mlir::isa<mlir::memref::AllocOp>(op);
    bool hasAliasProducerSemantics =
        mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp, TileRegionOp,
                  mlir::bufferization::ToMemrefOp, ViewReshapeOp>(op) ||
        mlir::isa<mlir::ViewLikeOpInterface, mlir::SelectLikeOpInterface>(op);
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
      hasAliasProducerSemantics =
          hasAliasProducerSemantics ||
          isSupportedDirectAliasCall(call, isTrackedType);
    if (!hasKnownTrackedSemantics && hasTrackedResult &&
        hasAliasProducerSemantics && point)
      hasKnownTrackedSemantics =
          llvm::all_of(op.getResults(), [&](mlir::Value result) {
            if (!isTrackedType(result.getType()))
              return true;
            return !rootsAt(result, point->path).empty() ||
                   !originsAt(result, point->path).empty();
          });
    // An explicit-root predicate is an owner-supplied proof for a supported
    // root form, not a generic escape hatch for arbitrary tracked producers.
    // Function-entry adapters are handled by their ToMemref alias semantics;
    // the only otherwise-unknown producer admitted here is memref.get_global.
    if (!hasKnownTrackedSemantics && hasTrackedResult && isExplicitRoot &&
        mlir::isa<mlir::memref::GetGlobalOp>(op))
      hasKnownTrackedSemantics =
          llvm::all_of(op.getResults(), [&](mlir::Value result) {
            return !isTrackedType(result.getType()) || isExplicitRoot(result);
          });
    if (hasTrackedResult && !hasKnownTrackedSemantics) {
      setLifetimeFailure(
          failure, LifetimeFailureKind::UnsupportedTrackedValueProducer, &op);
      return mlir::failure();
    }

    if (mlir::failed(mapAsyncDependencyResults(&op, failure)))
      return mlir::failure();
    completeAsyncTasks(&op);
    if (localCompletion &&
        mlir::failed(localCompletion->observe(&op, *this, failure)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
LifetimeDataflow::run(mlir::Operation *scope,
                      LocalCompletionTracker *localCompletion,
                      LifetimeFailure *failure) {
  if (!scope || !isTrackedType || mlir::failed(initialize(failure)))
    return mlir::failure();
  for (mlir::Region &region : scope->getRegions())
    if (mlir::failed(processRegion(region, localCompletion, failure)))
      return mlir::failure();
  if (localCompletion && mlir::failed(localCompletion->finish(scope, failure)))
    return mlir::failure();
  return finishAsyncTasks(failure);
}

void LocalCompletionTracker::processFence(ProgramPoint fencePoint,
                                          LifetimeDataflow &dataflow) {
  llvm::SmallVector<PendingIssue, 8> remainingIssues;
  for (PendingIssue issue : pendingIssues) {
    if (!issue.path.intersect(fencePoint.path)) {
      remainingIssues.push_back(issue);
      continue;
    }
    llvm::SmallVector<PathCondition, 2> remainingPaths;
    issue.path.subtract(fencePoint.path, remainingPaths);
    for (PathCondition path : remainingPaths)
      remainingIssues.push_back(PendingIssue{issue.origin, path});
  }
  pendingIssues = std::move(remainingIssues);

  llvm::SmallVector<PendingAccess, 8> remainingAccesses;
  for (PendingAccess access : pendingAccesses) {
    std::optional<PathCondition> completedPath =
        access.root.path.intersect(fencePoint.path);
    if (!completedPath) {
      remainingAccesses.push_back(access);
      continue;
    }
    dataflow.extendTo(RootRef{access.root.demandIndex, *completedPath},
                      ProgramPoint{fencePoint.event, *completedPath});

    llvm::SmallVector<PathCondition, 2> remainingPaths;
    access.root.path.subtract(fencePoint.path, remainingPaths);
    for (PathCondition path : remainingPaths)
      remainingAccesses.push_back(
          PendingAccess{access.origin, RootRef{access.root.demandIndex, path}});
  }
  pendingAccesses = std::move(remainingAccesses);
}

mlir::LogicalResult LocalCompletionTracker::observe(mlir::Operation *op,
                                                    LifetimeDataflow &dataflow,
                                                    LifetimeFailure *failure) {
  auto effectInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  std::optional<ProgramPoint> point = dataflow.timeline.lookup(op);
  if (!effectInterface || !point)
    return mlir::success();

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effects;
  effectInterface.getEffects(effects);
  bool hasFence = llvm::any_of(effects, [](const auto &effect) {
    return llvm::isa<WaferSyncResource>(effect.getResource()) &&
           llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect());
  });
  if (hasFence)
    processFence(*point, dataflow);

  bool hasLocalIssue = llvm::any_of(effects, [](const auto &effect) {
    return llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect()) &&
           llvm::isa<WaferComputeResource, WaferMovementResource>(
               effect.getResource());
  });
  if (!hasLocalIssue)
    return mlir::success();

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> trackedEffects;
  llvm::copy_if(
      effects, std::back_inserter(trackedEffects), [&](const auto &effect) {
        mlir::Value value = effect.getValue();
        return value &&
               (llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect()) ||
                llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect())) &&
               dataflow.isTrackedType &&
               dataflow.isTrackedType(value.getType());
      });
  if (trackedEffects.empty())
    return mlir::success();

  pendingIssues.push_back(PendingIssue{op, point->path});
  for (const auto &effect : trackedEffects) {
    mlir::Value value = effect.getValue();
    for (RootRef root : dataflow.rootsAt(value, point->path))
      pendingAccesses.push_back(PendingAccess{op, root});
  }
  (void)failure;
  return mlir::success();
}

mlir::LogicalResult
LocalCompletionTracker::verifyLoopBackedge(mlir::Operation *loop,
                                           LifetimeFailure *failure) const {
  for (const PendingIssue &issue : pendingIssues) {
    if (!isNestedIn(issue.origin, loop) || issue.origin == loop)
      continue;
    setLifetimeFailure(failure, LifetimeFailureKind::LoopBackedgeCompletion,
                       issue.origin);
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
LocalCompletionTracker::finish(mlir::Operation *scope,
                               LifetimeFailure *failure) const {
  if (!pendingIssues.empty()) {
    setLifetimeFailure(failure, LifetimeFailureKind::MissingLocalCompletion,
                       pendingIssues.front().origin);
    return mlir::failure();
  }
  if (!pendingAccesses.empty()) {
    setLifetimeFailure(
        failure, LifetimeFailureKind::InconsistentCompletionState,
        pendingAccesses.front().origin ? pendingAccesses.front().origin
                                       : scope);
    return mlir::failure();
  }
  return mlir::success();
}

bool lifetimesOverlap(const LifetimeDemand &lhs, const LifetimeDemand &rhs) {
  for (const LiveSegment &lhsSegment : lhs.segments) {
    for (const LiveSegment &rhsSegment : rhs.segments) {
      if (!lhsSegment.path.compatibleForPacking(rhsSegment.path))
        continue;
      if (lhsSegment.beginEvent <= rhsSegment.endEvent &&
          rhsSegment.beginEvent <= lhsSegment.endEvent)
        return true;
    }
  }
  return false;
}

std::optional<int64_t> combineAlignmentRequirements(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0)
    return std::nullopt;
  int64_t scaled = lhs / std::gcd(lhs, rhs);
  if (scaled > std::numeric_limits<int64_t>::max() / rhs)
    return std::nullopt;
  return scaled * rhs;
}

} // namespace wafer::memory_planning::detail
