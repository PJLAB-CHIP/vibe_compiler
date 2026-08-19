//===- LifetimeAnalysis.cpp - Structured memory lifetime analysis --------===//

#include "MemoryPlanning/LifetimeAnalysis.h"
#include "Wafer/Analysis/SingleExecutionRegionFlow.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

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

using analysis::getSingleExecutionRegionEntryOperand;
using analysis::getSingleExecutionRegionExitOperand;
using analysis::getSingleExecutionRegionFlow;

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

static bool canCarryStorageAlias(mlir::Value value) {
  return value &&
         mlir::isa<mlir::TensorType, mlir::BaseMemRefType>(value.getType());
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

static ProgramPoint
getGuaranteedCompletionPoint(mlir::Operation *op,
                             const StructuredTimeline &timeline,
                             ProgramPoint point) {
  for (mlir::Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (mlir::isa<mlir::scf::IfOp>(parent))
      break;
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
      if (isStaticallyNonEmpty(loop))
        if (std::optional<ProgramPoint> loopPoint = timeline.lookup(loop))
          point.path = loopPoint->path;
      break;
    }
    if (mlir::isa<TileRegionOp, mlir::func::FuncOp, mlir::async::FuncOp>(
            parent))
      break;
  }
  return point;
}

static bool hasPossibleBackedge(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getLowerBound()));
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getUpperBound()));
  std::optional<int64_t> step =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getStep()));
  if (!lower || !upper || !step || *step <= 0)
    return true;
  if (*lower >= *upper)
    return false;
  __int128 next = static_cast<__int128>(*lower) + static_cast<__int128>(*step);
  return next < static_cast<__int128>(*upper);
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

static bool isUnconditionallyNestedInStaticFor(mlir::Operation *operation,
                                               mlir::scf::ForOp outer) {
  for (mlir::Operation *parent = operation ? operation->getParentOp() : nullptr;
       parent; parent = parent->getParentOp()) {
    if (parent == outer)
      return true;
    auto nestedFor = mlir::dyn_cast<mlir::scf::ForOp>(parent);
    if (!nestedFor || !isStaticallyNonEmpty(nestedFor))
      return false;
  }
  return false;
}

static uint32_t
getNCCIssueWorkerMask(const NCCSynchronizationContract &completion) {
  if (!completion.issueWorker)
    return 0;
  uint32_t worker = static_cast<uint32_t>(*completion.issueWorker);
  if (worker >= kNCCWorkerCount)
    return 0;
  return uint32_t{1} << worker;
}

static bool hasOnlyWitnessedRootlessStorageEffects(mlir::Operation *op) {
  auto effectInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effectInterface)
    return false;

  const bool hasTypedDTEBufferContract =
      mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(op);
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effects;
  effectInterface.getEffects(effects);
  for (const mlir::MemoryEffects::EffectInstance &effect : effects) {
    if (effect.getValue())
      continue;

    mlir::SideEffects::Resource *resource = effect.getResource();
    if (resource == WaferComputeResource::get() ||
        resource == WaferMovementResource::get())
      continue;
    // Direct DTE transport occupancy is not an NCC buffer observation. The
    // send/receive buffer's value-associated SPM effect below is the exact
    // address-hazard contract, and dte_wait only completes its exact tokens.
    // Keep unknown communication-resource users fail-closed.
    if (hasTypedDTEBufferContract &&
        resource == WaferCommunicationResource::get())
      continue;

    std::optional<MemorySpace> memorySpace;
    if (resource == WaferSPMResource::get())
      memorySpace = MemorySpace::SPM;
    else if (resource == WaferDDRResource::get())
      memorySpace = MemorySpace::DDR;
    if (!memorySpace)
      return false;

    bool rootlessRead =
        llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
    bool rootlessWrite =
        llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect());
    if (!rootlessRead && !rootlessWrite)
      return false;

    bool witnessed = llvm::any_of(
        effects, [&](const mlir::MemoryEffects::EffectInstance &candidate) {
          mlir::Value value = candidate.getValue();
          auto type = value ? mlir::dyn_cast<mlir::MemRefType>(value.getType())
                            : mlir::MemRefType{};
          MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
          if (!memory || memory.getSpace() != *memorySpace)
            return false;
          return (rootlessRead && llvm::isa<mlir::MemoryEffects::Read>(
                                      candidate.getEffect())) ||
                 (rootlessWrite &&
                  llvm::isa<mlir::MemoryEffects::Write>(candidate.getEffect()));
        });
    if (!witnessed)
      return false;
  }
  return true;
}

static constexpr unsigned kUnresolvedRootIndex =
    std::numeric_limits<unsigned>::max();

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

template <typename Ref>
static bool pathsCoverQuery(PathCondition query,
                            llvm::ArrayRef<Ref> mappedRefs) {
  llvm::SmallVector<PathCondition, 2> uncovered{query};
  for (const Ref &ref : mappedRefs) {
    llvm::SmallVector<PathCondition, 2> next;
    for (PathCondition path : uncovered)
      path.subtract(ref.path, next);
    uncovered = std::move(next);
    if (uncovered.empty())
      return true;
  }
  return false;
}

static bool pathsCoverQuery(PathCondition query,
                            llvm::ArrayRef<PathCondition> mappedPaths) {
  llvm::SmallVector<PathCondition, 2> uncovered{query};
  for (const PathCondition &mappedPath : mappedPaths) {
    llvm::SmallVector<PathCondition, 2> next;
    for (PathCondition path : uncovered)
      path.subtract(mappedPath, next);
    uncovered = std::move(next);
    if (uncovered.empty())
      return true;
  }
  return false;
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
      } else if (std::optional<analysis::SingleExecutionRegionFlow> flow =
                     getSingleExecutionRegionFlow(&op)) {
        assignRegion(*flow->region, path, loopDepth);
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

StructuredTimelineAnalysis::StructuredTimelineAnalysis(mlir::Operation *scope) {
  mlir::FailureOr<StructuredTimeline> built =
      StructuredTimeline::build(scope, &failure);
  if (mlir::succeeded(built))
    timeline.emplace(std::move(*built));
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
  if (auto cached = normalizedValues.find(value);
      cached != normalizedValues.end())
    return cached->second;

  llvm::SmallVector<mlir::Value, 8> path;
  llvm::SmallDenseSet<mlir::Value, 8> seen;
  while (value && seen.insert(value).second) {
    if (auto cached = normalizedValues.find(value);
        cached != normalizedValues.end()) {
      value = cached->second;
      break;
    }
    path.push_back(value);
    mlir::Value resolved = resolveValue(value);
    if (!resolved || resolved == value)
      break;
    value = resolved;
  }
  for (mlir::Value traversed : path)
    normalizedValues.try_emplace(traversed, value);
  return value;
}

mlir::LogicalResult LifetimeDataflow::initialize(LifetimeFailure *failure) {
  normalizedValues.clear();
  valueRefs.clear();
  valueOrigins.clear();
  valueCacheRevision = 1;
  valueRefCacheRevisions.clear();
  valueOriginCacheRevisions.clear();
  valueRefCacheWrites.clear();
  valueOriginCacheWrites.clear();
  demandUseWrites.clear();
  valueRefCacheCoverage.clear();
  valueOriginCacheCoverage.clear();
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
    mlir::Value normalized = normalize(demand.allocation.getMemref());
    valueRefs[normalized] = llvm::SmallVector<RootRef, 2>{
        RootRef{static_cast<unsigned>(index), point->path}};
    markValueRefCacheCurrent(normalized);
    valueOrigins[normalized] = llvm::SmallVector<ValueOriginRef, 2>{
        ValueOriginRef{demand.allocation.getMemref(), point->path}};
    markValueOriginCacheCurrent(normalized);
  }
  return mlir::success();
}

void LifetimeDataflow::markValueRefCacheCurrent(mlir::Value value) {
  valueRefCacheRevisions[value] = valueCacheRevision;
  valueRefCacheWrites.push_back(value);
}

void LifetimeDataflow::markValueOriginCacheCurrent(mlir::Value value) {
  valueOriginCacheRevisions[value] = valueCacheRevision;
  valueOriginCacheWrites.push_back(value);
}

llvm::SmallVector<RootRef, 2>
LifetimeDataflow::rootsAt(mlir::Value value, PathCondition usePath) const {
  if (!canCarryStorageAlias(value))
    return {};
  value = normalize(value);
  llvm::SmallVector<RootRef, 2> cachedRefs;
  if (auto mapped = valueRefs.find(value); mapped != valueRefs.end()) {
    for (RootRef ref : mapped->second)
      if (std::optional<PathCondition> path = ref.path.intersect(usePath))
        appendUniqueRootRefs(cachedRefs, llvm::ArrayRef<RootRef>{
                                             RootRef{ref.demandIndex, *path}});
    if (valueRefCacheRevisions.lookup(value) == valueCacheRevision) {
      auto coverage = valueRefCacheCoverage.find(value);
      if ((coverage != valueRefCacheCoverage.end() &&
           pathsCoverQuery(
               usePath, llvm::ArrayRef<PathCondition>(coverage->second))) ||
          pathsCoverQuery(usePath,
                          llvm::ArrayRef<RootRef>(mapped->second)))
        return cachedRefs;
    }
  }
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
      // valueRefs is the forward dataflow cache for this SSA value. If its
      // path-qualified entries cover the complete query, recursively walking
      // the same loop/region alias chain cannot discover another reachable
      // root and turns long structured programs quadratic.
      if (valueRefCacheRevisions.lookup(current) == valueCacheRevision) {
        auto coverage = valueRefCacheCoverage.find(current);
        if ((coverage != valueRefCacheCoverage.end() &&
             pathsCoverQuery(
                 queryPath,
                 llvm::ArrayRef<PathCondition>(coverage->second))) ||
            pathsCoverQuery(queryPath,
                            llvm::ArrayRef<RootRef>(mapped->second)))
          return refs;
      }
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
      } else if (mlir::Value entry =
                     getSingleExecutionRegionEntryOperand(blockArg)) {
        mlir::Operation *parent = blockArg.getOwner()->getParentOp();
        if (std::optional<ProgramPoint> point = timeline.lookup(parent))
          appendAt(entry, point);
        else
          appendValue(entry, queryPath);
      }
    } else if (auto result = mlir::dyn_cast<mlir::OpResult>(current)) {
      mlir::Operation *def = result.getOwner();
      if (auto toMemref =
              mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(def)) {
        appendAt(toMemref.getTensor(), timeline.lookup(def));
      } else if (auto toTensor =
                     mlir::dyn_cast<mlir::bufferization::ToTensorOp>(def)) {
        appendAt(toTensor.getMemref(), timeline.lookup(def));
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
        if (!isStaticallyNonEmpty(forOp) && index < forOp.getInitArgs().size())
          appendAt(forOp.getInitArgs()[index], timeline.lookup(forOp));
        mlir::scf::YieldOp yield = mlir::dyn_cast<mlir::scf::YieldOp>(
            forOp.getBody()->getTerminator());
        if (yield && index < yield.getResults().size())
          appendAt(yield.getResults()[index], timeline.lookup(yield));
      } else if (mlir::Value exit =
                     getSingleExecutionRegionExitOperand(result)) {
        appendAt(exit, timeline.lookup(exit.getParentBlock()->getTerminator()));
      }
    }

    active.erase(current);
    return refs;
  };
  return collect(value, usePath);
}

llvm::SmallVector<ValueOriginRef, 2>
LifetimeDataflow::originsAt(mlir::Value value, PathCondition usePath) const {
  if (!canCarryStorageAlias(value))
    return {};
  value = normalize(value);
  llvm::SmallVector<ValueOriginRef, 2> cachedRefs;
  if (auto mapped = valueOrigins.find(value); mapped != valueOrigins.end()) {
    for (ValueOriginRef ref : mapped->second)
      if (std::optional<PathCondition> path = ref.path.intersect(usePath))
        appendUniqueOriginRefs(cachedRefs, llvm::ArrayRef<ValueOriginRef>{
                                             ValueOriginRef{ref.root, *path}});
    if (valueOriginCacheRevisions.lookup(value) == valueCacheRevision) {
      auto coverage = valueOriginCacheCoverage.find(value);
      if ((coverage != valueOriginCacheCoverage.end() &&
           pathsCoverQuery(
               usePath, llvm::ArrayRef<PathCondition>(coverage->second))) ||
          pathsCoverQuery(
              usePath, llvm::ArrayRef<ValueOriginRef>(mapped->second)))
        return cachedRefs;
    }
  }
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
      if (valueOriginCacheRevisions.lookup(current) == valueCacheRevision) {
        auto coverage = valueOriginCacheCoverage.find(current);
        if ((coverage != valueOriginCacheCoverage.end() &&
             pathsCoverQuery(
                 queryPath,
                 llvm::ArrayRef<PathCondition>(coverage->second))) ||
            pathsCoverQuery(
                queryPath,
                llvm::ArrayRef<ValueOriginRef>(mapped->second)))
          return refs;
      }
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
      } else if (mlir::Value entry =
                     getSingleExecutionRegionEntryOperand(blockArg)) {
        hasAliasSemantics = true;
        mlir::Operation *parent = blockArg.getOwner()->getParentOp();
        if (std::optional<ProgramPoint> point = timeline.lookup(parent))
          appendAt(entry, point);
        else
          appendValue(entry, queryPath);
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
        if (!isStaticallyNonEmpty(forOp) && index < forOp.getInitArgs().size())
          appendAt(forOp.getInitArgs()[index], timeline.lookup(forOp));
        mlir::scf::YieldOp yield = mlir::dyn_cast<mlir::scf::YieldOp>(
            forOp.getBody()->getTerminator());
        if (yield && index < yield.getResults().size())
          appendAt(yield.getResults()[index], timeline.lookup(yield));
      } else if (mlir::Value exit =
                     getSingleExecutionRegionExitOperand(result)) {
        std::optional<ProgramPoint> point =
            timeline.lookup(exit.getParentBlock()->getTerminator());
        hasAliasSemantics = point.has_value();
        if (point)
          appendAt(exit, point);
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

void LifetimeDataflow::recordUse(RootRef ref, int64_t event) {
  if (ref.demandIndex >= demands.size())
    return;
  LifetimeDemand &demand = demands[ref.demandIndex];
  if (event < demand.allocationPoint.event)
    return;
  demandUseWrites.push_back(ref.demandIndex);
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
  if (!viewLike || !point)
    return;
  mlir::Value source = viewLike.getViewSource();
  llvm::SmallVector<RootRef, 2> refs = rootsAt(source, point->path);
  llvm::SmallVector<ValueOriginRef, 2> origins = originsAt(source, point->path);
  for (mlir::Value result : op->getResults()) {
    mlir::Value normalized = normalize(result);
    valueRefs[normalized] = refs;
    markValueRefCacheCurrent(normalized);
    valueRefCacheCoverage[normalized] = {point->path};
    valueOrigins[normalized] = origins;
    markValueOriginCacheCurrent(normalized);
    valueOriginCacheCoverage[normalized] = {point->path};
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
    if (!canCarryStorageAlias(result))
      return mlir::success();
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
    mlir::Value normalized = normalize(result);
    valueRefs[normalized] = std::move(refs);
    markValueRefCacheCurrent(normalized);
    valueRefCacheCoverage[normalized] = {point->path};
    valueOrigins[normalized] = std::move(origins);
    markValueOriginCacheCurrent(normalized);
    valueOriginCacheCoverage[normalized] = {point->path};
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
    if (!canCarryStorageAlias(result))
      continue;
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
    mlir::Value normalized = normalize(result);
    valueRefs[normalized] = std::move(roots);
    markValueRefCacheCurrent(normalized);
    valueRefCacheCoverage[normalized] = {point->path};
    valueOrigins[normalized] = std::move(origins);
    markValueOriginCacheCurrent(normalized);
    valueOriginCacheCoverage[normalized] = {point->path};
  }
  return mlir::success();
}

mlir::LogicalResult
LifetimeDataflow::mapAsyncDependencyResults(mlir::Operation *op,
                                            LifetimeFailure *failure) {
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!point)
    return mlir::success();
  if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp>(op) ||
      mlir::isa<mlir::SelectLikeOpInterface>(op) ||
      getSingleExecutionRegionFlow(op).has_value())
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
  ProgramPoint completionPoint =
      getGuaranteedCompletionPoint(op, timeline, *point);

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
    for (AsyncTaskRef completed : asyncTasksAt(handle, completionPoint.path)) {
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
  std::optional<ProgramPoint> ifPoint = timeline.lookup(op);
  mlir::scf::YieldOp thenYield = getSingleBlockYield(ifOp.getThenRegion());
  mlir::scf::YieldOp elseYield = getSingleBlockYield(ifOp.getElseRegion());
  if (!thenYield || !elseYield)
    return;

  for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
    if (!hasAsyncDependencyType(result)) {
      if (!canCarryStorageAlias(result))
        continue;
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
      mlir::Value normalized = normalize(result);
      valueRefs[normalized] = std::move(refs);
      markValueRefCacheCurrent(normalized);
      if (ifPoint)
        valueRefCacheCoverage[normalized] = {ifPoint->path};
      valueOrigins[normalized] = std::move(origins);
      markValueOriginCacheCurrent(normalized);
      if (ifPoint)
        valueOriginCacheCoverage[normalized] = {ifPoint->path};
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
    if (!canCarryStorageAlias(init))
      continue;
    llvm::SmallVector<RootRef, 2> refs = rootsAt(init, point->path);
    mlir::Value normalized = normalize(iterArg);
    valueRefs[normalized] = std::move(refs);
    markValueRefCacheCurrent(normalized);
    valueRefCacheCoverage[normalized] = {point->path};
    llvm::SmallVector<ValueOriginRef, 2> origins = originsAt(init, point->path);
    valueOrigins[normalized] = std::move(origins);
    markValueOriginCacheCurrent(normalized);
    valueOriginCacheCoverage[normalized] = {point->path};
  }
}

void LifetimeDataflow::mapSingleExecutionRegionBlockArgs(
    mlir::Operation *op, const analysis::SingleExecutionRegionFlow &flow) {
  std::optional<ProgramPoint> point = timeline.lookup(op);
  if (!point)
    return;

  for (auto [input, blockArg] :
       llvm::zip_equal(flow.entryOperands, flow.entryArguments)) {
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
    if (!canCarryStorageAlias(input))
      continue;

    llvm::SmallVector<RootRef, 2> roots = rootsAt(input, point->path);
    mlir::Value normalized = normalize(input);
    normalizedValues[blockArg] = normalized;
    valueRefs[normalized] = std::move(roots);
    markValueRefCacheCurrent(normalized);
    valueRefCacheCoverage[normalized] = {point->path};
    llvm::SmallVector<ValueOriginRef, 2> origins =
        originsAt(input, point->path);
    valueOrigins[normalized] = std::move(origins);
    markValueOriginCacheCurrent(normalized);
    valueOriginCacheCoverage[normalized] = {point->path};
  }
}

void LifetimeDataflow::mapSingleExecutionRegionResults(
    mlir::Operation *op, const analysis::SingleExecutionRegionFlow &flow) {
  std::optional<ProgramPoint> point =
      timeline.lookup(flow.region->front().getTerminator());
  if (!point)
    return;

  for (auto [yielded, result] :
       llvm::zip_equal(flow.exitOperands, flow.results)) {
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
    if (!canCarryStorageAlias(result))
      continue;

    llvm::SmallVector<RootRef, 2> roots = rootsAt(yielded, point->path);
    mlir::Value normalized = normalize(yielded);
    normalizedValues[result] = normalized;
    valueRefs[normalized] = std::move(roots);
    markValueRefCacheCurrent(normalized);
    valueRefCacheCoverage[normalized] = {point->path};
    llvm::SmallVector<ValueOriginRef, 2> origins =
        originsAt(yielded, point->path);
    valueOrigins[normalized] = std::move(origins);
    markValueOriginCacheCurrent(normalized);
    valueOriginCacheCoverage[normalized] = {point->path};
  }
}

mlir::LogicalResult
LifetimeDataflow::mapForResultsAndBackedge(mlir::Operation *op,
                                           size_t refWriteBegin,
                                           size_t originWriteBegin,
                                           size_t demandUseBegin,
                                           LifetimeFailure *failure) {
  auto forOp = mlir::cast<mlir::scf::ForOp>(op);
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
  std::optional<ProgramPoint> loopPoint = timeline.lookup(op);
  std::optional<int64_t> loopEnd = timeline.lookupSubtreeEnd(op);
  if (!yield || !loopPoint || !loopEnd)
    return mlir::success();

  // The loop body was visited using only the initial recurrence mapping.
  // Recompute only loop-local mappings written during that visit. Earlier
  // function and TileRegion mappings are unrelated to this backedge.
  invalidateLoopLocalValueCaches(op, refWriteBegin, originWriteBegin);

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

      bool carriesLoopLocalTask =
          llvm::any_of(backedgeTasks, [&](AsyncTaskRef ref) {
            return ref.taskIndex < asyncTasks.size() &&
                   isNestedIn(asyncTasks[ref.taskIndex].origin, op);
          });
      if (carriesLoopLocalTask && !isStaticallyNonEmpty(forOp)) {
        setLifetimeFailure(
            failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
        return mlir::failure();
      }
      if (!isStaticallyNonEmpty(forOp)) {
        auto hasSameTaskIdentities = [](llvm::ArrayRef<AsyncTaskRef> lhs,
                                        llvm::ArrayRef<AsyncTaskRef> rhs) {
          return llvm::all_of(lhs, [&](AsyncTaskRef ref) {
            return llvm::any_of(rhs, [&](AsyncTaskRef other) {
              return ref.taskIndex == other.taskIndex;
            });
          });
        };
        if (!hasSameTaskIdentities(initialTasks, backedgeTasks) ||
            !hasSameTaskIdentities(backedgeTasks, initialTasks)) {
          setLifetimeFailure(
              failure, LifetimeFailureKind::UnsupportedAsyncCompletionFlow, op);
          return mlir::failure();
        }
      }

      // A loop-local branch can select differently on the next dynamic
      // iteration. Once a completion handle crosses the backedge, those
      // repeatable decisions no longer constrain which root it may refer to.
      llvm::SmallVector<RootRef, 4> relaxedRefs;
      for (RootRef ref : refs) {
        ref.path = loopPoint->path;
        appendUniqueRootRefs(relaxedRefs, llvm::ArrayRef<RootRef>{ref});
      }
      refs.assign(relaxedRefs.begin(), relaxedRefs.end());
      for (RootRef ref : refs)
        if (ref.demandIndex < demands.size())
          demands[ref.demandIndex].segments.push_back(
              LiveSegment{loopPoint->event, *loopEnd, ref.path});
      if (!refs.empty()) {
        asyncRefs[result] = refs;
        if (index < forOp.getRegionIterArgs().size())
          asyncRefs[forOp.getRegionIterArgs()[index]] = std::move(refs);
      }
      llvm::SmallVector<AsyncTaskRef, 2> recurrenceTasks = initialTasks;
      appendUniqueAsyncTaskRefs(recurrenceTasks, backedgeTasks);
      llvm::SmallVector<AsyncTaskRef, 2> resultTasks =
          isStaticallyNonEmpty(forOp) ? backedgeTasks : recurrenceTasks;
      auto forgetTaskRepeatableDecisions =
          [&](llvm::SmallVectorImpl<AsyncTaskRef> &tasks) {
            llvm::SmallVector<AsyncTaskRef, 4> relaxed;
            for (AsyncTaskRef task : tasks) {
              task.path = loopPoint->path;
              appendUniqueAsyncTaskRefs(relaxed,
                                        llvm::ArrayRef<AsyncTaskRef>{task});
            }
            tasks.assign(relaxed.begin(), relaxed.end());
          };
      forgetTaskRepeatableDecisions(recurrenceTasks);
      forgetTaskRepeatableDecisions(resultTasks);
      if (!recurrenceTasks.empty() && index < forOp.getRegionIterArgs().size())
        asyncTaskRefs[forOp.getRegionIterArgs()[index]] = recurrenceTasks;
      if (!resultTasks.empty())
        asyncTaskRefs[result] = std::move(resultTasks);
      continue;
    }
    if (!canCarryStorageAlias(result))
      continue;
    llvm::SmallVector<RootRef, 2> initialRefs;
    llvm::SmallVector<ValueOriginRef, 2> initialOrigins;
    if (index < forOp.getInitArgs().size()) {
      initialRefs = rootsAt(forOp.getInitArgs()[index], loopPoint->path);
      initialOrigins = originsAt(forOp.getInitArgs()[index], loopPoint->path);
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
      backedgeOrigins = originsAt(yield.getResults()[index], yieldPoint->path);
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
    // forget repeatable decisions before storing the fixed-point union.
    forgetRepeatableDecisions(recurrenceRefs);
    forgetRepeatableDecisions(recurrenceOrigins);
    forgetRepeatableDecisions(resultRefs);
    forgetRepeatableDecisions(resultOrigins);

    for (RootRef ref : recurrenceRefs)
      if (ref.demandIndex < demands.size())
        demands[ref.demandIndex].segments.push_back(
            LiveSegment{loopPoint->event, *loopEnd, ref.path});
    if (index < forOp.getRegionIterArgs().size()) {
      mlir::Value iterArg = normalize(forOp.getRegionIterArgs()[index]);
      valueRefs[iterArg] = recurrenceRefs;
      markValueRefCacheCurrent(iterArg);
      valueRefCacheCoverage[iterArg] = {loopPoint->path};
    }
    mlir::Value normalizedResult = normalize(result);
    valueRefs[normalizedResult] = std::move(resultRefs);
    markValueRefCacheCurrent(normalizedResult);
    valueRefCacheCoverage[normalizedResult] = {loopPoint->path};
    if (index < forOp.getRegionIterArgs().size()) {
      mlir::Value iterArg = normalize(forOp.getRegionIterArgs()[index]);
      valueOrigins[iterArg] = recurrenceOrigins;
      markValueOriginCacheCurrent(iterArg);
      valueOriginCacheCoverage[iterArg] = {loopPoint->path};
    }
    valueOrigins[normalizedResult] = std::move(resultOrigins);
    markValueOriginCacheCurrent(normalizedResult);
    valueOriginCacheCoverage[normalizedResult] = {loopPoint->path};
  }

  // A root defined outside the loop and referenced from its body may be read
  // again on every dynamic iteration. Static preorder alone would otherwise
  // let a later body-local allocation overwrite that root before the next
  // backedge. Conservatively keep every such captured root live across the
  // complete loop subtree. Body-local roots are recreated each iteration and
  // remain governed by the backedge completion proof; loop-carried ones were
  // handled above (including the dynamic-allocation rejection).
  assert(demandUseBegin <= demandUseWrites.size());
  llvm::DenseSet<unsigned> capturedDemands;
  for (size_t index = demandUseBegin; index < demandUseWrites.size(); ++index) {
    unsigned demandIndex = demandUseWrites[index];
    if (demandIndex < demands.size() &&
        !isNestedIn(demands[demandIndex].allocation.getOperation(), op))
      capturedDemands.insert(demandIndex);
  }
  for (unsigned demandIndex : capturedDemands)
    recordUse(RootRef{demandIndex, loopPoint->path}, *loopEnd);
  return mlir::success();
}

void LifetimeDataflow::invalidateLoopLocalValueCaches(
    mlir::Operation *loop, size_t refWriteBegin, size_t originWriteBegin) {
  auto isDefinedInsideLoop = [&](mlir::Value value) {
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
      return isNestedIn(result.getOwner(), loop);
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    mlir::Operation *owner =
        argument && argument.getOwner() ? argument.getOwner()->getParentOp()
                                        : nullptr;
    return isNestedIn(owner, loop);
  };

  assert(refWriteBegin <= valueRefCacheWrites.size());
  for (size_t index = refWriteBegin; index < valueRefCacheWrites.size();
       ++index) {
    mlir::Value value = valueRefCacheWrites[index];
    if (isDefinedInsideLoop(value))
      valueRefCacheRevisions.erase(value);
  }
  valueRefCacheWrites.resize(refWriteBegin);

  assert(originWriteBegin <= valueOriginCacheWrites.size());
  for (size_t index = originWriteBegin; index < valueOriginCacheWrites.size();
       ++index) {
    mlir::Value value = valueOriginCacheWrites[index];
    if (isDefinedInsideLoop(value))
      valueOriginCacheRevisions.erase(value);
  }
  valueOriginCacheWrites.resize(originWriteBegin);
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
    std::optional<analysis::SingleExecutionRegionFlow> singleExecutionFlow =
        getSingleExecutionRegionFlow(&op);
    {
      wafer::support::ScopedCompileTimingSpan timing(
          "analysis-algorithm", "lifetime-dataflow", "record-operands");
      recordOperands(&op);
    }

    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      const size_t refWriteBegin = valueRefCacheWrites.size();
      const size_t originWriteBegin = valueOriginCacheWrites.size();
      const size_t demandUseBegin = demandUseWrites.size();
      mapForRegionIterArgs(&op);
      if (mlir::failed(
              processRegion(forOp.getRegion(), localCompletion, failure)))
        return mlir::failure();
      if (localCompletion) {
        wafer::support::ScopedCompileTimingSpan timing(
            "analysis-algorithm", "lifetime-dataflow", "verify-loop-backedge");
        if (mlir::failed(
                localCompletion->verifyLoopBackedge(&op, *this, failure)))
          return mlir::failure();
      }
      {
        wafer::support::ScopedCompileTimingSpan timing(
            "analysis-algorithm", "lifetime-dataflow", "map-loop-results");
        if (mlir::failed(mapForResultsAndBackedge(
                &op, refWriteBegin, originWriteBegin, demandUseBegin,
                failure)))
          return mlir::failure();
      }
    } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      if (mlir::failed(
              processRegion(ifOp.getThenRegion(), localCompletion, failure)) ||
          mlir::failed(
              processRegion(ifOp.getElseRegion(), localCompletion, failure)))
        return mlir::failure();
      mapIfResults(&op);
    } else if (singleExecutionFlow) {
      mapSingleExecutionRegionBlockArgs(&op, *singleExecutionFlow);
      if (mlir::failed(processRegion(*singleExecutionFlow->region,
                                     localCompletion, failure)))
        return mlir::failure();
      mapSingleExecutionRegionResults(&op, *singleExecutionFlow);
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
            mlir::scf::IfOp, mlir::scf::ForOp>(op) ||
        mlir::isa<mlir::ViewLikeOpInterface, mlir::SelectLikeOpInterface,
                  mlir::MemoryEffectOpInterface>(op) ||
        singleExecutionFlow.has_value() ||
        op.hasTrait<mlir::OpTrait::IsTerminator>();
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
      hasSupportedTrackedUse = hasSupportedTrackedUse ||
                               isSupportedDirectAliasCall(call, isTrackedType);

    // Known memory/control-flow operations already have an explicit lifetime
    // rule above. Only an unsupported consumer (or raw address exposure) needs
    // the comparatively expensive origin query that decides whether it
    // actually touches tracked storage.
    bool hasTrackedOperand = false;
    if (point && (exposesRawMetadata || !hasSupportedTrackedUse)) {
      wafer::support::ScopedCompileTimingSpan timing(
          "analysis-algorithm", "lifetime-dataflow",
          "classify-unsupported-operands");
      hasTrackedOperand =
          llvm::any_of(op.getOperands(), [&](mlir::Value operand) {
            return !hasAsyncDependencyType(operand) &&
                   !originsAt(operand, point->path).empty();
          });
    }

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
    if (hasTrackedOperand) {
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
        mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp,
                  mlir::bufferization::ToMemrefOp>(op) ||
        mlir::isa<mlir::ViewLikeOpInterface, mlir::SelectLikeOpInterface>(op) ||
        singleExecutionFlow.has_value();
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
    if (localCompletion) {
      wafer::support::ScopedCompileTimingSpan timing(
          "analysis-algorithm", "lifetime-dataflow", "local-completion");
      if (mlir::failed(localCompletion->observe(&op, *this, failure)))
        return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult
LifetimeDataflow::run(mlir::Operation *scope,
                      LocalCompletionTracker *localCompletion,
                      LifetimeFailure *failure) {
  if (!scope || !isTrackedType || mlir::failed(initialize(failure)))
    return mlir::failure();
  for (mlir::Region &region : scope->getRegions()) {
    for (mlir::Block &block : region) {
      for (mlir::BlockArgument argument : block.getArguments()) {
        if (!canCarryStorageAlias(argument) ||
            !isTrackedType(argument.getType()))
          continue;
        valueRefs[argument] = {};
        markValueRefCacheCurrent(argument);
        valueRefCacheCoverage[argument] = {PathCondition::root()};
        valueOrigins[argument] = {ValueOriginRef{argument,
                                                 PathCondition::root()}};
        markValueOriginCacheCurrent(argument);
        valueOriginCacheCoverage[argument] = {PathCondition::root()};
      }
    }
  }
  for (mlir::Region &region : scope->getRegions())
    if (mlir::failed(processRegion(region, localCompletion, failure)))
      return mlir::failure();
  if (localCompletion && mlir::failed(localCompletion->finish(scope, failure)))
    return mlir::failure();
  return finishAsyncTasks(failure);
}

LocalCompletionTracker::AccessCollection
LocalCompletionTracker::collectAccesses(mlir::Operation *op, ProgramPoint point,
                                        uint32_t workerMask,
                                        LifetimeDataflow &dataflow) const {
  AccessCollection collected;
  // Region bodies are visited at their own program points. Treating the
  // containing control-flow op's recursive effects as another observer would
  // reject a branch whose every path already completes the pending issue.
  if (op->getNumRegions() != 0)
    return collected;

  auto effectInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effectInterface) {
    collected.hasUnknownObserverEffect = !mlir::isMemoryEffectFree(op);
    return collected;
  }

  // Rootless storage effects are admissible only when a value-associated
  // effect witnesses the same Wafer memory domain and direction. Otherwise a
  // pending NCC access has no address proof against this observer.
  collected.hasUnknownObserverEffect =
      !hasOnlyWitnessedRootlessStorageEffects(op);

  // Keep only identity-preserving container edges transparent. View-like
  // results intentionally retain their own identity: two views of one root
  // are not an exact range proof merely because they share an allocation.
  llvm::DenseSet<mlir::Value> activeIdentities;
  std::function<mlir::Value(mlir::Value)> resolveIdentity =
      [&](mlir::Value value) -> mlir::Value {
    value = dataflow.normalize(value);
    if (!value || !activeIdentities.insert(value).second)
      return {};
    auto finish = [&](mlir::Value identity) {
      activeIdentities.erase(value);
      return identity;
    };
    auto resolveSame = [&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
      mlir::Value lhsIdentity = resolveIdentity(lhs);
      mlir::Value rhsIdentity = resolveIdentity(rhs);
      if (!lhsIdentity || lhsIdentity != rhsIdentity)
        return {};
      return lhsIdentity;
    };

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArg.getOwner();
      mlir::Operation *parent = owner ? owner->getParentOp() : nullptr;
      if (mlir::Value entry = getSingleExecutionRegionEntryOperand(blockArg))
        return finish(resolveIdentity(entry));
      if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
        if (owner == forOp.getBody() && blockArg.getArgNumber() > 0) {
          unsigned index = blockArg.getArgNumber() - 1;
          auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
              forOp.getBody()->getTerminator());
          if (index < forOp.getInitArgs().size() && yield &&
              index < yield.getResults().size()) {
            mlir::Value initialIdentity =
                resolveIdentity(forOp.getInitArgs()[index]);
            mlir::Value yielded = yield.getResults()[index];
            mlir::Value yieldedIdentity = yielded == blockArg
                                              ? initialIdentity
                                              : resolveIdentity(yielded);
            if (initialIdentity && initialIdentity == yieldedIdentity)
              return finish(initialIdentity);
            return finish({});
          }
        }
      }
      return finish(value);
    }

    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return finish(value);
    mlir::Operation *def = result.getOwner();
    if (auto toMemref = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(def))
      return finish(resolveIdentity(toMemref.getTensor()));
    if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(def))
      return finish(resolveIdentity(toTensor.getMemref()));
    if (auto select = mlir::dyn_cast<mlir::SelectLikeOpInterface>(def))
      return finish(resolveSame(select.getTrueValue(), select.getFalseValue()));
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
      unsigned index = result.getResultNumber();
      mlir::scf::YieldOp thenYield = getSingleBlockYield(ifOp.getThenRegion());
      mlir::scf::YieldOp elseYield = getSingleBlockYield(ifOp.getElseRegion());
      if (!thenYield || !elseYield || index >= thenYield.getResults().size() ||
          index >= elseYield.getResults().size())
        return finish({});
      return finish(resolveSame(thenYield.getResults()[index],
                                elseYield.getResults()[index]));
    }
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
      unsigned index = result.getResultNumber();
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
      if (index >= forOp.getInitArgs().size() || !yield ||
          index >= yield.getResults().size())
        return finish({});
      mlir::Value initialIdentity = resolveIdentity(forOp.getInitArgs()[index]);
      mlir::Value yielded = yield.getResults()[index];
      mlir::Value iterArg = index < forOp.getRegionIterArgs().size()
                                ? forOp.getRegionIterArgs()[index]
                                : mlir::Value{};
      mlir::Value yieldedIdentity =
          yielded == iterArg ? initialIdentity : resolveIdentity(yielded);
      if (initialIdentity && initialIdentity == yieldedIdentity)
        return finish(initialIdentity);
      return finish({});
    }
    if (mlir::Value exit = getSingleExecutionRegionExitOperand(result))
      return finish(resolveIdentity(exit));
    return finish(value);
  };

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effects;
  effectInterface.getEffects(effects);
  for (const mlir::MemoryEffects::EffectInstance &effect : effects) {
    mlir::Value value = effect.getValue();
    bool reads = llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
    bool writes =
        llvm::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Free>(
            effect.getEffect());
    if (value && !reads && !writes &&
        !llvm::isa<mlir::MemoryEffects::Allocate>(effect.getEffect()))
      collected.hasUnknownObserverEffect = true;
    if (!value || (!reads && !writes) || !dataflow.isTrackedType ||
        !dataflow.isTrackedType(value.getType()))
      continue;

    collected.hasTrackedEffect = true;
    collected.hasWrite |= writes;
    mlir::Value identity = resolveIdentity(value);
    llvm::SmallVector<RootRef, 2> roots = dataflow.rootsAt(value, point.path);
    llvm::SmallVector<ValueOriginRef, 2> origins =
        dataflow.originsAt(value, point.path);
    // A canonical slot permutation intentionally gives one iter_arg a finite
    // union of loop-external allocation roots. That is still a resolved
    // hardware address domain: the issued runtime range selects one root, and
    // the same-worker busytable orders an actual overlap. Requiring one stable
    // SSA identity would turn every real multi-buffer backedge into a heavy
    // join. Rootless/unresolved effects and ambiguous origin-only values
    // remain fail-closed.
    bool hasResolvedRoots =
        !roots.empty() && llvm::all_of(roots, [](RootRef root) {
          return root.demandIndex != kUnresolvedRootIndex;
        });
    // A value-associated memref origin is a resolved runtime address domain
    // even when it is caller-owned or may alias another origin. The same-worker
    // busytable compares the actual issued ranges: overlap is ordered and
    // disjoint ranges need no edge. Empty origins and rootless resource effects
    // remain fail-closed.
    bool hasResolvedAddressOrigins = roots.empty() && !origins.empty();
    if (!hasResolvedRoots && !hasResolvedAddressOrigins)
      collected.allResolved = false;
    if (roots.empty()) {
      if (origins.empty()) {
        collected.accesses.push_back(PendingAccess{
            op, RootRef{kUnresolvedRootIndex, point.path}, workerMask,
            /*logicalRoot=*/{}, identity, writes});
        continue;
      }
      for (ValueOriginRef origin : origins)
        collected.accesses.push_back(
            PendingAccess{op, RootRef{kUnresolvedRootIndex, origin.path},
                          workerMask, origin.root, identity, writes});
      continue;
    }
    mlir::Value logicalRoot =
        origins.size() == 1 ? origins.front().root : mlir::Value{};
    for (RootRef root : roots)
      collected.accesses.push_back(
          PendingAccess{op, root, workerMask, logicalRoot, identity, writes});
  }
  return collected;
}

mlir::LogicalResult LocalCompletionTracker::verifyPendingObservers(
    mlir::Operation *op, ProgramPoint point,
    const NCCSynchronizationContract &contract, const AccessCollection &current,
    LifetimeFailure *failure) const {
  auto reachablePending =
      llvm::find_if(pendingAccesses, [&](const PendingAccess &pending) {
        return pending.root.path.intersect(point.path).has_value();
      });
  if (reachablePending != pendingAccesses.end() &&
      current.hasUnknownObserverEffect) {
    setLifetimeFailure(failure, LifetimeFailureKind::MissingLocalCompletion,
                       reachablePending->origin ? reachablePending->origin
                                                : op);
    return mlir::failure();
  }
  if (!current.hasTrackedEffect)
    return mlir::success();

  bool currentAllHaveResolvedRoots =
      llvm::all_of(current.accesses, [](const PendingAccess &access) {
        return access.root.demandIndex != kUnresolvedRootIndex;
      });
  bool currentAllHaveLogicalRoots =
      llvm::all_of(current.accesses, [](const PendingAccess &access) {
        return static_cast<bool>(access.logicalRoot);
      });
  // A same-worker ordered issue with a stable address domain is safe for every
  // reachable pending/current pair: the hardware busytable orders an actual
  // overlap, while disjoint runtime ranges need no edge. Preserve the exact
  // pairwise path below for mixed workers or unresolved address domains, but
  // avoid rescanning a long linear instruction chain when all pairs satisfy
  // the same proof.
  if (contract.behavior ==
          NCCSynchronizationBehavior::OrderedAsynchronousIssue &&
      commonPendingWorkerMask != 0 && pendingWorkerMasksAgree &&
      commonPendingWorkerMask == getNCCIssueWorkerMask(contract) &&
      ((pendingAllHaveResolvedRoots && currentAllHaveResolvedRoots) ||
       (pendingAllHaveLogicalRoots && currentAllHaveLogicalRoots)))
    return mlir::success();

  for (const PendingAccess &pending : pendingAccesses) {
    for (const PendingAccess &access : current.accesses) {
      if (!pending.root.path.intersect(access.root.path) ||
          (!pending.write && !access.write))
        continue;

      bool pendingRootResolved =
          pending.root.demandIndex != kUnresolvedRootIndex;
      bool currentRootResolved =
          access.root.demandIndex != kUnresolvedRootIndex;
      if (pendingRootResolved && currentRootResolved &&
          pending.root.demandIndex != access.root.demandIndex)
        continue;
      if ((!pendingRootResolved || !currentRootResolved) &&
          pending.logicalRoot && access.logicalRoot &&
          pending.logicalRoot != access.logicalRoot &&
          mlir::isa_and_nonnull<mlir::memref::AllocOp>(
              pending.logicalRoot.getDefiningOp()) &&
          mlir::isa_and_nonnull<mlir::memref::AllocOp>(
              access.logicalRoot.getDefiningOp()))
        continue;

      bool exactRange = pending.accessIdentity && access.accessIdentity &&
                        pending.accessIdentity == access.accessIdentity;
      // The target busytable compares the issued physical ranges. For two
      // value-associated accesses in one worker, a possible overlap is
      // therefore sufficient: an actual overlap creates RAW/WAR/WAW order,
      // while disjoint runtime ranges need no dependency. Resource-only or
      // otherwise rootless effects remain fail-closed.
      bool hasKnownAddressDomain =
          exactRange || (pendingRootResolved && currentRootResolved) ||
          (pending.logicalRoot && access.logicalRoot);
      bool sameWorkerOrdered =
          hasKnownAddressDomain &&
          contract.behavior ==
              NCCSynchronizationBehavior::OrderedAsynchronousIssue &&
          pending.workerMask != 0 && pending.workerMask == access.workerMask;
      if (sameWorkerOrdered)
        continue;

      setLifetimeFailure(failure, LifetimeFailureKind::MissingLocalCompletion,
                         pending.origin ? pending.origin : op);
      return mlir::failure();
    }
  }
  return mlir::success();
}

void LocalCompletionTracker::appendPendingAccess(PendingAccess access) {
  if (pendingAccesses.empty())
    commonPendingWorkerMask = access.workerMask;
  else
    pendingWorkerMasksAgree &= commonPendingWorkerMask == access.workerMask;
  pendingAllHaveResolvedRoots &=
      access.root.demandIndex != kUnresolvedRootIndex;
  pendingAllHaveLogicalRoots &= static_cast<bool>(access.logicalRoot);
  pendingAccesses.push_back(std::move(access));
}

void LocalCompletionTracker::refreshPendingAccessSummary() {
  commonPendingWorkerMask =
      pendingAccesses.empty() ? 0 : pendingAccesses.front().workerMask;
  pendingWorkerMasksAgree = true;
  pendingAllHaveResolvedRoots = true;
  pendingAllHaveLogicalRoots = true;
  for (const PendingAccess &access : pendingAccesses) {
    pendingWorkerMasksAgree &= commonPendingWorkerMask == access.workerMask;
    pendingAllHaveResolvedRoots &=
        access.root.demandIndex != kUnresolvedRootIndex;
    pendingAllHaveLogicalRoots &= static_cast<bool>(access.logicalRoot);
  }
}

void LocalCompletionTracker::processFence(ProgramPoint fencePoint,
                                          uint32_t participantMask,
                                          LifetimeDataflow &dataflow) {
  llvm::SmallVector<PendingIssue, 8> remainingIssues;
  for (PendingIssue issue : pendingIssues) {
    if ((issue.workerMask & participantMask) == 0 ||
        !issue.path.intersect(fencePoint.path)) {
      remainingIssues.push_back(issue);
      continue;
    }
    llvm::SmallVector<PathCondition, 2> remainingPaths;
    issue.path.subtract(fencePoint.path, remainingPaths);
    for (PathCondition path : remainingPaths)
      remainingIssues.push_back(
          PendingIssue{issue.origin, path, issue.workerMask,
                       issue.accessOrderResolved, issue.hasWrite});
  }
  pendingIssues = std::move(remainingIssues);

  llvm::SmallVector<PendingAccess, 8> remainingAccesses;
  for (PendingAccess access : pendingAccesses) {
    if ((access.workerMask & participantMask) == 0) {
      remainingAccesses.push_back(access);
      continue;
    }
    std::optional<PathCondition> completedPath =
        access.root.path.intersect(fencePoint.path);
    if (!completedPath) {
      remainingAccesses.push_back(access);
      continue;
    }
    if (access.root.demandIndex != kUnresolvedRootIndex)
      dataflow.extendTo(RootRef{access.root.demandIndex, *completedPath},
                        ProgramPoint{fencePoint.event, *completedPath});

    llvm::SmallVector<PathCondition, 2> remainingPaths;
    access.root.path.subtract(fencePoint.path, remainingPaths);
    for (PathCondition path : remainingPaths)
      remainingAccesses.push_back(
          PendingAccess{access.origin, RootRef{access.root.demandIndex, path},
                        access.workerMask, access.logicalRoot,
                        access.accessIdentity, access.write});
  }
  pendingAccesses = std::move(remainingAccesses);
  refreshPendingAccessSummary();
}

mlir::LogicalResult LocalCompletionTracker::observe(mlir::Operation *op,
                                                    LifetimeDataflow &dataflow,
                                                    LifetimeFailure *failure) {
  std::optional<ProgramPoint> point = dataflow.timeline.lookup(op);
  if (!point)
    return mlir::success();

  NCCSynchronizationContract contract = getNCCSynchronizationContract(op);
  if (contract.behavior == NCCSynchronizationBehavior::ParticipantJoin) {
    ProgramPoint completionPoint =
        getGuaranteedCompletionPoint(op, dataflow.timeline, *point);
    processFence(completionPoint, contract.participantMask, dataflow);
    // A typed participant join only drains the named worker domains. It has no
    // independent memory observation, so other workers may remain pending
    // across it. Treating the join itself as an unknown observer would
    // incorrectly require every partial join to serialize all NCC workers.
    return mlir::success();
  }
  if (contract.behavior == NCCSynchronizationBehavior::SynchronousWriteback) {
    ProgramPoint completionPoint =
        getGuaranteedCompletionPoint(op, dataflow.timeline, *point);
    processFence(completionPoint, contract.participantMask, dataflow);
  }

  uint32_t workerMask = getNCCIssueWorkerMask(contract);
  AccessCollection current = collectAccesses(op, *point, workerMask, dataflow);
  if (mlir::failed(
          verifyPendingObservers(op, *point, contract, current, failure)))
    return mlir::failure();
  if (contract.behavior !=
          NCCSynchronizationBehavior::OrderedAsynchronousIssue ||
      !current.hasTrackedEffect)
    return mlir::success();

  pendingIssues.push_back(PendingIssue{
      op, point->path, workerMask,
      current.allResolved && !current.accesses.empty(), current.hasWrite});
  for (PendingAccess access : current.accesses)
    appendPendingAccess(std::move(access));
  return mlir::success();
}

bool LocalCompletionTracker::provesLoopBackedgeOrder(
    const PendingIssue &issue, mlir::Operation *loop,
    LifetimeDataflow &dataflow) const {
  auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(loop);
  if (!forOp || issue.workerMask == 0 || !issue.accessOrderResolved)
    return false;

  std::optional<ProgramPoint> bodyPoint =
      dataflow.timeline.lookup(forOp.getBody()->getTerminator());
  if (!bodyPoint || !issue.path.implies(bodyPoint->path))
    return false;

  llvm::SmallVector<const PendingAccess *, 4> issueAccesses;
  for (const PendingAccess &access : pendingAccesses) {
    if (access.origin != issue.origin ||
        !access.root.path.intersect(issue.path))
      continue;
    issueAccesses.push_back(&access);
  }
  if (issueAccesses.empty())
    return false;

  auto accessesMayConflict = [](const PendingAccess &pending,
                                const PendingAccess &access) {
    if (!pending.root.path.intersect(access.root.path) ||
        (!pending.write && !access.write))
      return false;

    bool pendingRootResolved = pending.root.demandIndex != kUnresolvedRootIndex;
    bool currentRootResolved = access.root.demandIndex != kUnresolvedRootIndex;
    if (pendingRootResolved && currentRootResolved &&
        pending.root.demandIndex != access.root.demandIndex)
      return false;
    if ((!pendingRootResolved || !currentRootResolved) && pending.logicalRoot &&
        access.logicalRoot && pending.logicalRoot != access.logicalRoot &&
        mlir::isa_and_nonnull<mlir::memref::AllocOp>(
            pending.logicalRoot.getDefiningOp()) &&
        mlir::isa_and_nonnull<mlir::memref::AllocOp>(
            access.logicalRoot.getDefiningOp()))
      return false;
    return true;
  };

  // A structured loop whose complete effectful stream consists only of
  // value-associated, resolved NCC issues is ordered directly by the hardware
  // busytable across every dynamic backedge. Dependency-related accesses must
  // stay on the issue's worker; typed streams on provably disjoint allocation
  // roots may use another worker without a completion edge. Canonical
  // rotating-buffer recurrence changes the SSA identity and may produce
  // a finite union of external allocation origins; actual overlapping issued
  // ranges order, while disjoint slots require no dependency. Statically
  // non-empty nested scf.for bodies are part of that same stream. Any observer,
  // unknown effect, unwitnessed rootless storage, conditional/possibly-empty
  // region or cross-worker conflict rejects this whole-stream proof.
  std::function<bool(mlir::Block &)> provesStructuredOrderedStream;
  provesStructuredOrderedStream = [&](mlir::Block &block) {
    for (mlir::Operation &candidate : block.without_terminator()) {
      if (auto nestedFor = mlir::dyn_cast<mlir::scf::ForOp>(candidate)) {
        if (!isStaticallyNonEmpty(nestedFor) ||
            !provesStructuredOrderedStream(*nestedFor.getBody()))
          return false;
        continue;
      }
      if (candidate.getNumRegions() != 0)
        return false;

      std::optional<ProgramPoint> candidatePoint =
          dataflow.timeline.lookup(&candidate);
      if (!candidatePoint)
        return mlir::isMemoryEffectFree(&candidate);
      if (mlir::isMemoryEffectFree(&candidate))
        continue;
      if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(candidate)) {
        // Allocation establishes ownership; it does not observe an earlier NCC
        // access. A placement invocation must own the tracked demand; the
        // separate completion-only precheck intentionally has no demands and
        // relies on the later placement run to revalidate physical reuse.
        bool tracked = dataflow.isTrackedType &&
                       dataflow.isTrackedType(allocation.getResult().getType());
        if (tracked && !dataflow.demands.empty() &&
            llvm::none_of(dataflow.demands, [&](LifetimeDemand demand) {
              return demand.allocation == allocation;
            }))
          return false;
        continue;
      }
      if (!hasOnlyWitnessedRootlessStorageEffects(&candidate))
        return false;

      NCCSynchronizationContract candidateContract =
          getNCCSynchronizationContract(&candidate);
      uint32_t candidateWorkerMask = getNCCIssueWorkerMask(candidateContract);
      AccessCollection current = collectAccesses(&candidate, *candidatePoint,
                                                 candidateWorkerMask, dataflow);
      if (!candidatePoint->path.implies(bodyPoint->path) ||
          !mlir::isa<WaferNCCIssueOpInterface>(&candidate) ||
          candidateContract.behavior !=
              NCCSynchronizationBehavior::OrderedAsynchronousIssue ||
          candidateWorkerMask == 0)
        return false;
      if (!current.hasTrackedEffect) {
        // An NCC issue that touches only another witnessed storage domain is
        // transparent to this domain's backedge proof.
        continue;
      }
      if (!current.allResolved || current.accesses.empty())
        return false;
      bool conflictsWithIssue =
          llvm::any_of(issueAccesses, [&](const PendingAccess *pending) {
            return llvm::any_of(current.accesses,
                                [&](const PendingAccess &access) {
                                  return accessesMayConflict(*pending, access);
                                });
          });
      if (conflictsWithIssue && candidateWorkerMask != issue.workerMask)
        return false;
    }
    return true;
  };
  bool resolvedOrderedStream = provesStructuredOrderedStream(*forOp.getBody());
  if (resolvedOrderedStream)
    return true;

  llvm::SmallVector<mlir::Operation *, 16> nextIteration;
  forOp.getRegion().walk([&](mlir::Operation *candidate) {
    if (dataflow.timeline.lookup(candidate))
      nextIteration.push_back(candidate);
  });
  llvm::sort(nextIteration, [&](mlir::Operation *lhs, mlir::Operation *rhs) {
    return dataflow.timeline.lookup(lhs)->event <
           dataflow.timeline.lookup(rhs)->event;
  });

  llvm::SmallVector<bool, 4> exactlyOrderedAccesses;
  exactlyOrderedAccesses.reserve(issueAccesses.size());
  for (const PendingAccess *pending : issueAccesses) {
    // A read that remains in flight across the backedge is harmless until a
    // later operation writes the same address. The scan below still rejects
    // every such write unless it is a typed same-worker successor; no RAR
    // completion edge is required. Writes must be covered by an exact
    // unconditional successor or an explicit participant join.
    exactlyOrderedAccesses.push_back(!pending->write);
  }
  for (mlir::Operation *candidate : nextIteration) {
    std::optional<ProgramPoint> candidatePoint =
        dataflow.timeline.lookup(candidate);
    if (!candidatePoint ||
        !candidatePoint->path.compatibleWith(bodyPoint->path))
      continue;
    bool unconditionalInBody =
        candidatePoint->path.implies(bodyPoint->path) &&
        isUnconditionallyNestedInStaticFor(candidate, forOp);
    NCCSynchronizationContract candidateContract =
        getNCCSynchronizationContract(candidate);
    bool coversIssue =
        (candidateContract.behavior ==
             NCCSynchronizationBehavior::ParticipantJoin ||
         candidateContract.behavior ==
             NCCSynchronizationBehavior::SynchronousWriteback) &&
        (candidateContract.participantMask & issue.workerMask) != 0;
    if (coversIssue) {
      if (unconditionalInBody)
        return true;
      // A conditional completion is safe on the path where it executes, but
      // cannot prove the other next-iteration paths.
      continue;
    }

    if (auto nestedFor = mlir::dyn_cast<mlir::scf::ForOp>(candidate)) {
      if (isStaticallyNonEmpty(nestedFor) &&
          isUnconditionallyNestedInStaticFor(candidate, forOp))
        continue;
      return false;
    }
    // Structured region containers do not independently observe memory. Their
    // nested operations are present in `nextIteration` with exact path
    // conditions and are checked below; treating the recursive effect summary
    // on the container as another observer would discard that path proof.
    if (candidate->getNumRegions() != 0)
      continue;
    if (mlir::isMemoryEffectFree(candidate))
      continue;
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(candidate)) {
      bool tracked = dataflow.isTrackedType &&
                     dataflow.isTrackedType(allocation.getResult().getType());
      if (!tracked || dataflow.demands.empty() ||
          llvm::any_of(dataflow.demands, [&](LifetimeDemand demand) {
            return demand.allocation == allocation;
          }))
        continue;
      return false;
    }
    if (!hasOnlyWitnessedRootlessStorageEffects(candidate))
      return false;

    uint32_t candidateWorkerMask = getNCCIssueWorkerMask(candidateContract);
    AccessCollection current = collectAccesses(candidate, *candidatePoint,
                                               candidateWorkerMask, dataflow);
    if (!current.hasTrackedEffect)
      continue;
    for (auto [pendingIndex, pending] : llvm::enumerate(issueAccesses)) {
      for (const PendingAccess &access : current.accesses) {
        if (!accessesMayConflict(*pending, access))
          continue;

        // A typed same-worker successor is ordered for every overlapping
        // range, so it is not an observer that forces a host-side completion.
        // Accumulate only exact SSA access identities: one successor may cover
        // one range of a multi-access predecessor and a later successor may
        // cover another. DTE, cross-worker and conditional successors still
        // fail closed at the first conflicting access.
        bool sameWorkerOrderedSuccessor =
            mlir::isa<WaferNCCIssueOpInterface>(candidate) &&
            candidateContract.behavior ==
                NCCSynchronizationBehavior::OrderedAsynchronousIssue &&
            candidateWorkerMask == issue.workerMask && current.allResolved &&
            !current.accesses.empty();
        if (!sameWorkerOrderedSuccessor)
          return false;
        // A conditional same-worker issue is safe on paths where it executes,
        // but cannot by itself prove the other paths. Only an unconditional
        // exact identity contributes to full backedge coverage.
        if (unconditionalInBody && pending->accessIdentity &&
            access.accessIdentity &&
            pending->accessIdentity == access.accessIdentity)
          exactlyOrderedAccesses[pendingIndex] = true;
      }
    }
    if (llvm::all_of(exactlyOrderedAccesses,
                     [](bool ordered) { return ordered; }))
      return true;
  }
  return false;
}

mlir::LogicalResult
LocalCompletionTracker::verifyLoopBackedge(mlir::Operation *loop,
                                           LifetimeDataflow &dataflow,
                                           LifetimeFailure *failure) const {
  auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(loop);
  if (forOp && !hasPossibleBackedge(forOp))
    return mlir::success();
  if (forOp) {
    std::optional<ProgramPoint> bodyPoint =
        dataflow.timeline.lookup(forOp.getBody()->getTerminator());
    uint32_t uniformWorkerMask = 0;
    bool hasNestedIssue = false;
    bool uniformIssues = static_cast<bool>(bodyPoint);
    {
      wafer::support::ScopedCompileTimingSpan timing("analysis-algorithm-phase",
                                                     "verifyLoopBackedge",
                                                     "scan-pending-issues");
      for (const PendingIssue &issue : pendingIssues) {
        if (!isNestedIn(issue.origin, loop) || issue.origin == loop)
          continue;
        hasNestedIssue = true;
        if (!bodyPoint || issue.workerMask == 0 || !issue.accessOrderResolved ||
            !issue.path.implies(bodyPoint->path)) {
          uniformIssues = false;
          break;
        }
        if (uniformWorkerMask == 0)
          uniformWorkerMask = issue.workerMask;
        else if (uniformWorkerMask != issue.workerMask) {
          uniformIssues = false;
          break;
        }
      }
    }
    if (!hasNestedIssue)
      return mlir::success();

    // Prove a homogeneous ordered loop stream once for all of its pending
    // issues. The per-issue proof below remains necessary for mixed workers or
    // partially resolved streams, but rescanning the complete loop for every
    // operation is redundant when every tracked issue has the same stable
    // address domain and worker.
    std::function<bool(mlir::Block &)> guaranteesUniformIssueOnEveryPath;
    guaranteesUniformIssueOnEveryPath = [&](mlir::Block &block) {
      for (mlir::Operation &candidate : block.without_terminator()) {
        NCCSynchronizationContract contract =
            getNCCSynchronizationContract(&candidate);
        if (mlir::isa<WaferNCCIssueOpInterface>(candidate) &&
            contract.behavior ==
                NCCSynchronizationBehavior::OrderedAsynchronousIssue &&
            getNCCIssueWorkerMask(contract) == uniformWorkerMask)
          return true;
        if (auto nestedFor = mlir::dyn_cast<mlir::scf::ForOp>(candidate)) {
          if (isStaticallyNonEmpty(nestedFor) &&
              guaranteesUniformIssueOnEveryPath(*nestedFor.getBody()))
            return true;
          continue;
        }
        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(candidate)) {
          if (!ifOp.getElseRegion().empty() &&
              guaranteesUniformIssueOnEveryPath(ifOp.getThenRegion().front()) &&
              guaranteesUniformIssueOnEveryPath(ifOp.getElseRegion().front()))
            return true;
        }
      }
      return false;
    };

    std::function<bool(mlir::Block &)> provesUniformOrderedStream;
    provesUniformOrderedStream = [&](mlir::Block &block) {
      for (mlir::Operation &candidate : block.without_terminator()) {
        if (auto nestedFor = mlir::dyn_cast<mlir::scf::ForOp>(candidate)) {
          if (!isStaticallyNonEmpty(nestedFor) ||
              !provesUniformOrderedStream(*nestedFor.getBody()))
            return false;
          continue;
        }
        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(candidate)) {
          if (ifOp.getElseRegion().empty())
            return false;
          bool containsIssue = false;
          ifOp.walk([&](WaferNCCIssueOpInterface) { containsIssue = true; });
          if ((containsIssue && (!guaranteesUniformIssueOnEveryPath(
                                     ifOp.getThenRegion().front()) ||
                                 !guaranteesUniformIssueOnEveryPath(
                                     ifOp.getElseRegion().front()))) ||
              !provesUniformOrderedStream(ifOp.getThenRegion().front()) ||
              !provesUniformOrderedStream(ifOp.getElseRegion().front()))
            return false;
          continue;
        }
        if (candidate.getNumRegions() != 0)
          return false;

        std::optional<ProgramPoint> candidatePoint =
            dataflow.timeline.lookup(&candidate);
        if (!candidatePoint)
          return mlir::isMemoryEffectFree(&candidate);
        if (mlir::isMemoryEffectFree(&candidate))
          continue;
        if (auto allocation =
                mlir::dyn_cast<mlir::memref::AllocOp>(candidate)) {
          bool tracked =
              dataflow.isTrackedType &&
              dataflow.isTrackedType(allocation.getResult().getType());
          if (tracked && !dataflow.demands.empty() &&
              llvm::none_of(dataflow.demands, [&](LifetimeDemand demand) {
                return demand.allocation == allocation;
              }))
            return false;
          continue;
        }
        if (!hasOnlyWitnessedRootlessStorageEffects(&candidate))
          return false;

        NCCSynchronizationContract contract =
            getNCCSynchronizationContract(&candidate);
        uint32_t candidateWorkerMask = getNCCIssueWorkerMask(contract);
        AccessCollection current = collectAccesses(
            &candidate, *candidatePoint, candidateWorkerMask, dataflow);
        if (!candidatePoint->path.implies(bodyPoint->path) ||
            !mlir::isa<WaferNCCIssueOpInterface>(&candidate) ||
            contract.behavior !=
                NCCSynchronizationBehavior::OrderedAsynchronousIssue ||
            candidateWorkerMask == 0)
          return false;
        if (!current.hasTrackedEffect)
          continue;
        if (!current.allResolved || current.accesses.empty() ||
            candidateWorkerMask != uniformWorkerMask)
          return false;
      }
      return true;
    };
    if (hasNestedIssue && uniformIssues) {
      wafer::support::ScopedCompileTimingSpan timing("analysis-algorithm-phase",
                                                     "verifyLoopBackedge",
                                                     "prove-uniform-stream");
      if (provesUniformOrderedStream(*forOp.getBody()))
        return mlir::success();
    }

    // The exact per-issue proof below scans the same loop body once for every
    // pending issue. Summarize both of its success branches for every issue in
    // two body scans: a fully ordered structured stream, or an unconditional
    // participant join reached before an unsafe observer. This preserves the
    // per-issue root/worker/path decisions instead of requiring a homogeneous
    // stream. Any issue outside these exact preconditions falls back to the
    // original proof.
    auto accessesMayConflict = [](const PendingAccess &pending,
                                  const PendingAccess &access) {
      if (!pending.root.path.intersect(access.root.path) ||
          (!pending.write && !access.write))
        return false;

      bool pendingRootResolved =
          pending.root.demandIndex != kUnresolvedRootIndex;
      bool currentRootResolved =
          access.root.demandIndex != kUnresolvedRootIndex;
      if (pendingRootResolved && currentRootResolved &&
          pending.root.demandIndex != access.root.demandIndex)
        return false;
      if ((!pendingRootResolved || !currentRootResolved) &&
          pending.logicalRoot && access.logicalRoot &&
          pending.logicalRoot != access.logicalRoot &&
          mlir::isa_and_nonnull<mlir::memref::AllocOp>(
              pending.logicalRoot.getDefiningOp()) &&
          mlir::isa_and_nonnull<mlir::memref::AllocOp>(
              access.logicalRoot.getDefiningOp()))
        return false;
      return true;
    };

    struct BatchedIssueState {
      const PendingIssue *issue = nullptr;
      llvm::SmallVector<const PendingAccess *, 4> accesses;
      bool structuredEligible = true;
      bool fallbackAlive = true;
      bool fallbackCompleted = false;
    };
    llvm::SmallVector<BatchedIssueState, 16> batchedIssues;
    bool canBatchResolvedIssues = static_cast<bool>(bodyPoint);
    for (const PendingIssue &issue : pendingIssues) {
      if (!isNestedIn(issue.origin, loop) || issue.origin == loop)
        continue;
      if (!bodyPoint || issue.workerMask == 0 || !issue.accessOrderResolved ||
          !issue.path.implies(bodyPoint->path)) {
        canBatchResolvedIssues = false;
        break;
      }

      BatchedIssueState state;
      state.issue = &issue;
      for (const PendingAccess &access : pendingAccesses) {
        if (access.origin != issue.origin ||
            !access.root.path.intersect(issue.path))
          continue;
        state.accesses.push_back(&access);
      }
      if (state.accesses.empty()) {
        canBatchResolvedIssues = false;
        break;
      }
      batchedIssues.push_back(std::move(state));
    }

    llvm::DenseSet<mlir::Operation *> trackedAllocations;
    for (LifetimeDemand demand : dataflow.demands)
      trackedAllocations.insert(demand.allocation.getOperation());

    auto invalidateStructured = [&] {
      for (BatchedIssueState &state : batchedIssues)
        state.structuredEligible = false;
    };
    std::function<void(mlir::Block &)> summarizeStructuredOrderedStream;
    summarizeStructuredOrderedStream = [&](mlir::Block &block) {
      for (mlir::Operation &candidate : block.without_terminator()) {
        if (auto nestedFor = mlir::dyn_cast<mlir::scf::ForOp>(candidate)) {
          if (!isStaticallyNonEmpty(nestedFor)) {
            invalidateStructured();
            return;
          }
          summarizeStructuredOrderedStream(*nestedFor.getBody());
          continue;
        }
        if (candidate.getNumRegions() != 0) {
          invalidateStructured();
          return;
        }

        std::optional<ProgramPoint> candidatePoint =
            dataflow.timeline.lookup(&candidate);
        if (!candidatePoint) {
          if (!mlir::isMemoryEffectFree(&candidate)) {
            invalidateStructured();
            return;
          }
          continue;
        }
        if (mlir::isMemoryEffectFree(&candidate))
          continue;
        if (auto allocation =
                mlir::dyn_cast<mlir::memref::AllocOp>(candidate)) {
          bool tracked = dataflow.isTrackedType &&
                         dataflow.isTrackedType(allocation.getType());
          if (tracked && !dataflow.demands.empty() &&
              !trackedAllocations.contains(allocation.getOperation())) {
            invalidateStructured();
            return;
          }
          continue;
        }
        if (!hasOnlyWitnessedRootlessStorageEffects(&candidate)) {
          invalidateStructured();
          return;
        }

        NCCSynchronizationContract candidateContract =
            getNCCSynchronizationContract(&candidate);
        uint32_t candidateWorkerMask = getNCCIssueWorkerMask(candidateContract);
        AccessCollection current = collectAccesses(
            &candidate, *candidatePoint, candidateWorkerMask, dataflow);
        if (!candidatePoint->path.implies(bodyPoint->path) ||
            !mlir::isa<WaferNCCIssueOpInterface>(&candidate) ||
            candidateContract.behavior !=
                NCCSynchronizationBehavior::OrderedAsynchronousIssue ||
            candidateWorkerMask == 0) {
          invalidateStructured();
          return;
        }
        if (!current.hasTrackedEffect)
          continue;
        if (!current.allResolved || current.accesses.empty()) {
          invalidateStructured();
          return;
        }

        for (BatchedIssueState &state : batchedIssues) {
          if (!state.structuredEligible ||
              candidateWorkerMask == state.issue->workerMask)
            continue;
          bool conflicts =
              llvm::any_of(state.accesses, [&](const PendingAccess *pending) {
                return llvm::any_of(
                    current.accesses, [&](const PendingAccess &access) {
                      return accessesMayConflict(*pending, access);
                    });
              });
          if (conflicts)
            state.structuredEligible = false;
        }
      }
    };

    auto invalidateFallback = [&](llvm::ArrayRef<unsigned> activeStates) {
      for (unsigned stateIndex : activeStates)
        batchedIssues[stateIndex].fallbackAlive = false;
    };
    auto summarizeJoinBeforeObserver = [&] {
      llvm::SmallVector<mlir::Operation *, 16> nextIteration;
      forOp.getRegion().walk([&](mlir::Operation *candidate) {
        if (dataflow.timeline.lookup(candidate))
          nextIteration.push_back(candidate);
      });
      llvm::sort(nextIteration,
                 [&](mlir::Operation *lhs, mlir::Operation *rhs) {
                   return dataflow.timeline.lookup(lhs)->event <
                          dataflow.timeline.lookup(rhs)->event;
                 });

      for (mlir::Operation *candidate : nextIteration) {
        std::optional<ProgramPoint> candidatePoint =
            dataflow.timeline.lookup(candidate);
        if (!candidatePoint ||
            !candidatePoint->path.compatibleWith(bodyPoint->path))
          continue;
        bool unconditionalInBody =
            candidatePoint->path.implies(bodyPoint->path) &&
            isUnconditionallyNestedInStaticFor(candidate, forOp);
        NCCSynchronizationContract candidateContract =
            getNCCSynchronizationContract(candidate);

        llvm::SmallVector<unsigned, 16> activeStates;
        for (auto [stateIndex, state] : llvm::enumerate(batchedIssues)) {
          if (!state.fallbackAlive || state.fallbackCompleted)
            continue;
          bool coversIssue =
              (candidateContract.behavior ==
                   NCCSynchronizationBehavior::ParticipantJoin ||
               candidateContract.behavior ==
                   NCCSynchronizationBehavior::SynchronousWriteback) &&
              (candidateContract.participantMask & state.issue->workerMask) !=
                  0;
          if (coversIssue) {
            if (unconditionalInBody)
              state.fallbackCompleted = true;
            // A conditional completion is safe where it executes, but this
            // candidate cannot prove or invalidate the remaining paths.
            continue;
          }
          activeStates.push_back(stateIndex);
        }
        if (activeStates.empty())
          continue;

        if (auto nestedFor = mlir::dyn_cast<mlir::scf::ForOp>(candidate)) {
          if (!isStaticallyNonEmpty(nestedFor) || !unconditionalInBody)
            invalidateFallback(activeStates);
          continue;
        }
        // Region bodies are scanned separately with their timeline paths; the
        // containing op's recursive effect summary is not an extra observer.
        if (candidate->getNumRegions() != 0)
          continue;
        if (mlir::isMemoryEffectFree(candidate))
          continue;
        if (auto allocation =
                mlir::dyn_cast<mlir::memref::AllocOp>(candidate)) {
          bool tracked = dataflow.isTrackedType &&
                         dataflow.isTrackedType(allocation.getType());
          if (tracked && !dataflow.demands.empty() &&
              !trackedAllocations.contains(allocation.getOperation()))
            invalidateFallback(activeStates);
          continue;
        }
        if (!hasOnlyWitnessedRootlessStorageEffects(candidate)) {
          invalidateFallback(activeStates);
          continue;
        }

        uint32_t candidateWorkerMask = getNCCIssueWorkerMask(candidateContract);
        AccessCollection current = collectAccesses(
            candidate, *candidatePoint, candidateWorkerMask, dataflow);
        if (!current.hasTrackedEffect)
          continue;
        for (unsigned stateIndex : activeStates) {
          BatchedIssueState &state = batchedIssues[stateIndex];
          bool conflicts =
              llvm::any_of(state.accesses, [&](const PendingAccess *pending) {
                return llvm::any_of(
                    current.accesses, [&](const PendingAccess &access) {
                      return accessesMayConflict(*pending, access);
                    });
              });
          if (!conflicts)
            continue;
          bool sameWorkerOrderedSuccessor =
              mlir::isa<WaferNCCIssueOpInterface>(candidate) &&
              candidateContract.behavior ==
                  NCCSynchronizationBehavior::OrderedAsynchronousIssue &&
              candidateWorkerMask == state.issue->workerMask &&
              current.allResolved && !current.accesses.empty();
          if (!sameWorkerOrderedSuccessor)
            state.fallbackAlive = false;
        }
      }
    };

    if (!batchedIssues.empty() && canBatchResolvedIssues) {
      wafer::support::ScopedCompileTimingSpan timing("analysis-algorithm-phase",
                                                     "verifyLoopBackedge",
                                                     "prove-batched-stream");
      summarizeStructuredOrderedStream(*forOp.getBody());
      summarizeJoinBeforeObserver();
      bool provesEveryIssue =
          llvm::all_of(batchedIssues, [](const BatchedIssueState &state) {
            return state.structuredEligible || state.fallbackCompleted;
          });
      if (provesEveryIssue)
        return mlir::success();
    }
  }
  for (const PendingIssue &issue : pendingIssues) {
    if (!isNestedIn(issue.origin, loop) || issue.origin == loop)
      continue;
    {
      wafer::support::ScopedCompileTimingSpan timing("analysis-algorithm-phase",
                                                     "verifyLoopBackedge",
                                                     "prove-individual-stream");
      if (provesLoopBackedgeOrder(issue, loop, dataflow))
        continue;
    }
    setLifetimeFailure(failure, LifetimeFailureKind::LoopBackedgeCompletion,
                       issue.origin);
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
LocalCompletionTracker::finish(mlir::Operation *scope,
                               LifetimeFailure *failure) const {
  // A tile region is a scheduling/lifetime container, not an externally
  // observable completion cut. Its enclosing function analysis reconstructs
  // these pending issues and requires a typed join before a real observer or
  // terminal boundary.
  if (mlir::isa_and_nonnull<TileRegionOp>(scope))
    return mlir::success();
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
