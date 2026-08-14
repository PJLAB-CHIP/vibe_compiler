//===- DirectDTETransport.cpp - Physical Direct DTE acceptance ----------===//

#include "DirectDTETransport.h"

#include "AcceptedCallClosure.h"
#include "Wafer/Analysis/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/MemoryPlanning/StaticIndexRange.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct StructuredLoopSite {
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;
  int64_t siblingOrdinal = -1;

  auto asTuple() const { return std::tie(lower, upper, step, siblingOrdinal); }
  bool operator==(const StructuredLoopSite &other) const {
    return asTuple() == other.asTuple();
  }
  bool operator<(const StructuredLoopSite &other) const {
    return asTuple() < other.asTuple();
  }
};

using MessageBaseKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;

struct PhysicalRange {
  int64_t start = -1;
  int64_t end = -1;
};

struct AcceptedRangePattern {
  llvm::SmallVector<PhysicalRange, 4> ranges;
  mlir::Operation *rotatingLoop = nullptr;
};

struct IssueRecord {
  mlir::Operation *operation = nullptr;
  mlir::Operation *wait = nullptr;
  mlir::Block *block = nullptr;
  int64_t physicalTileIndex = -1;
  MessageBaseKey message;
  AcceptedRangePattern rangePattern;
  int64_t bytes = -1;
  unsigned issueIndex = 0;
  unsigned waitIndex = 0;
  int64_t receiverFsmId = -1;
  bool isSend = false;
};

struct DynamicIssue {
  unsigned issue = 0;
  PhysicalRange range;
  uint64_t selector = 0;
};

struct DynamicMessageStream {
  llvm::SmallVector<DynamicIssue, 16> sends;
  llvm::SmallVector<DynamicIssue, 16> recvs;
};

struct MatchedMessage {
  IssueRecord *send = nullptr;
  IssueRecord *recv = nullptr;
};

enum class TransportActionKind { SendIssue, ReceivePrepare, Wait };

enum class StructuredExecutionFrameKind { Loop, Call, TileRegion };

struct StructuredExecutionFrame {
  StructuredExecutionFrameKind kind = StructuredExecutionFrameKind::Loop;
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;
  int64_t siblingOrdinal = -1;

  auto asTuple() const {
    return std::tie(kind, lower, upper, step, siblingOrdinal);
  }
  bool operator==(const StructuredExecutionFrame &other) const {
    return asTuple() == other.asTuple();
  }
};

struct TransportAction {
  mlir::Operation *operation = nullptr;
  int64_t physicalTileIndex = -1;
  TransportActionKind kind = TransportActionKind::Wait;
  IssueRecord *issue = nullptr;
  llvm::SmallVector<unsigned, 4> waitedIssueActions;
  llvm::SmallVector<StructuredExecutionFrame, 4> occurrencePath;
};

struct StructuredTransportTrace {
  llvm::SmallVector<TransportAction, 64> actions;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 64> dependencies;
  llvm::SmallVector<std::pair<IssueRecord *, IssueRecord *>, 16>
      receiverConflicts;
};

static bool
haveSameDynamicOccurrencePath(llvm::ArrayRef<StructuredExecutionFrame> lhs,
                              llvm::ArrayRef<StructuredExecutionFrame> rhs) {
  auto lhsIt = lhs.begin();
  auto rhsIt = rhs.begin();
  while (true) {
    while (lhsIt != lhs.end() &&
           lhsIt->kind == StructuredExecutionFrameKind::TileRegion)
      ++lhsIt;
    while (rhsIt != rhs.end() &&
           rhsIt->kind == StructuredExecutionFrameKind::TileRegion)
      ++rhsIt;
    if (lhsIt == lhs.end() || rhsIt == rhs.end())
      return lhsIt == lhs.end() && rhsIt == rhs.end();
    if (!(*lhsIt == *rhsIt))
      return false;
    ++lhsIt;
    ++rhsIt;
  }
}

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 ||
      (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs))
    return false;
  result = lhs * rhs;
  return true;
}

static mlir::Value resolveTileRegionBoundaryValue(mlir::Value value) {
  while (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Value entry =
        analysis::getSingleExecutionRegionEntryOperand(blockArg);
    if (!entry)
      return value;
    value = entry;
  }
  return value;
}

static mlir::Value getRootViewSource(mlir::Value value) {
  value = resolveTileRegionBoundaryValue(value);
  while (mlir::Operation *def = value.getDefiningOp()) {
    auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(def);
    if (!viewLike)
      return value;
    mlir::Value source =
        resolveTileRegionBoundaryValue(viewLike.getViewSource());
    if (source == value)
      return value;
    value = source;
  }
  return value;
}

static mlir::FailureOr<int64_t>
getStaticViewOffsetBytes(mlir::Operation *op, mlir::MemRefType viewType) {
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offsetElements = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(viewType, strides, offsetElements)) ||
      offsetElements == mlir::ShapedType::kDynamic || offsetElements < 0)
    return op->emitError(
        "direct_dte_acceptance: buffer view requires a static non-negative "
        "layout offset");

  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(viewType);
  if (!info || info->physicalBytes < 0)
    return op->emitError(
        "direct_dte_acceptance: buffer physical range is not representable");
  if (info->layout == MemLayout::Cx || info->layout == MemLayout::NCx ||
      info->bitPackedElement) {
    if (offsetElements != 0)
      return op->emitError(
          "direct_dte_acceptance: non-zero blocked or bitpacked view offset "
          "is unsupported");
    return 0;
  }
  if (info->elementBytes <= 0)
    return op->emitError(
        "direct_dte_acceptance: buffer element byte size is invalid");
  int64_t offsetBytes = 0;
  if (!checkedMul(offsetElements, info->elementBytes, offsetBytes))
    return op->emitError(
        "direct_dte_acceptance: buffer view byte offset overflows int64");
  return offsetBytes;
}

static mlir::LogicalResult
collectPlannedSPMRoots(mlir::Operation *op, mlir::Value value,
                       llvm::DenseSet<mlir::Value> &active,
                       llvm::SmallVectorImpl<mlir::memref::AllocOp> &roots) {
  value = resolveTileRegionBoundaryValue(value);
  if (!active.insert(value).second)
    return mlir::success();
  auto finish = [&](mlir::LogicalResult result) {
    active.erase(value);
    return result;
  };

  if (auto allocation = value.getDefiningOp<mlir::memref::AllocOp>()) {
    if (!llvm::is_contained(roots, allocation))
      roots.push_back(allocation);
    return finish(mlir::success());
  }
  if (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
          value.getDefiningOp()))
    return finish(
        collectPlannedSPMRoots(op, view.getViewSource(), active, roots));

  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = argument.getOwner();
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
        owner ? owner->getParentOp() : nullptr);
    if (!loop || owner != loop.getBody() || argument.getArgNumber() == 0)
      return finish(op->emitError(
          "direct_dte_acceptance: rotating SPM value must be rooted in a "
          "static scf.for iter_arg"));
    unsigned index = argument.getArgNumber() - 1;
    if (index >= loop.getInitArgs().size())
      return finish(op->emitError(
          "direct_dte_acceptance: scf.for iter_arg has no matching init"));
    if (mlir::failed(collectPlannedSPMRoots(op, loop.getInitArgs()[index],
                                            active, roots)))
      return finish(mlir::failure());
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || index >= yield.getResults().size())
      return finish(op->emitError(
          "direct_dte_acceptance: scf.for iter_arg has no matching yield"));
    return finish(
        collectPlannedSPMRoots(op, yield.getResults()[index], active, roots));
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  auto loop = result ? mlir::dyn_cast<mlir::scf::ForOp>(result.getOwner())
                     : mlir::scf::ForOp();
  if (loop) {
    unsigned index = result.getResultNumber();
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    if (index >= loop.getInitArgs().size() || !yield ||
        index >= yield.getResults().size())
      return finish(op->emitError(
          "direct_dte_acceptance: scf.for result has no exact recurrence"));
    if (mlir::failed(collectPlannedSPMRoots(op, loop.getInitArgs()[index],
                                            active, roots)))
      return finish(mlir::failure());
    return finish(
        collectPlannedSPMRoots(op, yield.getResults()[index], active, roots));
  }

  return finish(op->emitError(
      "direct_dte_acceptance: buffer must be rooted in a planned SPM "
      "allocation or a static rotating slot family"));
}

struct StaticLoopBounds {
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;
};

static std::optional<int64_t> getStaticIndexConstant(mlir::Value value,
                                                     mlir::Operation *use) {
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
    return constant;
  ::wafer::memory_planning::detail::StaticIndexRangeResult range =
      ::wafer::memory_planning::detail::evaluateNonNegativeStaticIndexRange(
          value, use);
  if (!range.succeeded() || range.range.empty ||
      range.range.min != range.range.max)
    return std::nullopt;
  return range.range.min;
}

static std::optional<StaticLoopBounds>
getStaticLoopBounds(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      getStaticIndexConstant(loop.getLowerBound(), loop);
  std::optional<int64_t> upper =
      getStaticIndexConstant(loop.getUpperBound(), loop);
  std::optional<int64_t> step = getStaticIndexConstant(loop.getStep(), loop);
  if (!lower || !upper || !step || *step <= 0)
    return std::nullopt;
  return StaticLoopBounds{*lower, *upper, *step};
}

static std::optional<uint64_t> getStaticTripCount(mlir::scf::ForOp loop) {
  std::optional<StaticLoopBounds> bounds = getStaticLoopBounds(loop);
  if (!bounds)
    return std::nullopt;
  if (bounds->lower >= bounds->upper)
    return uint64_t{0};
  __int128 span = static_cast<__int128>(bounds->upper) -
                  static_cast<__int128>(bounds->lower);
  __int128 count = (span + static_cast<__int128>(bounds->step) - 1) /
                   static_cast<__int128>(bounds->step);
  if (count < 0 ||
      count > static_cast<__int128>(std::numeric_limits<uint64_t>::max()))
    return std::nullopt;
  return static_cast<uint64_t>(count);
}

static mlir::FailureOr<AcceptedRangePattern>
resolveAcceptedRanges(mlir::Operation *op, mlir::Value buffer, int64_t bytes) {
  auto viewType = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
  if (!viewType || !isWaferSPMMemRefType(viewType))
    return op->emitError(
        "direct_dte_acceptance: issue buffer must be a Wafer SPM memref");
  std::optional<WaferPhysicalTensorInfo> viewInfo =
      computeWaferPhysicalTensorInfo(viewType);
  if (!viewInfo || viewInfo->physicalBytes < bytes)
    return op->emitError(
        "direct_dte_acceptance: issue bytes exceed the buffer view range");
  mlir::FailureOr<int64_t> viewOffset = getStaticViewOffsetBytes(op, viewType);
  if (mlir::failed(viewOffset))
    return mlir::failure();

  llvm::SmallVector<mlir::memref::AllocOp, 4> roots;
  llvm::DenseSet<mlir::Value> active;
  if (mlir::failed(collectPlannedSPMRoots(op, buffer, active, roots)) ||
      roots.empty())
    return mlir::failure();

  WaferTargetPolicy policy = getDefaultWaferTargetPolicy();
  llvm::SmallVector<PhysicalRange, 4> ranges;
  for (mlir::memref::AllocOp allocation : roots) {
    auto rootType = mlir::dyn_cast<mlir::MemRefType>(allocation.getType());
    if (!rootType || !isWaferSPMMemRefType(rootType))
      return op->emitError(
          "direct_dte_acceptance: rotating root is not a Wafer SPM "
          "allocation");
    auto acceptedOffset =
        allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
    if (!acceptedOffset)
      return op->emitError(
          "direct_dte_acceptance: SPM root is missing accepted "
          "wafer.spm.offset");
    std::optional<WaferPhysicalTensorInfo> rootInfo =
        computeWaferPhysicalTensorInfo(rootType);
    if (!rootInfo || rootInfo->physicalBytes < 0)
      return op->emitError(
          "direct_dte_acceptance: root physical range is not representable");

    int64_t start = 0;
    int64_t end = 0;
    int64_t rootEnd = 0;
    if (!checkedAdd(acceptedOffset.getOffset(), *viewOffset, start) ||
        !checkedAdd(start, bytes, end) ||
        !checkedAdd(acceptedOffset.getOffset(), rootInfo->physicalBytes,
                    rootEnd))
      return op->emitError(
          "direct_dte_acceptance: accepted SPM range overflows int64");
    if (start < policy.memory.spmBase || end > policy.memory.spmLimit ||
        end > rootEnd)
      return op->emitError(
          "direct_dte_acceptance: issue range is outside its accepted SPM "
          "allocation");
    ranges.push_back(PhysicalRange{start, end});
  }
  mlir::Value root = getRootViewSource(buffer);
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root)) {
    mlir::Block *owner = argument.getOwner();
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
        owner ? owner->getParentOp() : nullptr);
    if (loop && owner == loop.getBody() && argument.getArgNumber() > 0)
      return AcceptedRangePattern{std::move(ranges), loop.getOperation()};
  }
  if (auto result = mlir::dyn_cast<mlir::OpResult>(root)) {
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(result.getOwner())) {
      std::optional<uint64_t> tripCount = getStaticTripCount(loop);
      if (!tripCount || ranges.empty())
        return op->emitError(
            "direct_dte_acceptance: rotating loop result requires a "
            "representable static trip count");
      PhysicalRange finalRange =
          ranges[static_cast<size_t>(*tripCount % ranges.size())];
      ranges.assign(1, finalRange);
    }
  }
  if (ranges.size() > 1)
    return op->emitError(
        "direct_dte_acceptance: multi-root SPM range has no explicit static "
        "rotation");
  return AcceptedRangePattern{std::move(ranges), nullptr};
}

static mlir::FailureOr<mlir::Operation *> findUniqueSameBlockWait(
    mlir::Operation *issue, mlir::Value token,
    const llvm::DenseMap<mlir::Operation *, unsigned> &operationIndices) {
  if (!token.hasOneUse())
    return issue->emitError(
        "direct_dte_acceptance: issue token must have exactly one wait use");
  mlir::Operation *wait = token.use_begin()->getOwner();
  if (!mlir::isa<InstrDTEWaitOp>(wait) || wait->getBlock() != issue->getBlock())
    return issue->emitError(
        "direct_dte_acceptance: issue token must be consumed by one same-block "
        "wafer.instr.dte_wait");
  auto issueIt = operationIndices.find(issue);
  auto waitIt = operationIndices.find(wait);
  if (issueIt == operationIndices.end() || waitIt == operationIndices.end() ||
      issueIt->second >= waitIt->second)
    return issue->emitError(
        "direct_dte_acceptance: DTE wait must follow its issue");
  return wait;
}

static mlir::LogicalResult verifyIssueBufferIsolation(mlir::Operation *issue,
                                                      mlir::Operation *wait,
                                                      mlir::Value buffer) {
  mlir::Value root = getRootViewSource(buffer);
  const bool issueWritesBuffer = mlir::isa<InstrDTERecvOp>(issue);
  for (mlir::Operation *operation = issue->getNextNode(); operation != wait;
       operation = operation->getNextNode()) {
    if (!operation)
      llvm_unreachable("same-block ordered wait must be reachable from issue");

    bool hasRootOperand = false;
    operation->walk([&](mlir::Operation *nested) {
      hasRootOperand |=
          llvm::any_of(nested->getOperands(), [&](mlir::Value operand) {
            return getRootViewSource(operand) == root;
          });
    });
    // Region control such as scf.for has recursive memory effects.  Inspect
    // the complete value-specific summary so an explicitly prepared receive
    // may overlap unrelated compute while every nested access to its exact
    // root remains forbidden. Any nested operation without an effect contract
    // makes getEffectsRecursively fail, preserving fail-closed behavior.
    std::optional<llvm::SmallVector<mlir::MemoryEffects::EffectInstance>>
        recursiveEffects = mlir::getEffectsRecursively(operation);
    if (!recursiveEffects)
      return operation->emitError(
          "direct_dte_acceptance: issue buffer has an unknown access before "
          "its matching wait");
    llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> instances =
        *recursiveEffects;
    const bool hasRootValueEffect =
        llvm::any_of(instances, [&](const auto &effect) {
          mlir::Value value = effect.getValue();
          return value && getRootViewSource(value) == root;
        });
    if (hasRootOperand && !hasRootValueEffect)
      return operation->emitError(
          "direct_dte_acceptance: issue buffer has no value-specific memory "
          "effect before its matching wait");

    llvm::StringRef conflictingEffect;
    bool conflicts = llvm::any_of(instances, [&](const auto &effect) {
      mlir::Value value = effect.getValue();
      // Rootless resource effects summarize an execution engine (for example
      // ComputeResource) rather than an operand range. Operand annotations
      // carry the concrete SPM accesses used for buffer isolation.
      if (!value || getRootViewSource(value) != root)
        return false;
      // A send keeps reading its source until completion, so another read of
      // the exact source range is compatible. A receive owns a pending write
      // to its destination, and any read or write before the exact wait is a
      // conflict. Unknown non-read effects remain mutating and fail closed.
      const bool read =
          llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
      const bool conflict = issueWritesBuffer || !read;
      if (conflict) {
        if (read)
          conflictingEffect = "read";
        else if (llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect()))
          conflictingEffect = "write";
        else if (llvm::isa<mlir::MemoryEffects::Allocate>(effect.getEffect()))
          conflictingEffect = "allocate";
        else if (llvm::isa<mlir::MemoryEffects::Free>(effect.getEffect()))
          conflictingEffect = "free";
        else
          conflictingEffect = "mutating-effect";
      }
      return conflict;
    });
    if (conflicts) {
      assert(!conflictingEffect.empty() &&
             "a conflicting memory effect must be classified");
      return operation->emitError(
                 "direct_dte_acceptance: issue buffer must remain isolated "
                 "until its matching wait")
             << ": issue=" << issue->getName()
             << " conflict=" << operation->getName()
             << " effect=" << conflictingEffect;
    }
  }
  return mlir::success();
}

static int64_t getStructuredLoopSiblingOrdinal(mlir::scf::ForOp loop) {
  // This is a path in the structured-control tree, not an ordinal assigned to
  // DTE issues.  It distinguishes sibling transport-bearing main/edge loop
  // families while leaving unrelated static control and the typed DTE message
  // identity unchanged.
  int64_t ordinal = 0;
  for (mlir::Operation &sibling : *loop->getBlock()) {
    if (&sibling == loop.getOperation())
      return ordinal;
    auto siblingLoop = mlir::dyn_cast<mlir::scf::ForOp>(sibling);
    if (!siblingLoop)
      continue;
    bool containsTransport = false;
    siblingLoop.walk([&](mlir::Operation *operation) {
      if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(
              operation)) {
        containsTransport = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (containsTransport)
      ++ordinal;
  }
  llvm_unreachable("scf.for must be present in its parent block");
}

static mlir::FailureOr<llvm::SmallVector<StructuredLoopSite, 4>>
getStructuredLoopSite(mlir::Operation *issue) {
  llvm::SmallVector<StructuredLoopSite, 4> reversedLoops;
  mlir::Operation *nested = issue;
  while (mlir::Operation *parent = nested->getParentOp()) {
    mlir::Region *parentRegion = nested->getParentRegion();
    if (parentRegion && !parentRegion->hasOneBlock())
      return issue->emitError(
          "direct_dte_acceptance: DTE issue requires single-block structured "
          "control flow");

    if (mlir::isa<mlir::scf::IfOp>(parent))
      return issue->emitError(
          "direct_dte_acceptance: DTE issue under scf.if has no statically "
          "matched control instance");

    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
      std::optional<StaticLoopBounds> bounds = getStaticLoopBounds(loop);
      if (!bounds)
        return issue->emitError(
            "direct_dte_acceptance: enclosing scf.for requires constant "
            "bounds and a positive constant step");
      reversedLoops.push_back(
          StructuredLoopSite{bounds->lower, bounds->upper, bounds->step,
                             getStructuredLoopSiblingOrdinal(loop)});
    } else if (!mlir::isa<mlir::func::FuncOp, mlir::ModuleOp, TileRegionOp>(
                   parent) &&
               parent->getNumRegions() != 0) {
      return issue->emitError()
             << "direct_dte_acceptance: unsupported control ancestor "
             << parent->getName() << " for DTE issue";
    }
    nested = parent;
  }
  std::reverse(reversedLoops.begin(), reversedLoops.end());
  return reversedLoops;
}

static MessageBaseKey makeMessageBaseKey(int64_t physicalTileIndex,
                                         int64_t peer, DTEMessageAttr message,
                                         bool isSend) {
  return std::make_tuple(isSend ? physicalTileIndex : peer,
                         isSend ? peer : physicalTileIndex,
                         message.getCommunicationId(), message.getRound(),
                         message.getPayloadSlice());
}

static mlir::LogicalResult
collectIssues(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules,
              llvm::SmallVectorImpl<IssueRecord> &issues) {
  for (size_t physicalTileIndex = 0;
       physicalTileIndex < physicalTileModules.size(); ++physicalTileIndex) {
    mlir::ModuleOp module = physicalTileModules[physicalTileIndex];
    llvm::DenseMap<mlir::Operation *, unsigned> operationIndices;
    module.walk([&](mlir::Block *block) {
      for (auto [index, operation] : llvm::enumerate(*block))
        operationIndices[&operation] = static_cast<unsigned>(index);
    });

    mlir::LogicalResult result = mlir::success();
    module.walk([&](mlir::Operation *operation) {
      if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation)) {
        for (mlir::Value token : wait.getTokens()) {
          mlir::Operation *def = token.getDefiningOp();
          if (!def || !mlir::isa<InstrDTESendOp, InstrDTERecvOp>(def)) {
            wait.emitError(
                "direct_dte_acceptance: every wait token must be produced by "
                "a Direct DTE issue");
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
        }
      }
      auto send = mlir::dyn_cast<InstrDTESendOp>(operation);
      auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation);
      if (!send && !recv)
        return mlir::WalkResult::advance();
      if ((send && send.getBindingAttr()) || (recv && recv.getBindingAttr())) {
        operation->emitError(
            "direct_dte_acceptance: candidate issue already has a physical "
            "binding");
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }

      mlir::Value buffer = send ? send.getBuffer() : recv.getBuffer();
      mlir::Value token = send ? send.getToken() : recv.getToken();
      int64_t peer =
          send ? send.getPeerAttr().getInt() : recv.getPeerAttr().getInt();
      if (peer < 0 ||
          peer >= static_cast<int64_t>(physicalTileModules.size())) {
        operation->emitError(
            "direct_dte_acceptance: peer is outside the supplied physical "
            "Tile domain");
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      int64_t bytes =
          send ? send.getBytesAttr().getInt() : recv.getBytesAttr().getInt();
      DTEMessageAttr message =
          send ? send.getMessageAttr() : recv.getMessageAttr();
      MessageBaseKey messageBase =
          makeMessageBaseKey(static_cast<int64_t>(physicalTileIndex), peer,
                             message, static_cast<bool>(send));
      mlir::FailureOr<llvm::SmallVector<StructuredLoopSite, 4>> loopSite =
          getStructuredLoopSite(operation);
      mlir::FailureOr<AcceptedRangePattern> rangePattern =
          resolveAcceptedRanges(operation, buffer, bytes);
      mlir::FailureOr<mlir::Operation *> wait =
          findUniqueSameBlockWait(operation, token, operationIndices);
      if (mlir::failed(loopSite) || mlir::failed(rangePattern) ||
          mlir::failed(wait)) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      if (mlir::failed(verifyIssueBufferIsolation(operation, *wait, buffer))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      issues.push_back(IssueRecord{
          operation, *wait, operation->getBlock(),
          static_cast<int64_t>(physicalTileIndex), messageBase,
          std::move(*rangePattern), bytes, operationIndices.lookup(operation),
          operationIndices.lookup(*wait), -1, static_cast<bool>(send)});
      return mlir::WalkResult::advance();
    });
    if (mlir::failed(result))
      return mlir::failure();
  }
  return mlir::success();
}

static mlir::FailureOr<PhysicalRange> resolveDynamicRange(
    const IssueRecord &issue,
    const llvm::DenseMap<mlir::Operation *, uint64_t> &loopIterations) {
  if (issue.rangePattern.ranges.empty())
    return issue.operation->emitError(
        "direct_dte_acceptance: accepted SPM range pattern is empty");
  if (!issue.rangePattern.rotatingLoop)
    return issue.rangePattern.ranges.front();
  auto iteration = loopIterations.find(issue.rangePattern.rotatingLoop);
  if (iteration == loopIterations.end())
    return issue.operation->emitError(
        "direct_dte_acceptance: rotating SPM range is not controlled by the "
        "issue's dynamic loop instance");
  return issue.rangePattern.ranges[static_cast<size_t>(
      iteration->second % issue.rangePattern.ranges.size())];
}

static mlir::LogicalResult appendDynamicIssues(
    mlir::Operation *operation, llvm::ArrayRef<IssueRecord> issues,
    const llvm::DenseMap<mlir::Operation *, unsigned> &issueIndices,
    const llvm::DenseSet<mlir::Operation *> &issueAncestors,
    llvm::DenseMap<mlir::Operation *, uint64_t> &loopIterations,
    llvm::MutableArrayRef<uint64_t> occurrenceCounts,
    std::map<MessageBaseKey, DynamicMessageStream> &streams) {
  auto issue = issueIndices.find(operation);
  if (issue != issueIndices.end()) {
    mlir::FailureOr<PhysicalRange> range =
        resolveDynamicRange(issues[issue->second], loopIterations);
    if (mlir::failed(range))
      return mlir::failure();
    DynamicMessageStream &stream = streams[issues[issue->second].message];
    auto &instances =
        issues[issue->second].isSend ? stream.sends : stream.recvs;
    uint64_t selector = occurrenceCounts[issue->second]++;
    instances.push_back(DynamicIssue{issue->second, *range, selector});
  }

  if (!issueAncestors.contains(operation))
    return mlir::success();

  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
    std::optional<uint64_t> tripCount = getStaticTripCount(loop);
    if (!tripCount)
      return loop.emitError(
          "direct_dte_acceptance: DTE execution stream requires a "
          "representable static loop trip count");
    for (uint64_t iteration = 0; iteration < *tripCount; ++iteration) {
      loopIterations[loop.getOperation()] = iteration;
      for (mlir::Operation &nested : *loop.getBody())
        if (mlir::failed(appendDynamicIssues(&nested, issues, issueIndices,
                                             issueAncestors, loopIterations,
                                             occurrenceCounts, streams)))
          return mlir::failure();
    }
    loopIterations.erase(loop.getOperation());
    return mlir::success();
  }

  for (mlir::Region &region : operation->getRegions()) {
    if (region.empty())
      continue;
    if (!region.hasOneBlock())
      return operation->emitError(
          "direct_dte_acceptance: DTE execution stream requires single-block "
          "structured control");
    for (mlir::Operation &nested : region.front())
      if (mlir::failed(appendDynamicIssues(&nested, issues, issueIndices,
                                           issueAncestors, loopIterations,
                                           occurrenceCounts, streams)))
        return mlir::failure();
  }
  return mlir::success();
}

static mlir::LogicalResult buildDynamicMessageStreams(
    llvm::ArrayRef<mlir::ModuleOp> physicalTileModules,
    llvm::ArrayRef<IssueRecord> issues,
    std::map<MessageBaseKey, DynamicMessageStream> &streams) {
  llvm::DenseMap<mlir::Operation *, unsigned> issueIndices;
  llvm::DenseSet<mlir::Operation *> issueAncestors;
  for (auto [index, issue] : llvm::enumerate(issues)) {
    issueIndices.try_emplace(issue.operation, static_cast<unsigned>(index));
    for (mlir::Operation *ancestor = issue.operation->getParentOp(); ancestor;
         ancestor = ancestor->getParentOp())
      issueAncestors.insert(ancestor);
  }

  llvm::DenseMap<mlir::Operation *, uint64_t> loopIterations;
  llvm::SmallVector<uint64_t, 32> occurrenceCounts(issues.size(), 0);
  for (mlir::ModuleOp module : physicalTileModules)
    if (mlir::failed(appendDynamicIssues(
            module.getOperation(), issues, issueIndices, issueAncestors,
            loopIterations, occurrenceCounts, streams)))
      return mlir::failure();
  return mlir::success();
}

/// A structured transport proof has already established that every static
/// issue is reached through the same non-empty call/region/loop occurrence on
/// its peer and that no endpoint crosses a loop backedge. When every accepted
/// endpoint has one invariant physical range, repeating that occurrence cannot
/// change route selection. Keep one representative per static issue instead of
/// materializing one DynamicIssue for every workload loop iteration.
static bool buildInvariantMessageRepresentatives(
    llvm::ArrayRef<IssueRecord> issues,
    std::map<MessageBaseKey, DynamicMessageStream> &streams) {
  if (llvm::any_of(issues, [](const IssueRecord &issue) {
        return issue.rangePattern.ranges.size() != 1;
      }))
    return false;

  for (auto [index, issue] : llvm::enumerate(issues)) {
    DynamicMessageStream &stream = streams[issue.message];
    auto &instances = issue.isSend ? stream.sends : stream.recvs;
    instances.push_back(DynamicIssue{static_cast<unsigned>(index),
                                     issue.rangePattern.ranges.front(),
                                     /*selector=*/0});
  }
  return true;
}

static mlir::LogicalResult
verifySenderResources(llvm::ArrayRef<IssueRecord> issues) {
  for (size_t leftIndex = 0; leftIndex < issues.size(); ++leftIndex) {
    const IssueRecord &left = issues[leftIndex];
    if (!left.isSend)
      continue;
    for (size_t rightIndex = leftIndex + 1; rightIndex < issues.size();
         ++rightIndex) {
      const IssueRecord &right = issues[rightIndex];
      if (!right.isSend || left.block != right.block)
        continue;
      bool overlaps = left.issueIndex < right.waitIndex &&
                      right.issueIndex < left.waitIndex;
      if (overlaps)
        return right.operation->emitError(
            "direct_dte_acceptance: normal allocation profile permits at "
            "most one live sender per physical Tile block");
    }
  }
  return mlir::success();
}

static bool operationContainsTransport(
    mlir::Operation *operation,
    const llvm::DenseSet<mlir::Operation *> &transportFunctions,
    mlir::ModuleOp module) {
  bool containsTransport = false;
  operation->walk([&](mlir::Operation *nested) {
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(nested)) {
      containsTransport = true;
      return mlir::WalkResult::interrupt();
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(nested)) {
      mlir::func::FuncOp callee =
          module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (callee && transportFunctions.contains(callee.getOperation())) {
        containsTransport = true;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  return containsTransport;
}

static llvm::DenseSet<mlir::Operation *>
findTransportFunctions(const AcceptedCallClosure &closure,
                       mlir::ModuleOp module) {
  llvm::DenseSet<mlir::Operation *> transportFunctions;
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::func::FuncOp function : closure.functions) {
      if (transportFunctions.contains(function.getOperation()))
        continue;
      bool hasTransport = false;
      function.walk([&](mlir::Operation *operation) {
        if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(
                operation)) {
          hasTransport = true;
          return mlir::WalkResult::interrupt();
        }
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
          mlir::func::FuncOp callee =
              module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
          if (callee && transportFunctions.contains(callee.getOperation())) {
            hasTransport = true;
            return mlir::WalkResult::interrupt();
          }
        }
        return mlir::WalkResult::advance();
      });
      if (hasTransport) {
        transportFunctions.insert(function.getOperation());
        changed = true;
      }
    }
  }
  return transportFunctions;
}

class StructuredTraceBuilder {
public:
  StructuredTraceBuilder(
      int64_t physicalTileIndex, mlir::ModuleOp module,
      const AcceptedCallClosure &closure,
      const llvm::DenseSet<mlir::Operation *> &transportFunctions,
      const llvm::DenseMap<mlir::Operation *, IssueRecord *> &issueByOperation,
      StructuredTransportTrace &trace)
      : physicalTileIndex(physicalTileIndex), module(module), closure(closure),
        transportFunctions(transportFunctions),
        issueByOperation(issueByOperation), trace(trace) {}

  mlir::LogicalResult build() {
    if (mlir::failed(traceFunction(closure.entry)))
      return mlir::failure();
    if (!pending.empty()) {
      mlir::func::FuncOp entry = closure.entry;
      return entry.emitError(
          "direct_dte_acceptance: structured transport trace ended with live "
          "issue occurrences");
    }
    return mlir::success();
  }

private:
  unsigned appendAction(TransportAction action) {
    unsigned index = static_cast<unsigned>(trace.actions.size());
    trace.actions.push_back(std::move(action));
    trace.dependencies.emplace_back();
    if (previousAction)
      trace.dependencies.back().push_back(*previousAction);
    previousAction = index;
    return index;
  }

  mlir::LogicalResult traceFunction(mlir::func::FuncOp function) {
    if (!function.getBody().hasOneBlock())
      return function.emitError(
          "direct_dte_acceptance: DTE-bearing call occurrence requires a "
          "single-block function body");
    if (!activeFunctions.insert(function.getOperation()).second)
      return function.emitError(
          "direct_dte_acceptance: recursive DTE call occurrence is "
          "unsupported");
    mlir::LogicalResult result = traceBlock(function.getBody().front());
    activeFunctions.erase(function.getOperation());
    return result;
  }

  mlir::LogicalResult traceBlock(mlir::Block &block) {
    for (mlir::Operation &operation : block)
      if (mlir::failed(traceOperation(&operation)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult traceIssue(mlir::Operation *operation) {
    auto recordIt = issueByOperation.find(operation);
    if (recordIt == issueByOperation.end())
      return operation->emitError(
          "direct_dte_acceptance: structured trace cannot resolve a DTE "
          "issue occurrence");
    IssueRecord *record = recordIt->second;
    if (pending.contains(record))
      return operation->emitError(
          "direct_dte_acceptance: one static DTE issue has overlapping dynamic "
          "occurrences");

    if (record->isSend) {
      for (const auto &[liveRecord, action] : pending) {
        (void)action;
        if (liveRecord->isSend)
          return operation->emitError(
              "direct_dte_acceptance: normal allocation profile permits at "
              "most one live sender per physical Tile structured occurrence");
      }
    } else {
      for (const auto &[liveRecord, action] : pending) {
        (void)action;
        if (!liveRecord->isSend)
          trace.receiverConflicts.push_back({record, liveRecord});
      }
    }

    unsigned action = appendAction(
        TransportAction{operation,
                        physicalTileIndex,
                        record->isSend ? TransportActionKind::SendIssue
                                       : TransportActionKind::ReceivePrepare,
                        record,
                        {},
                        {occurrencePath.begin(), occurrencePath.end()}});
    pending[record] = action;
    return mlir::success();
  }

  mlir::LogicalResult traceWait(InstrDTEWaitOp wait) {
    llvm::SmallVector<unsigned, 4> waitedActions;
    for (mlir::Value token : wait.getTokens()) {
      mlir::Operation *definition = token.getDefiningOp();
      auto recordIt = issueByOperation.find(definition);
      if (recordIt == issueByOperation.end())
        return wait.emitError(
            "direct_dte_acceptance: structured trace cannot resolve a DTE "
            "wait occurrence");
      auto pendingIt = pending.find(recordIt->second);
      if (pendingIt == pending.end())
        return wait.emitError(
            "direct_dte_acceptance: DTE wait has no live issue in this "
            "structured occurrence");
      waitedActions.push_back(pendingIt->second);
      pending.erase(pendingIt);
    }
    appendAction(TransportAction{wait.getOperation(), physicalTileIndex,
                                 TransportActionKind::Wait, nullptr,
                                 std::move(waitedActions)});
    return mlir::success();
  }

  mlir::LogicalResult traceLoop(mlir::scf::ForOp loop) {
    std::optional<StaticLoopBounds> bounds = getStaticLoopBounds(loop);
    if (!bounds)
      return loop.emitError(
          "direct_dte_acceptance: DTE-bearing caller loop requires constant "
          "bounds and a positive constant step");
    if (bounds->lower >= bounds->upper)
      return mlir::success();

    // Every accepted issue has an exact later wait in the same block. Hence no
    // Direct-DTE endpoint crosses a loop backedge, and one symbolic iteration
    // is a complete proof of every identical steady-state occurrence.
    const size_t pendingBefore = pending.size();
    occurrencePath.push_back(StructuredExecutionFrame{
        StructuredExecutionFrameKind::Loop, bounds->lower, bounds->upper,
        bounds->step, getStructuredLoopSiblingOrdinal(loop)});
    mlir::LogicalResult result = traceBlock(loop.getRegion().front());
    occurrencePath.pop_back();
    if (mlir::failed(result))
      return mlir::failure();
    if (pending.size() != pendingBefore)
      return loop.emitError(
          "direct_dte_acceptance: DTE endpoint crosses a structured loop "
          "backedge");
    return mlir::success();
  }

  mlir::LogicalResult traceOperation(mlir::Operation *operation) {
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation))
      return traceIssue(operation);
    if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation))
      return traceWait(wait);

    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
      mlir::func::FuncOp callee =
          module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (callee && transportFunctions.contains(callee.getOperation())) {
        occurrencePath.push_back(StructuredExecutionFrame{
            StructuredExecutionFrameKind::Call, 0, 0, 0,
            getStructuredSiblingOrdinal(operation,
                                        StructuredExecutionFrameKind::Call)});
        mlir::LogicalResult result = traceFunction(callee);
        occurrencePath.pop_back();
        return result;
      }
      return mlir::success();
    }

    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      if (operationContainsTransport(operation, transportFunctions, module))
        return traceLoop(loop);
      return mlir::success();
    }

    if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(operation)) {
      if (!operationContainsTransport(operation, transportFunctions, module))
        return mlir::success();
      if (!tileRegion.getBody().hasOneBlock())
        return operation->emitError(
            "direct_dte_acceptance: DTE-bearing tile region requires one "
            "structured block");
      occurrencePath.push_back(StructuredExecutionFrame{
          StructuredExecutionFrameKind::TileRegion, 0, 0, 0,
          getStructuredSiblingOrdinal(
              operation, StructuredExecutionFrameKind::TileRegion)});
      mlir::LogicalResult result = traceBlock(tileRegion.getBody().front());
      occurrencePath.pop_back();
      return result;
    }

    if (operation->getNumRegions() != 0 &&
        operationContainsTransport(operation, transportFunctions, module))
      return operation->emitError()
             << "direct_dte_acceptance: DTE-bearing occurrence under "
             << operation->getName()
             << " is not statically executable structured control";
    return mlir::success();
  }

  int64_t getStructuredSiblingOrdinal(mlir::Operation *operation,
                                      StructuredExecutionFrameKind kind) const {
    int64_t ordinal = 0;
    for (mlir::Operation &sibling : *operation->getBlock()) {
      if (&sibling == operation)
        return ordinal;
      bool sameKind = (kind == StructuredExecutionFrameKind::Call &&
                       mlir::isa<mlir::func::CallOp>(sibling)) ||
                      (kind == StructuredExecutionFrameKind::TileRegion &&
                       mlir::isa<TileRegionOp>(sibling));
      if (sameKind)
        ++ordinal;
    }
    llvm_unreachable("structured operation must be in its parent block");
  }

  int64_t physicalTileIndex;
  mlir::ModuleOp module;
  const AcceptedCallClosure &closure;
  const llvm::DenseSet<mlir::Operation *> &transportFunctions;
  const llvm::DenseMap<mlir::Operation *, IssueRecord *> &issueByOperation;
  StructuredTransportTrace &trace;
  llvm::SmallVector<StructuredExecutionFrame, 4> occurrencePath;
  llvm::DenseMap<IssueRecord *, unsigned> pending;
  llvm::DenseSet<mlir::Operation *> activeFunctions;
  std::optional<unsigned> previousAction;
};

static void addDependency(StructuredTransportTrace &trace, unsigned action,
                          unsigned dependency) {
  llvm::SmallVectorImpl<unsigned> &dependencies = trace.dependencies[action];
  if (!llvm::is_contained(dependencies, dependency))
    dependencies.push_back(dependency);
}

static llvm::StringRef getActionName(TransportActionKind kind) {
  switch (kind) {
  case TransportActionKind::SendIssue:
    return "send issue";
  case TransportActionKind::ReceivePrepare:
    return "receive prepare";
  case TransportActionKind::Wait:
    return "wait";
  }
  llvm_unreachable("unknown transport action kind");
}

static mlir::FailureOr<mlir::Value>
materializeRouteSelector(mlir::Operation *issue) {
  llvm::SmallVector<mlir::scf::ForOp, 4> loops;
  for (mlir::Operation *parent = issue->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
      loops.push_back(loop);
  std::reverse(loops.begin(), loops.end());
  if (loops.empty())
    return issue->emitError(
        "direct_dte_acceptance: selector-table binding requires an enclosing "
        "static loop");

  mlir::OpBuilder builder(issue);
  mlir::Location location = issue->getLoc();
  mlir::Value ordinal =
      builder.create<mlir::arith::ConstantIndexOp>(location, 0);
  for (mlir::scf::ForOp loop : loops) {
    std::optional<uint64_t> tripCount = getStaticTripCount(loop);
    if (!tripCount || *tripCount == 0 ||
        *tripCount > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return issue->emitError(
          "direct_dte_acceptance: route selector loop trip count is not "
          "representable");
    mlir::Value relative = builder.create<mlir::arith::SubIOp>(
        location, loop.getInductionVar(), loop.getLowerBound());
    mlir::Value iteration = builder.create<mlir::arith::DivUIOp>(
        location, relative, loop.getStep());
    ordinal = builder.create<mlir::arith::MulIOp>(
        location, ordinal,
        builder.create<mlir::arith::ConstantIndexOp>(
            location, static_cast<int64_t>(*tripCount)));
    ordinal = builder.create<mlir::arith::AddIOp>(location, ordinal, iteration);
  }
  return builder
      .create<mlir::arith::IndexCastOp>(location, builder.getI64Type(), ordinal)
      .getResult();
}

static mlir::LogicalResult matchDynamicMessages(
    llvm::ArrayRef<IssueRecord> issues,
    std::map<MessageBaseKey, DynamicMessageStream> &streams,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, DirectDTEBindingAttr>>
        &acceptedBindings) {
  struct SendRoute {
    uint64_t selector = 0;
    int64_t remoteAddress = -1;
    int64_t receiverFsmId = -1;
    int64_t sourceRelativeDisplacement = 0;
  };
  llvm::DenseMap<unsigned, llvm::SmallVector<SendRoute, 8>> sendRoutes;
  llvm::SmallVector<bool, 32> observed(issues.size(), false);

  for (auto &entry : streams) {
    DynamicMessageStream &stream = entry.second;
    if (stream.sends.size() != stream.recvs.size()) {
      const DynamicIssue *instance =
          !stream.sends.empty()
              ? &stream.sends.front()
              : (!stream.recvs.empty() ? &stream.recvs.front() : nullptr);
      if (!instance)
        continue;
      return issues[instance->issue].operation->emitError()
             << "direct_dte_acceptance: dynamic send/receive counts differ "
                "(source="
             << std::get<0>(entry.first)
             << ", destination=" << std::get<1>(entry.first)
             << ", communication_id=" << std::get<2>(entry.first)
             << ", round=" << std::get<3>(entry.first)
             << ", payload_slice=" << std::get<4>(entry.first)
             << ", sends=" << stream.sends.size()
             << ", receives=" << stream.recvs.size() << ")";
    }

    for (auto [sendInstance, recvInstance] :
         llvm::zip(stream.sends, stream.recvs)) {
      const IssueRecord &send = issues[sendInstance.issue];
      const IssueRecord &recv = issues[recvInstance.issue];
      if (send.bytes != recv.bytes)
        return recv.operation->emitError(
            "direct_dte_acceptance: dynamically matched send and receive byte "
            "counts differ");
      __int128 displacement = static_cast<__int128>(recvInstance.range.start) -
                              static_cast<__int128>(sendInstance.range.start);
      if (displacement <
              static_cast<__int128>(std::numeric_limits<int64_t>::min()) ||
          displacement >
              static_cast<__int128>(std::numeric_limits<int64_t>::max()))
        return send.operation->emitError(
            "direct_dte_acceptance: source-relative remote address "
            "displacement is not representable");
      observed[sendInstance.issue] = true;
      observed[recvInstance.issue] = true;
      sendRoutes[sendInstance.issue].push_back(
          SendRoute{sendInstance.selector, recvInstance.range.start,
                    recv.receiverFsmId, static_cast<int64_t>(displacement)});
    }
  }

  for (auto [index, issue] : llvm::enumerate(issues)) {
    unsigned issueIndex = static_cast<unsigned>(index);
    if (!observed[issueIndex])
      return issue.operation->emitError(
          "direct_dte_acceptance: static DTE site has no dynamic peer "
          "execution");

    mlir::MLIRContext *context = issue.operation->getContext();
    mlir::DenseI64ArrayAttr emptyTable =
        mlir::DenseI64ArrayAttr::get(context, {});
    if (!issue.isSend) {
      acceptedBindings.push_back(
          {issue.operation,
           DirectDTEBindingAttr::get(
               context, DTEAllocationProfile::Normal, issue.receiverFsmId,
               DTERemoteAddressMode::Absolute,
               issue.rangePattern.ranges.front().start, emptyTable,
               DTECompletionProfile::SenderWaitReceiverFSM)});
      continue;
    }

    auto routes = sendRoutes.find(issueIndex);
    if (routes == sendRoutes.end() || routes->second.empty())
      return issue.operation->emitError(
          "direct_dte_acceptance: send site has no accepted dynamic route");
    std::map<uint64_t, std::pair<int64_t, int64_t>> bySelector;
    bool absoluteStable = true;
    bool fsmStable = true;
    bool displacementStable = true;
    int64_t firstRemote = routes->second.front().remoteAddress;
    int64_t firstFsm = routes->second.front().receiverFsmId;
    int64_t firstDisplacement =
        routes->second.front().sourceRelativeDisplacement;
    for (const SendRoute &route : routes->second) {
      auto [slot, inserted] = bySelector.try_emplace(
          route.selector,
          std::make_pair(route.remoteAddress, route.receiverFsmId));
      if (!inserted && slot->second != std::make_pair(route.remoteAddress,
                                                      route.receiverFsmId))
        return issue.operation->emitError(
            "direct_dte_acceptance: one route selector would require "
            "different remote address or receiver FSM bindings");
      absoluteStable &= route.remoteAddress == firstRemote;
      fsmStable &= route.receiverFsmId == firstFsm;
      displacementStable &=
          route.sourceRelativeDisplacement == firstDisplacement;
    }

    DTERemoteAddressMode mode = DTERemoteAddressMode::Absolute;
    int64_t address = firstRemote;
    mlir::DenseI64ArrayAttr table = emptyTable;
    if (absoluteStable && fsmStable) {
      mode = DTERemoteAddressMode::Absolute;
    } else if (displacementStable && fsmStable) {
      mode = DTERemoteAddressMode::SourceRelative;
      address = firstDisplacement;
    } else {
      llvm::SmallVector<std::pair<int64_t, int64_t>, 8> orderedRoutes;
      orderedRoutes.reserve(bySelector.size());
      uint64_t expectedSelector = 0;
      for (const auto &[selector, route] : bySelector) {
        if (selector != expectedSelector++)
          return issue.operation->emitError(
              "direct_dte_acceptance: dynamic route selectors are not "
              "contiguous");
        orderedRoutes.push_back(route);
      }
      size_t period = 1;
      for (; period < orderedRoutes.size(); ++period) {
        bool repeats = true;
        for (size_t index = period; index < orderedRoutes.size(); ++index)
          if (orderedRoutes[index] != orderedRoutes[index % period]) {
            repeats = false;
            break;
          }
        if (repeats)
          break;
      }
      llvm::SmallVector<int64_t, 12> flattened;
      flattened.reserve(period * 3);
      for (size_t selector = 0; selector < period; ++selector) {
        flattened.push_back(static_cast<int64_t>(selector));
        flattened.push_back(orderedRoutes[selector].first);
        flattened.push_back(orderedRoutes[selector].second);
      }
      table = mlir::DenseI64ArrayAttr::get(context, flattened);
      mode = DTERemoteAddressMode::SelectorTable;
    }
    acceptedBindings.push_back(
        {issue.operation,
         DirectDTEBindingAttr::get(
             context, DTEAllocationProfile::Normal, firstFsm, mode, address,
             table, DTECompletionProfile::SenderWaitReceiverFSM)});
  }
  for (auto &[operation, binding] : acceptedBindings) {
    if (binding.getRemoteAddressMode() != DTERemoteAddressMode::SelectorTable)
      continue;
    mlir::FailureOr<mlir::Value> selector = materializeRouteSelector(operation);
    if (mlir::failed(selector))
      return mlir::failure();
    operation->insertOperands(1, *selector);
  }
  return mlir::success();
}

static mlir::LogicalResult
verifyAcyclicWaitGraph(StructuredTransportTrace &trace) {
  llvm::SmallVector<uint8_t, 64> state(trace.actions.size(), 0);
  llvm::SmallVector<unsigned, 64> stack;
  llvm::SmallVector<int64_t, 64> stackPosition(trace.actions.size(), -1);
  llvm::SmallVector<unsigned, 16> cycle;

  std::function<bool(unsigned)> visit = [&](unsigned action) {
    state[action] = 1;
    stackPosition[action] = static_cast<int64_t>(stack.size());
    stack.push_back(action);
    for (unsigned dependency : trace.dependencies[action]) {
      if (state[dependency] == 0) {
        if (visit(dependency))
          return true;
        continue;
      }
      if (state[dependency] == 1) {
        cycle.append(stack.begin() + stackPosition[dependency], stack.end());
        cycle.push_back(dependency);
        return true;
      }
    }
    stack.pop_back();
    stackPosition[action] = -1;
    state[action] = 2;
    return false;
  };

  for (unsigned action = 0; action < trace.actions.size(); ++action)
    if (state[action] == 0 && visit(action)) {
      TransportAction &anchor = trace.actions[cycle.front()];
      mlir::InFlightDiagnostic diagnostic = anchor.operation->emitError(
          "direct_dte_acceptance: whole-program structured transport "
          "wait graph contains a cyclic dependency");
      const size_t noteCount = std::min<size_t>(cycle.size(), 16);
      for (size_t index = 0; index < noteCount; ++index) {
        const unsigned actionIndex = cycle[index];
        const TransportAction &member = trace.actions[actionIndex];
        diagnostic << (index == 0 ? "; cycle=[" : ", ") << actionIndex
                   << ":tile" << member.physicalTileIndex << ':'
                   << getActionName(member.kind);
        if (member.issue) {
          const auto &[source, destination, communication, round, payload] =
              member.issue->message;
          diagnostic << "(" << source << "->" << destination
                     << ",comm=" << communication << ",round=" << round
                     << ",slice=" << payload << ')';
        }
        diagnostic.attachNote(member.operation->getLoc())
            << "physical Tile index " << member.physicalTileIndex << " "
            << getActionName(member.kind) << " participates in the wait cycle";
      }
      diagnostic << ']';
      return mlir::failure();
    }
  return mlir::success();
}

static mlir::LogicalResult verifyStructuredTransportWaitGraph(
    llvm::ArrayRef<mlir::ModuleOp> physicalTileModules,
    llvm::SmallVectorImpl<IssueRecord> &issues,
    llvm::SmallVectorImpl<MatchedMessage> &messages,
    StructuredTransportTrace &trace) {
  llvm::DenseMap<mlir::Operation *, IssueRecord *> issueByOperation;
  for (IssueRecord &issue : issues)
    issueByOperation[issue.operation] = &issue;

  for (size_t physicalTileIndex = 0;
       physicalTileIndex < physicalTileModules.size(); ++physicalTileIndex) {
    mlir::ModuleOp module = physicalTileModules[physicalTileIndex];
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(module);
    if (!closure) {
      std::string error = llvm::toString(closure.takeError());
      module.emitError()
          << "direct_dte_acceptance: DTE call occurrence is not statically "
             "provable: "
          << error;
      return mlir::failure();
    }
    llvm::DenseSet<mlir::Operation *> transportFunctions =
        findTransportFunctions(*closure, module);
    StructuredTraceBuilder builder(static_cast<int64_t>(physicalTileIndex),
                                   module, *closure, transportFunctions,
                                   issueByOperation, trace);
    if (mlir::failed(builder.build()))
      return mlir::failure();
  }

  struct EndpointOccurrences {
    llvm::SmallVector<unsigned, 4> sends;
    llvm::SmallVector<unsigned, 4> receives;
  };
  std::map<MessageBaseKey, EndpointOccurrences> dynamicMessages;
  llvm::DenseSet<IssueRecord *> executedIssues;
  for (auto [actionIndex, action] : llvm::enumerate(trace.actions)) {
    if (!action.issue)
      continue;
    EndpointOccurrences &occurrences = dynamicMessages[action.issue->message];
    (action.issue->isSend ? occurrences.sends : occurrences.receives)
        .push_back(static_cast<unsigned>(actionIndex));
    executedIssues.insert(action.issue);
  }
  for (IssueRecord &issue : issues)
    if (!executedIssues.contains(&issue))
      return issue.operation->emitError(
          "direct_dte_acceptance: DTE issue has no executable structured "
          "occurrence; zero-trip or unreachable transport is unsupported");

  llvm::DenseMap<unsigned, unsigned> matchingSend;
  llvm::DenseMap<unsigned, unsigned> matchingReceive;
  llvm::DenseMap<IssueRecord *, IssueRecord *> matchedPeer;
  for (auto &[message, occurrences] : dynamicMessages) {
    (void)message;
    if (occurrences.sends.size() != occurrences.receives.size()) {
      unsigned anchor = occurrences.sends.empty() ? occurrences.receives.front()
                                                  : occurrences.sends.front();
      return trace.actions[anchor].operation->emitError(
          "direct_dte_acceptance: message identity has different executable "
          "send and receive occurrence counts");
    }
    llvm::SmallVector<bool, 4> matchedReceives(occurrences.receives.size(),
                                               false);
    llvm::SmallVector<std::pair<unsigned, unsigned>, 4> matchedOccurrences;
    matchedOccurrences.reserve(occurrences.sends.size());
    for (unsigned sendIndex : occurrences.sends) {
      auto indexedReceives = llvm::enumerate(occurrences.receives);
      auto recvPosition =
          llvm::find_if(indexedReceives, [&](auto indexedReceive) {
            return !matchedReceives[indexedReceive.index()] &&
                   haveSameDynamicOccurrencePath(
                       trace.actions[sendIndex].occurrencePath,
                       trace.actions[indexedReceive.value()].occurrencePath);
          });
      if (recvPosition == indexedReceives.end())
        return trace.actions[sendIndex].operation->emitError(
            "direct_dte_acceptance: matched message call/loop "
            "occurrence paths are not structurally identical across physical "
          "Tiles");
      auto indexedReceive = *recvPosition;
      matchedReceives[indexedReceive.index()] = true;
      matchedOccurrences.emplace_back(sendIndex, indexedReceive.value());
    }
    for (auto [sendIndex, recvIndex] : matchedOccurrences) {
      TransportAction &send = trace.actions[sendIndex];
      TransportAction &recv = trace.actions[recvIndex];
      if (send.issue->bytes != recv.issue->bytes)
        return recv.operation->emitError(
            "direct_dte_acceptance: matched send and receive byte counts "
            "differ");
      auto sendPeer = matchedPeer.find(send.issue);
      if (sendPeer != matchedPeer.end() && sendPeer->second != recv.issue)
        return send.operation->emitError(
            "direct_dte_acceptance: one static DTE site would require "
            "different physical bindings across call occurrences");
      matchedPeer.try_emplace(send.issue, recv.issue);
      auto recvPeer = matchedPeer.find(recv.issue);
      if (recvPeer != matchedPeer.end() && recvPeer->second != send.issue)
        return recv.operation->emitError(
            "direct_dte_acceptance: one static DTE site would require "
            "different physical bindings across call occurrences");
      matchedPeer.try_emplace(recv.issue, send.issue);
      matchingSend[sendIndex] = sendIndex;
      matchingSend[recvIndex] = sendIndex;
      matchingReceive[sendIndex] = recvIndex;
      matchingReceive[recvIndex] = recvIndex;
    }
  }

  for (IssueRecord &issue : issues) {
    if (!issue.isSend)
      continue;
    auto peer = matchedPeer.find(&issue);
    if (peer == matchedPeer.end())
      return issue.operation->emitError(
          "direct_dte_acceptance: message identity has no matching peer "
          "issue");
    messages.push_back(MatchedMessage{&issue, peer->second});
  }

  for (auto [actionIndex, action] : llvm::enumerate(trace.actions)) {
    if (action.kind != TransportActionKind::Wait)
      continue;
    for (unsigned waitedIssue : action.waitedIssueActions) {
      auto sendIt = matchingSend.find(waitedIssue);
      if (sendIt == matchingSend.end())
        return action.operation->emitError(
            "direct_dte_acceptance: wait occurrence has no matched dynamic "
            "message issue");
      addDependency(trace, static_cast<unsigned>(actionIndex), sendIt->second);
      auto receiveIt = matchingReceive.find(waitedIssue);
      if (receiveIt == matchingReceive.end())
        return action.operation->emitError(
            "direct_dte_acceptance: wait occurrence has no matched receive "
            "preparation");
      // Programming a sender does not require the remote receiver FSM to have
      // executed already; completion does. Keep the issue independently
      // schedulable, then prove every send/receive wait occurs after both the
      // local issue and the matched receive preparation. This distinction is
      // essential when exact DDR cuts place the two endpoint preparations in
      // different sequential TileRegions.
      addDependency(trace, static_cast<unsigned>(actionIndex),
                    receiveIt->second);
    }
  }
  return verifyAcyclicWaitGraph(trace);
}

static mlir::LogicalResult allocateReceiverFSMs(
    llvm::SmallVectorImpl<IssueRecord> &issues,
    llvm::ArrayRef<std::pair<IssueRecord *, IssueRecord *>> conflicts) {
  llvm::SmallVector<IssueRecord *, 16> receivers;
  llvm::DenseMap<IssueRecord *, unsigned> receiverIndex;
  for (IssueRecord &issue : issues) {
    if (issue.isSend)
      continue;
    receiverIndex[&issue] = static_cast<unsigned>(receivers.size());
    receivers.push_back(&issue);
  }

  std::vector<std::vector<bool>> adjacency(
      receivers.size(), std::vector<bool>(receivers.size(), false));
  for (auto [left, right] : conflicts) {
    unsigned leftIndex = receiverIndex.lookup(left);
    unsigned rightIndex = receiverIndex.lookup(right);
    if (leftIndex == rightIndex)
      return left->operation->emitError(
          "direct_dte_acceptance: one receiver has overlapping dynamic "
          "occurrences");
    adjacency[leftIndex][rightIndex] = true;
    adjacency[rightIndex][leftIndex] = true;
  }

  llvm::SmallVector<int64_t, 16> colors(receivers.size(), -1);
  std::function<bool(unsigned)> color = [&](unsigned index) {
    if (index == receivers.size())
      return true;
    for (int64_t candidate = 0; candidate < 4; ++candidate) {
      bool available = true;
      for (unsigned other = 0; other < index; ++other)
        if (adjacency[index][other] && colors[other] == candidate) {
          available = false;
          break;
        }
      if (!available)
        continue;
      colors[index] = candidate;
      if (color(index + 1))
        return true;
      colors[index] = -1;
    }
    return false;
  };
  if (!color(0)) {
    mlir::Operation *anchor =
        receivers.empty() ? nullptr : receivers.back()->operation;
    if (anchor)
      return anchor->emitError(
          "direct_dte_acceptance: more than four receiver FSM live ranges "
          "overlap in the structured whole-program trace");
    return mlir::failure();
  }
  for (auto [index, receiver] : llvm::enumerate(receivers))
    receiver->receiverFsmId = colors[index];
  return mlir::success();
}

static mlir::LogicalResult
analyzeDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules,
                          llvm::SmallVectorImpl<IssueRecord> &issues,
                          llvm::SmallVectorImpl<MatchedMessage> &messages,
                          StructuredTransportTrace &trace) {
  if (mlir::failed(collectIssues(physicalTileModules, issues)))
    return mlir::failure();
  if (issues.empty())
    return mlir::success();
  if (mlir::failed(verifySenderResources(issues)) ||
      mlir::failed(verifyStructuredTransportWaitGraph(
          physicalTileModules, issues, messages, trace)) ||
      mlir::failed(allocateReceiverFSMs(issues, trace.receiverConflicts)))
    return mlir::failure();
  return mlir::success();
}

} // namespace

mlir::LogicalResult verifyDirectDTETransportSchedule(
    llvm::ArrayRef<mlir::ModuleOp> physicalTileModules) {
  llvm::SmallVector<IssueRecord, 32> issues;
  llvm::SmallVector<MatchedMessage, 32> messages;
  StructuredTransportTrace trace;
  return analyzeDirectDTETransport(physicalTileModules, issues, messages,
                                   trace);
}

mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules) {
  llvm::SmallVector<IssueRecord, 32> issues;
  llvm::SmallVector<MatchedMessage, 32> messages;
  StructuredTransportTrace trace;
  if (mlir::failed(analyzeDirectDTETransport(physicalTileModules, issues,
                                             messages, trace)))
    return mlir::failure();
  if (issues.empty())
    return TransportContract::None;

  std::map<MessageBaseKey, DynamicMessageStream> streams;
  if (!buildInvariantMessageRepresentatives(issues, streams) &&
      mlir::failed(
          buildDynamicMessageStreams(physicalTileModules, issues, streams)))
    return mlir::failure();

  llvm::SmallVector<std::pair<mlir::Operation *, DirectDTEBindingAttr>, 32>
      acceptedBindings;
  if (mlir::failed(matchDynamicMessages(issues, streams, acceptedBindings)))
    return mlir::failure();
  for (auto &[operation, binding] : acceptedBindings) {
    if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation)) {
      send.setBindingAttr(binding);
      continue;
    }
    mlir::cast<InstrDTERecvOp>(operation).setBindingAttr(binding);
  }
  return TransportContract::DirectDTE;
}

} // namespace wafer::compiler::detail

namespace wafer::compiler::testing {

mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules) {
  return detail::acceptDirectDTETransport(physicalTileModules);
}

} // namespace wafer::compiler::testing
