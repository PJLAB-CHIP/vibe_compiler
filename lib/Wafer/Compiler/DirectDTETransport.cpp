//===- DirectDTETransport.cpp - Physical Direct DTE acceptance ----------===//

#include "DirectDTETransport.h"

#include "AcceptedCallClosure.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"

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

using MessageBaseKey =
    std::tuple<int64_t, int64_t, int64_t, DTEProtocolPhase, int64_t, int64_t>;

struct PhysicalRange {
  int64_t start = -1;
  int64_t end = -1;
};

struct IssueRecord {
  mlir::Operation *operation = nullptr;
  mlir::Operation *wait = nullptr;
  mlir::Block *block = nullptr;
  int64_t rank = -1;
  MessageBaseKey message;
  PhysicalRange range;
  int64_t bytes = -1;
  unsigned issueIndex = 0;
  unsigned waitIndex = 0;
  int64_t receiverFsmId = -1;
  bool isSend = false;
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
  int64_t rank = -1;
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
    mlir::Block *owner = blockArg.getOwner();
    auto tileRegion =
        owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
              : TileRegionOp();
    if (!tileRegion || tileRegion.getBody().empty() ||
        owner != &tileRegion.getBody().front() ||
        blockArg.getArgNumber() >= tileRegion.getInputs().size())
      return value;
    value = tileRegion.getInputs()[blockArg.getArgNumber()];
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

static mlir::FailureOr<PhysicalRange>
resolveAcceptedRange(mlir::Operation *op, mlir::Value buffer, int64_t bytes) {
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

  mlir::Value root = getRootViewSource(buffer);
  auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
  auto alloc = root.getDefiningOp<mlir::memref::AllocOp>();
  if (!rootType || !isWaferSPMMemRefType(rootType) || !alloc)
    return op->emitError(
        "direct_dte_acceptance: buffer must be rooted in a planned SPM "
        "allocation");
  auto acceptedOffset =
      alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
  if (!acceptedOffset)
    return op->emitError(
        "direct_dte_acceptance: SPM root is missing accepted wafer.spm.offset");
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
      !checkedAdd(acceptedOffset.getOffset(), rootInfo->physicalBytes, rootEnd))
    return op->emitError(
        "direct_dte_acceptance: accepted SPM range overflows int64");
  WaferTargetPolicy policy = getDefaultWaferTargetPolicy();
  if (start < policy.memory.spmBase || end > policy.memory.spmLimit ||
      end > rootEnd)
    return op->emitError(
        "direct_dte_acceptance: issue range is outside its accepted SPM "
        "allocation");
  return PhysicalRange{start, end};
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
  for (mlir::Operation *operation = issue->getNextNode(); operation != wait;
       operation = operation->getNextNode()) {
    if (!operation)
      llvm_unreachable("same-block ordered wait must be reachable from issue");

    bool hasRootOperand =
        llvm::any_of(operation->getOperands(), [&](mlir::Value operand) {
          return getRootViewSource(operand) == root;
        });
    if (operation->getNumRegions() != 0 && !mlir::isMemoryEffectFree(operation))
      return operation->emitError(
          "direct_dte_acceptance: effectful nested control is unsupported "
          "between a DTE issue and its matching wait");

    auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (!effects) {
      if (!mlir::isMemoryEffectFree(operation))
        return operation->emitError(
            "direct_dte_acceptance: issue buffer has an unknown access before "
            "its matching wait");
      continue;
    }

    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
    effects.getEffects(instances);
    bool conflicts = llvm::any_of(instances, [&](const auto &effect) {
      mlir::Value value = effect.getValue();
      if (value)
        return getRootViewSource(value) == root;
      return hasRootOperand;
    });
    if (conflicts)
      return operation->emitError(
          "direct_dte_acceptance: issue buffer must remain isolated until its "
          "matching wait");
  }
  return mlir::success();
}

static int64_t getStructuredLoopSiblingOrdinal(mlir::scf::ForOp loop) {
  // This is a path in the structured-control tree, not an ordinal assigned to
  // DTE issues.  It distinguishes sibling main/edge loop families while
  // leaving the typed DTE message identity unchanged.
  int64_t ordinal = 0;
  for (mlir::Operation &sibling : *loop->getBlock()) {
    if (&sibling == loop.getOperation())
      return ordinal;
    if (mlir::isa<mlir::scf::ForOp>(sibling))
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
      std::optional<int64_t> lower = mlir::getConstantIntValue(
          mlir::getAsOpFoldResult(loop.getLowerBound()));
      std::optional<int64_t> upper = mlir::getConstantIntValue(
          mlir::getAsOpFoldResult(loop.getUpperBound()));
      std::optional<int64_t> step =
          mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getStep()));
      if (!lower || !upper || !step || *step <= 0)
        return issue->emitError(
            "direct_dte_acceptance: enclosing scf.for requires constant "
            "bounds and a positive constant step");
      reversedLoops.push_back(StructuredLoopSite{
          *lower, *upper, *step, getStructuredLoopSiblingOrdinal(loop)});
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

static MessageBaseKey makeMessageBaseKey(int64_t rank, int64_t peer,
                                         DTEMessageAttr message, bool isSend) {
  return std::make_tuple(isSend ? rank : peer, isSend ? peer : rank,
                         message.getCommunicationId(), message.getPhase(),
                         message.getRound(), message.getPayloadSlice());
}

static mlir::LogicalResult
collectIssues(llvm::ArrayRef<mlir::ModuleOp> rankModules,
              llvm::SmallVectorImpl<IssueRecord> &issues) {
  for (size_t rankIndex = 0; rankIndex < rankModules.size(); ++rankIndex) {
    mlir::ModuleOp module = rankModules[rankIndex];
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
      if (operation->getAttr("binding")) {
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
      if (peer < 0 || peer >= static_cast<int64_t>(rankModules.size())) {
        operation->emitError(
            "direct_dte_acceptance: peer is outside the complete rank domain");
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      int64_t bytes =
          send ? send.getBytesAttr().getInt() : recv.getBytesAttr().getInt();
      DTEMessageAttr message =
          send ? send.getMessageAttr() : recv.getMessageAttr();
      MessageBaseKey messageBase =
          makeMessageBaseKey(static_cast<int64_t>(rankIndex), peer, message,
                             static_cast<bool>(send));
      mlir::FailureOr<llvm::SmallVector<StructuredLoopSite, 4>> loopSite =
          getStructuredLoopSite(operation);
      mlir::FailureOr<PhysicalRange> range =
          resolveAcceptedRange(operation, buffer, bytes);
      mlir::FailureOr<mlir::Operation *> wait =
          findUniqueSameBlockWait(operation, token, operationIndices);
      if (mlir::failed(loopSite) || mlir::failed(range) || mlir::failed(wait)) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      if (mlir::failed(verifyIssueBufferIsolation(operation, *wait, buffer))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      issues.push_back(IssueRecord{
          operation, *wait, operation->getBlock(),
          static_cast<int64_t>(rankIndex), messageBase, *range, bytes,
          operationIndices.lookup(operation), operationIndices.lookup(*wait),
          -1, static_cast<bool>(send)});
      return mlir::WalkResult::advance();
    });
    if (mlir::failed(result))
      return mlir::failure();
  }
  return mlir::success();
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
            "most one live sender per rank block");
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
      int64_t rank, mlir::ModuleOp module, const AcceptedCallClosure &closure,
      const llvm::DenseSet<mlir::Operation *> &transportFunctions,
      const llvm::DenseMap<mlir::Operation *, IssueRecord *> &issueByOperation,
      StructuredTransportTrace &trace)
      : rank(rank), module(module), closure(closure),
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
              "most one live sender per rank structured occurrence");
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
                        rank,
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
    appendAction(TransportAction{wait.getOperation(), rank,
                                 TransportActionKind::Wait, nullptr,
                                 std::move(waitedActions)});
    return mlir::success();
  }

  mlir::LogicalResult traceLoop(mlir::scf::ForOp loop) {
    std::optional<int64_t> lower = mlir::getConstantIntValue(
        mlir::getAsOpFoldResult(loop.getLowerBound()));
    std::optional<int64_t> upper = mlir::getConstantIntValue(
        mlir::getAsOpFoldResult(loop.getUpperBound()));
    std::optional<int64_t> step =
        mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getStep()));
    if (!lower || !upper || !step || *step <= 0)
      return loop.emitError(
          "direct_dte_acceptance: DTE-bearing caller loop requires constant "
          "bounds and a positive constant step");
    if (*lower >= *upper)
      return mlir::success();

    // Every accepted issue has an exact later wait in the same block. Hence no
    // Direct-DTE endpoint crosses a loop backedge, and one symbolic iteration
    // is a complete proof of every identical steady-state occurrence.
    const size_t pendingBefore = pending.size();
    occurrencePath.push_back(StructuredExecutionFrame{
        StructuredExecutionFrameKind::Loop, *lower, *upper, *step,
        getStructuredLoopSiblingOrdinal(loop)});
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

  int64_t rank;
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
        const TransportAction &member = trace.actions[cycle[index]];
        diagnostic.attachNote(member.operation->getLoc())
            << "rank " << member.rank << " " << getActionName(member.kind)
            << " participates in the wait cycle";
      }
      return mlir::failure();
    }
  return mlir::success();
}

static mlir::LogicalResult verifyStructuredTransportWaitGraph(
    llvm::ArrayRef<mlir::ModuleOp> rankModules,
    llvm::SmallVectorImpl<IssueRecord> &issues,
    llvm::SmallVectorImpl<MatchedMessage> &messages,
    StructuredTransportTrace &trace) {
  llvm::DenseMap<mlir::Operation *, IssueRecord *> issueByOperation;
  for (IssueRecord &issue : issues)
    issueByOperation[issue.operation] = &issue;

  for (size_t rank = 0; rank < rankModules.size(); ++rank) {
    mlir::ModuleOp module = rankModules[rank];
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
    StructuredTraceBuilder builder(static_cast<int64_t>(rank), module, *closure,
                                   transportFunctions, issueByOperation, trace);
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
    for (auto [sendIndex, recvIndex] :
         llvm::zip_equal(occurrences.sends, occurrences.receives)) {
      TransportAction &send = trace.actions[sendIndex];
      TransportAction &recv = trace.actions[recvIndex];
      if (send.occurrencePath != recv.occurrencePath)
        return send.operation->emitError(
            "direct_dte_acceptance: matched message call/region/loop "
            "occurrence paths are not structurally identical across ranks");
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
      addDependency(trace, sendIndex, recvIndex);
      matchingSend[sendIndex] = sendIndex;
      matchingSend[recvIndex] = sendIndex;
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
analyzeDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                          llvm::SmallVectorImpl<IssueRecord> &issues,
                          llvm::SmallVectorImpl<MatchedMessage> &messages,
                          StructuredTransportTrace &trace) {
  if (mlir::failed(collectIssues(rankModules, issues)))
    return mlir::failure();
  if (issues.empty())
    return mlir::success();
  if (mlir::failed(verifySenderResources(issues)) ||
      mlir::failed(verifyStructuredTransportWaitGraph(rankModules, issues,
                                                      messages, trace)) ||
      mlir::failed(allocateReceiverFSMs(issues, trace.receiverConflicts)))
    return mlir::failure();
  return mlir::success();
}

} // namespace

mlir::LogicalResult
verifyDirectDTETransportSchedule(llvm::ArrayRef<mlir::ModuleOp> rankModules) {
  llvm::SmallVector<IssueRecord, 32> issues;
  llvm::SmallVector<MatchedMessage, 32> messages;
  StructuredTransportTrace trace;
  return analyzeDirectDTETransport(rankModules, issues, messages, trace);
}

mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> rankModules) {
  llvm::SmallVector<IssueRecord, 32> issues;
  llvm::SmallVector<MatchedMessage, 32> messages;
  StructuredTransportTrace trace;
  if (mlir::failed(
          analyzeDirectDTETransport(rankModules, issues, messages, trace)))
    return mlir::failure();
  if (issues.empty())
    return TransportContract::None;

  llvm::SmallVector<std::pair<mlir::Operation *, DirectDTEBindingAttr>, 32>
      acceptedBindings;
  for (MatchedMessage &matched : messages) {
    auto binding = DirectDTEBindingAttr::get(
        matched.recv->operation->getContext(), DTEAllocationProfile::Normal,
        matched.recv->receiverFsmId, matched.recv->range.start,
        DTECompletionProfile::SenderWaitReceiverFSM);
    acceptedBindings.push_back({matched.send->operation, binding});
    acceptedBindings.push_back({matched.recv->operation, binding});
  }
  for (auto &[operation, binding] : acceptedBindings)
    operation->setAttr("binding", binding);
  return TransportContract::DirectDTE;
}

} // namespace wafer::compiler::detail

namespace wafer::compiler::testing {

mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> rankModules) {
  return detail::acceptDirectDTETransport(rankModules);
}

} // namespace wafer::compiler::testing
