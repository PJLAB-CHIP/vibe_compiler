//===- DirectDTETransport.cpp - Physical Direct DTE acceptance ----------===//

#include "DirectDTETransport.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

struct MessageKey {
  int64_t sourceRank = -1;
  int64_t destinationRank = -1;
  int64_t communicationId = -1;
  DTEProtocolPhase phase = DTEProtocolPhase::CollectivePermute;
  int64_t round = -1;
  int64_t payloadSlice = -1;

  auto asTuple() const {
    return std::tie(sourceRank, destinationRank, communicationId, phase, round,
                    payloadSlice);
  }
  bool operator<(const MessageKey &other) const {
    return asTuple() < other.asTuple();
  }
};

struct PhysicalRange {
  int64_t start = -1;
  int64_t end = -1;
};

struct IssueRecord {
  mlir::Operation *operation = nullptr;
  mlir::Operation *wait = nullptr;
  mlir::Block *block = nullptr;
  MessageKey key;
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

static MessageKey makeMessageKey(int64_t rank, int64_t peer,
                                 DTEMessageAttr message, bool isSend) {
  return MessageKey{isSend ? rank : peer,         isSend ? peer : rank,
                    message.getCommunicationId(), message.getPhase(),
                    message.getRound(),           message.getPayloadSlice()};
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
      mlir::FailureOr<PhysicalRange> range =
          resolveAcceptedRange(operation, buffer, bytes);
      mlir::FailureOr<mlir::Operation *> wait =
          findUniqueSameBlockWait(operation, token, operationIndices);
      if (mlir::failed(range) || mlir::failed(wait)) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      issues.push_back(IssueRecord{
          operation, *wait, operation->getBlock(),
          makeMessageKey(static_cast<int64_t>(rankIndex), peer, message,
                         static_cast<bool>(send)),
          *range, bytes, operationIndices.lookup(operation),
          operationIndices.lookup(*wait), -1, static_cast<bool>(send)});
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

static mlir::LogicalResult
allocateReceiverFSMs(llvm::SmallVectorImpl<IssueRecord> &issues) {
  std::map<mlir::Block *, llvm::SmallVector<IssueRecord *, 8>> byBlock;
  for (IssueRecord &issue : issues)
    if (!issue.isSend)
      byBlock[issue.block].push_back(&issue);

  for (auto &entry : byBlock) {
    llvm::SmallVector<IssueRecord *, 8> &blockIssues = entry.second;
    llvm::sort(blockIssues, [](const IssueRecord *lhs, const IssueRecord *rhs) {
      return lhs->issueIndex < rhs->issueIndex;
    });
    int64_t liveUntil[4] = {-1, -1, -1, -1};
    for (IssueRecord *issue : blockIssues) {
      int64_t selected = -1;
      for (int64_t fsm = 0; fsm < 4; ++fsm)
        if (liveUntil[fsm] < static_cast<int64_t>(issue->issueIndex)) {
          selected = fsm;
          break;
        }
      if (selected < 0)
        return issue->operation->emitError(
            "direct_dte_acceptance: more than four receiver FSM live ranges "
            "overlap");
      issue->receiverFsmId = selected;
      liveUntil[selected] = static_cast<int64_t>(issue->waitIndex);
    }
  }
  return mlir::success();
}

static mlir::LogicalResult
matchMessages(llvm::SmallVectorImpl<IssueRecord> &issues,
              std::map<MessageKey, MatchedMessage> &messages) {
  for (IssueRecord &issue : issues) {
    MatchedMessage &matched = messages[issue.key];
    IssueRecord *&slot = issue.isSend ? matched.send : matched.recv;
    if (slot)
      return issue.operation->emitError(
          "direct_dte_acceptance: duplicate send or receive message identity");
    slot = &issue;
  }
  for (auto &entry : messages) {
    MatchedMessage &matched = entry.second;
    if (!matched.send || !matched.recv) {
      mlir::Operation *anchor =
          matched.send ? matched.send->operation : matched.recv->operation;
      return anchor->emitError(
          "direct_dte_acceptance: message identity has no matching peer "
          "issue");
    }
    if (matched.send->bytes != matched.recv->bytes)
      return matched.recv->operation->emitError(
          "direct_dte_acceptance: matched send and receive byte counts differ");
  }
  return mlir::success();
}

} // namespace

mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> rankModules) {
  llvm::SmallVector<IssueRecord, 32> issues;
  if (mlir::failed(collectIssues(rankModules, issues)))
    return mlir::failure();
  if (issues.empty())
    return TransportContract::None;
  if (mlir::failed(verifySenderResources(issues)) ||
      mlir::failed(allocateReceiverFSMs(issues)))
    return mlir::failure();

  std::map<MessageKey, MatchedMessage> messages;
  if (mlir::failed(matchMessages(issues, messages)))
    return mlir::failure();

  llvm::SmallVector<std::pair<mlir::Operation *, DirectDTEBindingAttr>, 32>
      acceptedBindings;
  for (auto &entry : messages) {
    MatchedMessage &matched = entry.second;
    auto binding = DirectDTEBindingAttr::get(
        matched.recv->operation->getContext(), DTEAllocationProfile::Normal,
        matched.recv->receiverFsmId,
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
