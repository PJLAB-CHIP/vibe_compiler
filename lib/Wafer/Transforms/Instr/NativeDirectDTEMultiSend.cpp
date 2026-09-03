//===- NativeDirectDTEMultiSend.cpp - Form native DTE sends -------------===//

#include "NativeDirectDTEMultiSend.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

struct BroadcastGroup {
  mlir::Block *block = nullptr;
  mlir::Value buffer;
  int64_t sourceOffset = 0;
  int64_t communication = -1;
  int64_t round = -1;
  llvm::SmallVector<InstrDTESendOp, 16> sends;
};

struct StaticBufferSlice {
  mlir::Value root;
  int64_t offset = 0;
  int64_t bytes = 0;
};

struct ScatterMember {
  InstrDTESendOp send;
  StaticBufferSlice slice;
};

struct ScatterGroup {
  mlir::Block *block = nullptr;
  mlir::Value root;
  int64_t communication = -1;
  int64_t round = -1;
  llvm::SmallVector<ScatterMember, 16> members;
};

static bool isQualifiedDestinationCount(size_t count) {
  return count == 2 || count == 4 || count == 8 || count == 15;
}

static NativeDirectDTEMultiSendResult
fail(NativeDirectDTEMultiSendFailureKind kind, llvm::StringRef detail) {
  NativeDirectDTEMultiSendResult result;
  result.failure = kind;
  result.detail = detail.str();
  return result;
}

static bool areSendsConsecutive(llvm::ArrayRef<InstrDTESendOp> sends) {
  if (sends.empty())
    return false;
  llvm::DenseSet<mlir::Operation *> members;
  for (InstrDTESendOp send : sends)
    members.insert(send.getOperation());
  InstrDTESendOp firstSend = sends.front();
  InstrDTESendOp lastSend = sends.back();
  mlir::Operation *cursor = firstSend.getOperation();
  mlir::Operation *last = lastSend.getOperation();
  while (cursor) {
    if (!members.contains(cursor) &&
        !mlir::isa<mlir::ViewLikeOpInterface>(cursor))
      return false;
    if (cursor == last)
      return true;
    cursor = cursor->getNextNode();
  }
  return false;
}

static bool
areOperationsConsecutive(llvm::ArrayRef<mlir::Operation *> operations) {
  if (operations.empty())
    return false;
  llvm::DenseSet<mlir::Operation *> members(operations.begin(),
                                            operations.end());
  mlir::Operation *first = *llvm::min_element(
      operations, [](mlir::Operation *lhs, mlir::Operation *rhs) {
        return lhs->isBeforeInBlock(rhs);
      });
  mlir::Operation *last = *llvm::max_element(
      operations, [](mlir::Operation *lhs, mlir::Operation *rhs) {
        return lhs->isBeforeInBlock(rhs);
      });
  for (mlir::Operation *cursor = first; cursor;
       cursor = cursor->getNextNode()) {
    if (!members.contains(cursor) &&
        !mlir::isa<mlir::ViewLikeOpInterface>(cursor))
      return false;
    if (cursor == last)
      return true;
  }
  return false;
}

static std::optional<StaticBufferSlice>
getStaticContiguousSlice(mlir::Value buffer, int64_t requestedBytes,
                         int64_t explicitOffset) {
  mlir::Value current = buffer;
  auto currentType = mlir::dyn_cast<mlir::MemRefType>(current.getType());
  if (!currentType || !currentType.hasStaticShape())
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> offsets(currentType.getRank(), 0);
  llvm::SmallVector<int64_t, 4> sizes(currentType.getShape());
  while (auto subview = current.getDefiningOp<mlir::memref::SubViewOp>()) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(subview.getSource().getType());
    if (!sourceType || sourceType.getRank() != currentType.getRank() ||
        llvm::any_of(subview.getStaticOffsets(),
                     [](int64_t value) {
                       return mlir::ShapedType::isDynamic(value) || value < 0;
                     }) ||
        llvm::any_of(subview.getStaticStrides(),
                     [](int64_t value) { return value != 1; }))
      return std::nullopt;
    for (auto [index, addend] : llvm::enumerate(subview.getStaticOffsets()))
      if (llvm::AddOverflow(offsets[index], addend, offsets[index]))
        return std::nullopt;
    current = subview.getSource();
    currentType = sourceType;
  }
  MemoryAttr memory = getWaferMemoryAttr(currentType);
  if (!memory || (memory.getLayout() != MemLayout::Tensor &&
                  memory.getLayout() != MemLayout::NTensor))
    return std::nullopt;
  std::optional<WaferStaticPhysicalOffsetCalculator> calculator =
      WaferStaticPhysicalOffsetCalculator::create(currentType);
  if (!calculator || calculator->getInfo().bitPackedElement ||
      calculator->getInfo().elementBytes <= 0)
    return std::nullopt;
  int64_t elements = 1;
  llvm::SmallVector<int64_t, 4> last(offsets);
  for (auto [index, size] : llvm::enumerate(sizes)) {
    if (size <= 0 || llvm::MulOverflow(elements, size, elements) ||
        llvm::AddOverflow(last[index], size - 1, last[index]))
      return std::nullopt;
  }
  int64_t bytes = 0;
  if (llvm::MulOverflow(elements, calculator->getInfo().elementBytes, bytes) ||
      bytes != requestedBytes)
    return std::nullopt;
  std::optional<int64_t> firstOffset = calculator->getByteOffset(offsets);
  std::optional<int64_t> lastOffset = calculator->getByteOffset(last);
  int64_t end = 0;
  if (!firstOffset || !lastOffset ||
      llvm::AddOverflow(*lastOffset, calculator->getInfo().elementBytes, end) ||
      end - *firstOffset != bytes)
    return std::nullopt;
  int64_t adjusted = 0;
  if (explicitOffset < 0 ||
      llvm::AddOverflow(*firstOffset, explicitOffset, adjusted))
    return std::nullopt;
  return StaticBufferSlice{current, adjusted, bytes};
}

static std::optional<StaticBufferSlice>
getStaticContiguousSlice(InstrDTESendOp send) {
  return getStaticContiguousSlice(send.getBuffer(),
                                  send.getBytesAttr().getInt(),
                                  send.getBufferOffset().value_or(0));
}

using P2PMessageKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;

struct P2PEndpoint {
  mlir::Operation *operation = nullptr;
  mlir::Value token;
  mlir::Value buffer;
  StaticBufferSlice slice;
  int64_t peer = -1;
  int64_t bytes = 0;
  DTEMessageAttr message;
};

struct P2PEdge {
  int64_t sourceTile = -1;
  int64_t destinationTile = -1;
  P2PEndpoint send;
  P2PEndpoint receive;
};

struct P2PCoalescingGroup {
  int64_t sourceTile = -1;
  int64_t destinationTile = -1;
  mlir::Block *sourceBlock = nullptr;
  mlir::Block *destinationBlock = nullptr;
  int64_t communication = -1;
  int64_t round = -1;
  llvm::SmallVector<P2PEdge *, 8> edges;
};

static P2PMessageKey getP2PMessageKey(int64_t tile, int64_t peer,
                                      DTEMessageAttr message, bool send) {
  return std::make_tuple(send ? tile : peer, send ? peer : tile,
                         message.getCommunicationId(), message.getRound(),
                         message.getPayloadSlice());
}

} // namespace

NativeDirectDTEMultiSendResult
coalesceExactDirectDTETransfers(llvm::ArrayRef<mlir::ModuleOp> tileModules) {
  std::map<P2PMessageKey, P2PEndpoint> sends;
  std::map<P2PMessageKey, P2PEndpoint> receives;
  for (auto [tile, module] : llvm::enumerate(tileModules)) {
    mlir::ModuleOp currentModule = module;
    if (!currentModule || mlir::failed(mlir::verify(currentModule)))
      return fail(NativeDirectDTEMultiSendFailureKind::BrokenContract,
                  "exact DTE coalescing requires verifier-valid Instr "
                  "modules");
    bool malformed = false;
    currentModule.walk([&](mlir::Operation *operation) {
      auto send = mlir::dyn_cast<InstrDTESendOp>(operation);
      auto receive = mlir::dyn_cast<InstrDTERecvOp>(operation);
      if (!send && !receive)
        return;
      if ((send && (send.getBindingSelector() || send.getBindingAttr())) ||
          (receive &&
           (receive.getBindingSelector() || receive.getBindingAttr())))
        return;
      mlir::Value token = send ? send.getToken() : receive.getToken();
      if (!token.use_empty()) {
        malformed = true;
        return;
      }
      mlir::Value buffer = send ? send.getBuffer() : receive.getBuffer();
      int64_t bytes =
          send ? send.getBytesAttr().getInt() : receive.getBytesAttr().getInt();
      int64_t offset = send ? send.getBufferOffset().value_or(0)
                            : receive.getBufferOffset().value_or(0);
      std::optional<StaticBufferSlice> slice =
          getStaticContiguousSlice(buffer, bytes, offset);
      if (!slice)
        return;
      int64_t peer =
          send ? send.getPeerAttr().getInt() : receive.getPeerAttr().getInt();
      DTEMessageAttr message =
          send ? send.getMessageAttr() : receive.getMessageAttr();
      P2PEndpoint endpoint{operation, token, buffer, *slice,
                           peer,      bytes, message};
      auto &map = send ? sends : receives;
      if (!map.try_emplace(getP2PMessageKey(static_cast<int64_t>(tile), peer,
                                            message, static_cast<bool>(send)),
                           std::move(endpoint))
               .second)
        malformed = true;
    });
    if (malformed)
      return fail(NativeDirectDTEMultiSendFailureKind::BrokenContract,
                  "exact DTE coalescing found duplicate messages or "
                  "preexisting completion/binding");
  }

  llvm::SmallVector<P2PEdge, 32> edges;
  edges.reserve(sends.size());
  for (auto &[key, send] : sends) {
    auto receive = receives.find(key);
    if (receive == receives.end() || send.bytes != receive->second.bytes)
      continue;
    const auto &[source, destination, communication, round, slice] = key;
    (void)communication;
    (void)round;
    (void)slice;
    edges.push_back(P2PEdge{source, destination, send, receive->second});
  }

  llvm::SmallVector<P2PCoalescingGroup, 16> groups;
  for (P2PEdge &edge : edges) {
    auto found = llvm::find_if(groups, [&](const P2PCoalescingGroup &group) {
      return group.sourceTile == edge.sourceTile &&
             group.destinationTile == edge.destinationTile &&
             group.sourceBlock == edge.send.operation->getBlock() &&
             group.destinationBlock == edge.receive.operation->getBlock() &&
             group.communication == edge.send.message.getCommunicationId() &&
             group.round == edge.send.message.getRound();
    });
    if (found == groups.end()) {
      P2PCoalescingGroup group;
      group.sourceTile = edge.sourceTile;
      group.destinationTile = edge.destinationTile;
      group.sourceBlock = edge.send.operation->getBlock();
      group.destinationBlock = edge.receive.operation->getBlock();
      group.communication = edge.send.message.getCommunicationId();
      group.round = edge.send.message.getRound();
      group.edges.push_back(&edge);
      groups.push_back(std::move(group));
    } else {
      found->edges.push_back(&edge);
    }
  }

  llvm::SmallVector<P2PCoalescingGroup *, 16> selected;
  for (P2PCoalescingGroup &group : groups) {
    if (group.edges.size() < 2)
      continue;
    llvm::sort(group.edges, [](const P2PEdge *lhs, const P2PEdge *rhs) {
      return std::make_tuple(lhs->send.slice.offset,
                             lhs->send.message.getPayloadSlice()) <
             std::make_tuple(rhs->send.slice.offset,
                             rhs->send.message.getPayloadSlice());
    });
    mlir::Value sourceRoot = group.edges.front()->send.slice.root;
    mlir::Value destinationRoot = group.edges.front()->receive.slice.root;
    int64_t sourceEnd = group.edges.front()->send.slice.offset;
    int64_t destinationEnd = group.edges.front()->receive.slice.offset;
    int64_t totalBytes = 0;
    bool contiguous = true;
    llvm::SmallVector<mlir::Operation *, 8> sourceOperations;
    llvm::SmallVector<mlir::Operation *, 8> destinationOperations;
    for (P2PEdge *edge : group.edges) {
      contiguous &= edge->send.slice.root == sourceRoot &&
                    edge->receive.slice.root == destinationRoot &&
                    edge->send.slice.offset == sourceEnd &&
                    edge->receive.slice.offset == destinationEnd &&
                    edge->send.bytes == edge->receive.bytes;
      if (llvm::AddOverflow(sourceEnd, edge->send.bytes, sourceEnd) ||
          llvm::AddOverflow(destinationEnd, edge->receive.bytes,
                            destinationEnd) ||
          llvm::AddOverflow(totalBytes, edge->send.bytes, totalBytes) ||
          totalBytes > std::numeric_limits<int32_t>::max())
        contiguous = false;
      sourceOperations.push_back(edge->send.operation);
      destinationOperations.push_back(edge->receive.operation);
    }
    if (contiguous && areOperationsConsecutive(sourceOperations) &&
        areOperationsConsecutive(destinationOperations))
      selected.push_back(&group);
  }

  NativeDirectDTEMultiSendResult result;
  if (selected.empty())
    return result;
  mlir::ModuleOp firstModule = tileModules.front();
  mlir::IRRewriter rewriter(firstModule.getContext());
  for (P2PCoalescingGroup *group : selected) {
    P2PEdge *first = group->edges.front();
    int64_t totalBytes = 0;
    for (P2PEdge *edge : group->edges)
      totalBytes += edge->send.bytes;
    rewriter.setInsertionPoint(first->send.operation);
    rewriter.create<InstrDTESendOp>(
        first->send.operation->getLoc(),
        rewriter.getType<mlir::async::TokenType>(), first->send.slice.root,
        mlir::Value(), rewriter.getI64IntegerAttr(first->send.slice.offset),
        rewriter.getI64IntegerAttr(group->destinationTile),
        rewriter.getI64IntegerAttr(totalBytes), first->send.message,
        DirectDTEBindingAttr());
    rewriter.setInsertionPoint(first->receive.operation);
    rewriter.create<InstrDTERecvOp>(
        first->receive.operation->getLoc(),
        rewriter.getType<mlir::async::TokenType>(), first->receive.slice.root,
        mlir::Value(), rewriter.getI64IntegerAttr(first->receive.slice.offset),
        rewriter.getI64IntegerAttr(group->sourceTile),
        rewriter.getI64IntegerAttr(totalBytes), first->send.message,
        DirectDTEBindingAttr());
    for (P2PEdge *edge : llvm::reverse(group->edges)) {
      rewriter.eraseOp(edge->send.operation);
      rewriter.eraseOp(edge->receive.operation);
    }
    ++result.statistics.coalescedP2PTransfers;
    result.statistics.unicastSendOperationsRemoved += group->edges.size();
    result.statistics.unicastReceiveOperationsRemoved += group->edges.size();
  }
  for (mlir::ModuleOp module : tileModules)
    if (mlir::failed(mlir::verify(module)))
      return fail(NativeDirectDTEMultiSendFailureKind::CompilerFailure,
                  "exact DTE coalescing produced invalid Instr IR");
  return result;
}

NativeDirectDTEMultiSendResult materializeNativeDirectDTEMultiSends(
    llvm::ArrayRef<mlir::ModuleOp> tileModules) {
  llvm::SmallVector<BroadcastGroup, 16> groups;
  llvm::SmallVector<ScatterGroup, 16> scatterGroups;
  for (mlir::ModuleOp module : tileModules) {
    if (!module || mlir::failed(mlir::verify(module)))
      return fail(NativeDirectDTEMultiSendFailureKind::BrokenContract,
                  "native DTE multi-send requires verifier-valid Instr "
                  "modules");
    bool malformed = false;
    module.walk([&](InstrDTESendOp send) {
      if (send.getBytesAttr().getInt() != 256 || send.getBindingSelector() ||
          send.getBindingAttr())
        return;
      if (!send.getToken().use_empty()) {
        malformed = true;
        return;
      }
      DTEMessageAttr message = send.getMessageAttr();
      auto found = llvm::find_if(groups, [&](const BroadcastGroup &group) {
        return group.block == send->getBlock() &&
               group.buffer == send.getBuffer() &&
               group.sourceOffset ==
                   static_cast<int64_t>(send.getBufferOffset().value_or(0)) &&
               group.communication == message.getCommunicationId() &&
               group.round == message.getRound();
      });
      if (found == groups.end()) {
        BroadcastGroup group;
        group.block = send->getBlock();
        group.buffer = send.getBuffer();
        group.sourceOffset =
            static_cast<int64_t>(send.getBufferOffset().value_or(0));
        group.communication = message.getCommunicationId();
        group.round = message.getRound();
        group.sends.push_back(send);
        groups.push_back(std::move(group));
      } else {
        found->sends.push_back(send);
      }
      std::optional<StaticBufferSlice> slice = getStaticContiguousSlice(send);
      if (!slice)
        return;
      auto scatterGroup =
          llvm::find_if(scatterGroups, [&](const ScatterGroup &group) {
            return group.block == send->getBlock() &&
                   group.root == slice->root &&
                   group.communication == message.getCommunicationId() &&
                   group.round == message.getRound();
          });
      if (scatterGroup == scatterGroups.end()) {
        ScatterGroup group;
        group.block = send->getBlock();
        group.root = slice->root;
        group.communication = message.getCommunicationId();
        group.round = message.getRound();
        group.members.push_back(ScatterMember{send, *slice});
        scatterGroups.push_back(std::move(group));
      } else {
        scatterGroup->members.push_back(ScatterMember{send, *slice});
      }
    });
    if (malformed)
      return fail(NativeDirectDTEMultiSendFailureKind::BrokenContract,
                  "native DTE grouping requires sender tokens without "
                  "preexisting completion uses");
  }

  llvm::SmallVector<BroadcastGroup *, 16> selected;
  for (BroadcastGroup &group : groups) {
    if (!isQualifiedDestinationCount(group.sends.size()))
      continue;
    llvm::sort(group.sends, [](InstrDTESendOp lhs, InstrDTESendOp rhs) {
      return std::make_tuple(lhs.getPeerAttr().getInt(),
                             lhs.getMessageAttr().getPayloadSlice()) <
             std::make_tuple(rhs.getPeerAttr().getInt(),
                             rhs.getMessageAttr().getPayloadSlice());
    });
    llvm::DenseSet<int64_t> peers;
    bool uniquePeers = true;
    for (InstrDTESendOp send : group.sends)
      uniquePeers &= peers.insert(send.getPeerAttr().getInt()).second;
    if (!uniquePeers)
      continue;
    llvm::sort(group.sends, [](InstrDTESendOp lhs, InstrDTESendOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    if (!areSendsConsecutive(group.sends))
      continue;
    selected.push_back(&group);
  }

  llvm::SmallVector<ScatterGroup *, 16> selectedScatter;
  for (ScatterGroup &group : scatterGroups) {
    if (!isQualifiedDestinationCount(group.members.size()))
      continue;
    llvm::sort(group.members, [](const ScatterMember &lhs,
                                 const ScatterMember &rhs) {
      InstrDTESendOp lhsSend = lhs.send;
      InstrDTESendOp rhsSend = rhs.send;
      return std::make_tuple(lhs.slice.offset, lhsSend.getPeerAttr().getInt()) <
             std::make_tuple(rhs.slice.offset, rhsSend.getPeerAttr().getInt());
    });
    llvm::DenseSet<int64_t> peers;
    int64_t expectedOffset = group.members.front().slice.offset;
    bool contiguous = true;
    llvm::SmallVector<InstrDTESendOp, 16> sends;
    for (const ScatterMember &member : group.members) {
      InstrDTESendOp send = member.send;
      contiguous &= member.slice.bytes == 256 &&
                    member.slice.offset == expectedOffset &&
                    peers.insert(send.getPeerAttr().getInt()).second;
      if (llvm::AddOverflow(expectedOffset, int64_t{256}, expectedOffset))
        contiguous = false;
      sends.push_back(send);
    }
    llvm::sort(sends, [](InstrDTESendOp lhs, InstrDTESendOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    if (!contiguous || !areSendsConsecutive(sends))
      continue;
    bool overlapsBroadcast = llvm::any_of(selected, [&](BroadcastGroup *other) {
      return llvm::any_of(group.members, [&](const ScatterMember &member) {
        return llvm::is_contained(other->sends, member.send);
      });
    });
    if (!overlapsBroadcast)
      selectedScatter.push_back(&group);
  }

  NativeDirectDTEMultiSendResult result;
  if (selected.empty() && selectedScatter.empty())
    return result;
  mlir::ModuleOp firstModule = tileModules.front();
  mlir::IRRewriter rewriter(firstModule.getContext());
  for (BroadcastGroup *group : selected) {
    InstrDTESendOp anchor = *llvm::min_element(
        group->sends, [](InstrDTESendOp lhs, InstrDTESendOp rhs) {
          return lhs->isBeforeInBlock(rhs);
        });
    llvm::sort(group->sends, [](InstrDTESendOp lhs, InstrDTESendOp rhs) {
      return std::make_tuple(lhs.getPeerAttr().getInt(),
                             lhs.getMessageAttr().getPayloadSlice()) <
             std::make_tuple(rhs.getPeerAttr().getInt(),
                             rhs.getMessageAttr().getPayloadSlice());
    });
    llvm::SmallVector<int64_t, 16> peers;
    llvm::SmallVector<mlir::Attribute, 16> messages;
    for (InstrDTESendOp send : group->sends) {
      peers.push_back(send.getPeerAttr().getInt());
      messages.push_back(send.getMessageAttr());
    }
    rewriter.setInsertionPoint(anchor);
    auto broadcast = rewriter.create<InstrDTEBroadcastOp>(
        anchor.getLoc(), rewriter.getType<mlir::async::TokenType>(),
        group->buffer, group->sourceOffset, peers, /*bytes=*/256,
        mlir::ArrayAttr::get(rewriter.getContext(), messages),
        mlir::ArrayAttr());
    (void)broadcast;
    for (InstrDTESendOp send : llvm::reverse(group->sends))
      rewriter.eraseOp(send);
    ++result.statistics.broadcastOperations;
    result.statistics.unicastSendOperationsRemoved += group->sends.size();
  }
  for (ScatterGroup *group : selectedScatter) {
    llvm::SmallVector<int64_t, 16> peers;
    llvm::SmallVector<mlir::Attribute, 16> messages;
    llvm::SmallVector<InstrDTESendOp, 16> sends;
    for (const ScatterMember &member : group->members) {
      InstrDTESendOp send = member.send;
      peers.push_back(send.getPeerAttr().getInt());
      messages.push_back(send.getMessageAttr());
      sends.push_back(send);
    }
    InstrDTESendOp anchor =
        *llvm::min_element(sends, [](InstrDTESendOp lhs, InstrDTESendOp rhs) {
          return lhs->isBeforeInBlock(rhs);
        });
    rewriter.setInsertionPoint(anchor);
    auto scatter = rewriter.create<InstrDTEScatterOp>(
        anchor.getLoc(), rewriter.getType<mlir::async::TokenType>(),
        group->root, static_cast<uint64_t>(group->members.front().slice.offset),
        peers, /*bytes=*/256,
        mlir::ArrayAttr::get(rewriter.getContext(), messages),
        mlir::ArrayAttr());
    (void)scatter;
    for (InstrDTESendOp send : llvm::reverse(sends))
      rewriter.eraseOp(send);
    ++result.statistics.scatterOperations;
    result.statistics.unicastSendOperationsRemoved += sends.size();
  }
  for (mlir::ModuleOp module : tileModules)
    if (mlir::failed(mlir::verify(module)))
      return fail(NativeDirectDTEMultiSendFailureKind::CompilerFailure,
                  "native DTE grouping produced invalid Instr IR");
  return result;
}

} // namespace wafer::compiler::detail
