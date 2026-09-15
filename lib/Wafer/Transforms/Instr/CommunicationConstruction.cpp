//===- CommunicationConstruction.cpp - Legal current communication choices ===//

#include "CommunicationConstruction.h"
#include "Wafer/Transforms/Instr/CommunicationScheduling.h"

#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

using Outcome = SharedDDRCompletionResult;

Outcome fail(SharedDDRCompletionFailure kind, llvm::StringRef detail) {
  return {kind, detail.str()};
}

TileRegionOp directRegion(mlir::Operation *op) {
  auto region = mlir::dyn_cast_or_null<TileRegionOp>(op->getParentOp());
  auto function = region ? region->getParentOfType<mlir::func::FuncOp>()
                         : mlir::func::FuncOp{};
  return function && !function.isPrivate() &&
                 function.getBody().hasOneBlock() &&
                 region->getBlock() == &function.getBody().front()
             ? region
             : TileRegionOp{};
}

mlir::Value bindInput(TileRegionOp region, mlir::Value value) {
  for (auto [index, input] : llvm::enumerate(region.getInputs()))
    if (input == value)
      return region.getBody().front().getArgument(index);
  region.getInputsMutable().append(value);
  return region.getBody().front().addArgument(value.getType(), value.getLoc());
}

bool waitsOnly(mlir::Value token) {
  return llvm::all_of(token.getUsers(), [](mlir::Operation *op) {
    return mlir::isa<InstrDTEWaitOp>(op);
  });
}

enum class AccessKind { None, Read, Write };

// Issue placement changes only operation order and memory-free DTE waits.
// Buffer operands, SSA aliases and memory effects are unchanged, so these
// summaries remain valid for this call. Rebuilding waits or replacing messages
// happens only after this object has been destroyed.
class BufferAccesses {
public:
  AccessKind get(mlir::Operation *op, mlir::Value buffer) {
    if (mlir::isa<InstrDTEWaitOp, mlir::ViewLikeOpInterface>(op))
      return AccessKind::None;
    auto [entry, inserted] = summaries.try_emplace(op);
    auto &summary = entry->second;
    if (inserted) {
      auto effects = mlir::getEffectsRecursively(op);
      summary.unknown = !effects;
      if (effects)
        for (const auto &effect : *effects) {
          auto value = effect.getValue();
          if (!value) {
            summary.unknown |= effect.getResource() ==
                               mlir::SideEffects::DefaultResource::get();
            continue;
          }
          if (!mlir::isa<mlir::BaseMemRefType>(value.getType()))
            continue;
          AccessKind kind = AccessKind::None;
          if (mlir::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Free>(
                  effect.getEffect()))
            kind = AccessKind::Write;
          else if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
            kind = AccessKind::Read;
          for (mlir::Value root : roots.getStorageRoots(value))
            summary.accesses[root] = std::max(summary.accesses[root], kind);
        }
    }
    if (summary.unknown)
      return AccessKind::Write;
    AccessKind kind = AccessKind::None;
    for (mlir::Value root : roots.getStorageRoots(buffer))
      kind = std::max(kind, summary.accesses.lookup(root));
    return kind;
  }

  uint64_t size() const { return summaries.size(); }

private:
  struct Summary {
    bool unknown = false;
    llvm::DenseMap<mlir::Value, AccessKind> accesses;
  };
  StorageRootMemo roots;
  llvm::DenseMap<mlir::Operation *, Summary> summaries;
};

mlir::Operation *nextBufferAccess(mlir::Operation *op, mlir::Value buffer) {
  BufferAccesses accesses;
  for (auto *next = op->getNextNode(); next; next = next->getNextNode())
    if (next->hasTrait<mlir::OpTrait::IsTerminator>() ||
        accesses.get(next, buffer) != AccessKind::None)
      return next;
  return nullptr;
}

mlir::Operation *precedingBufferWrite(mlir::Operation *op, mlir::Value buffer) {
  BufferAccesses accesses;
  for (auto *previous = op->getPrevNode(); previous;
       previous = previous->getPrevNode())
    if (previous == buffer.getDefiningOp() ||
        accesses.get(previous, buffer) == AccessKind::Write)
      return previous;
  return nullptr;
}

void eraseWaits(mlir::Value token, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<mlir::Operation *> users(token.getUsers());
  for (auto *user : users) {
    auto wait = mlir::cast<InstrDTEWaitOp>(user);
    llvm::SmallVector<mlir::Value> kept;
    for (auto value : wait.getTokens())
      if (value != token)
        kept.push_back(value);
    if (kept.empty())
      rewriter.eraseOp(wait);
    else
      rewriter.modifyOpInPlace(wait,
                               [&] { wait.getTokensMutable().assign(kept); });
  }
}

struct Receiver {
  unsigned tile;
  InstrDTERecvOp operation;
  int64_t packetOffset;
};

struct Packet {
  mlir::Operation *sender;
  unsigned tile;
  mlir::Value source;
  int64_t sourceOffset;
  int64_t bytes;
  int64_t pieceBytes;
  llvm::SmallVector<Receiver, 4> receivers;
};

using Message = std::tuple<uint64_t, uint64_t, int64_t, int64_t, int64_t>;
Message messageKey(uint64_t source, uint64_t dest, DTEMessageAttr message) {
  return {source, dest, message.getCommunicationId(), message.getRound(),
          message.getPayloadSlice()};
}

mlir::FailureOr<llvm::SmallVector<Packet>>
collectPackets(llvm::ArrayRef<mlir::ModuleOp> modules,
               llvm::ArrayRef<TileId> tileIds) {
  support::ScopedCompileTimingSpan timing(
      "construction-phase", "communication-proposal", "collect-packets");
  std::map<Message, std::pair<unsigned, InstrDTERecvOp>> receives;
  for (auto [index, module] : llvm::enumerate(modules))
    mlir::ModuleOp(module).walk([&](InstrDTERecvOp recv) {
      auto entry = recv->getParentOfType<mlir::func::FuncOp>();
      if (!entry || entry.isPrivate())
        return;
      receives.emplace(messageKey(recv.getPeer(), tileIds[index].getValue(),
                                  recv.getMessage()),
                       std::make_pair(index, recv));
    });
  llvm::SmallVector<Packet> packets;
  bool invalid = false;
  for (auto [index, module] : llvm::enumerate(modules))
    mlir::ModuleOp(module).walk([&](mlir::Operation *op) {
      Packet packet{op, static_cast<unsigned>(index), {}, 0, 0, 0, {}};
      llvm::SmallVector<std::pair<int64_t, DTEMessageAttr>> endpoints;
      bool scatter = false;
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(op)) {
        if (send.getBindingSelector() || send.getBinding())
          return;
        packet.source = send.getBuffer();
        packet.sourceOffset = send.getBufferOffset().value_or(0);
        packet.bytes = packet.pieceBytes = send.getBytes();
        endpoints.push_back({send.getPeer(), send.getMessage()});
      } else {
        auto append = [&](auto send) {
          packet.source = send.getBuffer();
          packet.sourceOffset = send.getSourceOffset();
          packet.bytes = packet.pieceBytes = send.getBytes();
          for (auto [peer, message] :
               llvm::zip_equal(send.getPeers(), send.getMessages()))
            endpoints.push_back({peer, mlir::cast<DTEMessageAttr>(message)});
        };
        if (auto send = mlir::dyn_cast<InstrDTEBroadcastOp>(op))
          append(send);
        else if (auto send = mlir::dyn_cast<InstrDTEScatterOp>(op)) {
          append(send);
          scatter = true;
          if (llvm::MulOverflow(packet.bytes,
                                static_cast<int64_t>(endpoints.size()),
                                packet.bytes)) {
            invalid = true;
            return;
          }
        } else
          return;
      }
      if (!directRegion(op) || !waitsOnly(op->getResult(0)))
        return;
      for (auto [ordinal, endpoint] : llvm::enumerate(endpoints)) {
        auto found = receives.find(messageKey(tileIds[index].getValue(),
                                              endpoint.first, endpoint.second));
        if (found == receives.end()) {
          invalid = true;
          return;
        }
        if (!directRegion(found->second.second) ||
            found->second.second.getBindingSelector() ||
            found->second.second.getBinding() ||
            !waitsOnly(found->second.second.getToken()))
          return;
        packet.receivers.push_back(
            {found->second.first, found->second.second,
             scatter ? static_cast<int64_t>(ordinal) * packet.pieceBytes : 0});
      }
      packets.push_back(std::move(packet));
    });
  return invalid
             ? mlir::FailureOr<llvm::SmallVector<Packet>>(mlir::failure())
             : mlir::FailureOr<llvm::SmallVector<Packet>>(std::move(packets));
}

struct InputOrigin {
  mlir::BlockArgument argument;
  llvm::SmallVector<mlir::Operation *, 4> views;
};

std::optional<InputOrigin> inputOrigin(mlir::Value value) {
  InputOrigin result;
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (mlir::isa<mlir::func::FuncOp>(argument.getOwner()->getParentOp())) {
        result.argument = argument;
        return result;
      }
      value = analysis::getSingleExecutionRegionEntryOperand(argument);
      continue;
    }
    auto *owner = value.getDefiningOp();
    auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(owner);
    if (!view || owner->getNumOperands() != 1 || owner->getNumResults() != 1)
      return std::nullopt;
    result.views.push_back(owner);
    value = view.getViewSource();
  }
  return std::nullopt;
}

InstrRDMAOp soleInputLoad(mlir::Value buffer) {
  if (!buffer.getDefiningOp<mlir::memref::AllocOp>())
    return {};
  InstrRDMAOp load;
  for (mlir::Operation *user : buffer.getUsers()) {
    auto effects = mlir::getEffectsRecursively(user);
    if (!effects || mlir::isa<mlir::ViewLikeOpInterface>(user))
      return {};
    for (const auto &effect : *effects) {
      if (effect.getValue() != buffer ||
          !mlir::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Free>(
              effect.getEffect()))
        continue;
      auto candidate = mlir::dyn_cast<InstrRDMAOp>(user);
      if (!candidate || candidate.getDest() != buffer ||
          (load && load != candidate))
        return {};
      load = candidate;
    }
  }
  return load && directRegion(load) ? load : InstrRDMAOp{};
}

bool hasOnlyInputReads(mlir::Value value, llvm::DenseSet<mlir::Value> &seen) {
  if (!seen.insert(value).second)
    return true;
  for (mlir::OpOperand &use : value.getUses()) {
    auto *op = use.getOwner();
    if (auto region = mlir::dyn_cast<TileRegionOp>(op)) {
      if (!hasOnlyInputReads(
              region.getBody().front().getArgument(use.getOperandNumber()),
              seen))
        return false;
      continue;
    }
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
      if (view.getViewSource() != value)
        return false;
      for (auto result : op->getResults())
        if (!hasOnlyInputReads(result, seen))
          return false;
      continue;
    }
    auto effects = mlir::getEffectsRecursively(op);
    if (!effects || op->getNumRegions() ||
        op->hasTrait<mlir::OpTrait::IsTerminator>())
      return false;
    bool read = false;
    for (const auto &effect : *effects)
      if (effect.getValue() == value) {
        if (!mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
          return false;
        read = true;
      }
    if (!read)
      return false;
  }
  return true;
}

// A local alternative is reconstructed from the actual donor DMA and its
// typed program input, never from a remembered pre-lowering load or a name.
mlir::FailureOr<bool>
restoreLocalInput(const Packet &packet,
                  llvm::ArrayRef<mlir::ModuleOp> modules) {
  support::ScopedCompileTimingSpan timing(
      "construction-phase", "communication-proposal", "restore-input");
  if (packet.receivers.size() != 1 || packet.sourceOffset != 0)
    return false;
  auto load = soleInputLoad(packet.source);
  if (!load || load.getByteCount() != static_cast<uint64_t>(packet.bytes) ||
      load.getDstOffset().value_or(0) != 0)
    return false;
  auto origin = inputOrigin(load.getSource());
  if (!origin)
    return false;
  auto function = mlir::cast<mlir::func::FuncOp>(
      origin->argument.getOwner()->getParentOp());
  auto identity = function.getArgAttrOfType<ProgramArgumentAttr>(
      origin->argument.getArgNumber(), kWaferProgramArgumentAttrName);
  if (!identity)
    return false;
  for (auto module : modules)
    for (auto entry : module.getOps<mlir::func::FuncOp>()) {
      if (entry.isExternal() || entry.isPrivate())
        continue;
      for (auto input : entry.getArguments())
        if (entry.getArgAttrOfType<ProgramArgumentAttr>(
                input.getArgNumber(), kWaferProgramArgumentAttrName) ==
            identity) {
          llvm::DenseSet<mlir::Value> seen;
          if (!hasOnlyInputReads(input, seen))
            return false;
        }
    }
  auto recv = packet.receivers.front().operation;
  if (recv.getBufferOffset().value_or(0) != 0 ||
      recv.getBuffer().getType() != packet.source.getType())
    return false;
  auto region = directRegion(recv);
  auto target = region->getParentOfType<mlir::func::FuncOp>();
  mlir::BlockArgument argument;
  for (auto candidate : target.getArguments()) {
    auto key = target.getArgAttrOfType<ProgramArgumentAttr>(
        candidate.getArgNumber(), kWaferProgramArgumentAttrName);
    if (key == identity && candidate.getType() == origin->argument.getType())
      argument = candidate;
  }
  if (!argument)
    return false;
  mlir::IRRewriter rewriter(recv.getContext());
  rewriter.setInsertionPoint(recv);
  mlir::IRMapping mapping;
  mlir::Value source = bindInput(region, argument);
  for (auto *view : llvm::reverse(origin->views)) {
    mapping.map(mlir::cast<mlir::ViewLikeOpInterface>(view).getViewSource(),
                source);
    source = rewriter.clone(*view, mapping)->getResult(0);
  }
  mapping.map(load.getSource(), source);
  mapping.map(load.getDest(), recv.getBuffer());
  auto replacement = mlir::cast<InstrRDMAOp>(rewriter.clone(*load, mapping));
  replacement.setWorkerAttr(
      NCCWorkerAttr::get(rewriter.getContext(), NCCWorker::Worker0));
  eraseWaits(recv.getToken(), rewriter);
  eraseWaits(packet.sender->getResult(0), rewriter);
  rewriter.eraseOp(recv);
  rewriter.eraseOp(packet.sender);
  return true;
}

Outcome useDDRPacket(const Packet &packet,
                     llvm::ArrayRef<mlir::ModuleOp> modules) {
  support::ScopedCompileTimingSpan timing(
      "construction-phase", "communication-proposal", "materialize-packet");
  auto type = mlir::cast<mlir::MemRefType>(packet.source.getType());
  auto element = type.getElementType();
  if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(element) ||
      packet.bytes <= 0)
    return fail(SharedDDRCompletionFailure::Unsupported,
                "communication packet has no fixed numeric representation");
  int64_t bits;
  const auto width = element.getIntOrFloatBitWidth();
  if (llvm::MulOverflow(packet.bytes, int64_t{8}, bits) || !width ||
      bits % width)
    return fail(
        SharedDDRCompletionFailure::Unsupported,
        "communication packet is not representable in its source dtype");
  mlir::IRRewriter rewriter(packet.sender->getContext());
  auto packetType = mlir::MemRefType::get(
      {1, 1, bits / width}, element, mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(rewriter.getContext(), MemorySpace::DDR,
                      MemLayout::Tensor));
  auto physical = computeWaferPhysicalTensorInfo(packetType);
  if (!physical || physical->physicalBytes != packet.bytes)
    return fail(SharedDDRCompletionFailure::Unsupported,
                "DDR packet type does not preserve the actual payload bytes");
  int64_t resourceId = 0;
  for (auto module : modules)
    for (auto global : module.getOps<mlir::memref::GlobalOp>())
      if (auto resource = global->getAttrOfType<DDRResourceAttr>(
              kWaferDDRResourceAttrName)) {
        if (resource.getResourceId() > std::numeric_limits<int64_t>::max() - 3)
          return fail(SharedDDRCompletionFailure::Unsupported,
                      "communication packet exhausts DDR resource identities");
        resourceId = std::max(resourceId, resource.getResourceId() + 1);
      }
  std::string symbol =
      "__wafer_communication_packet_" + std::to_string(resourceId);
  auto resource = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), symbol);
  std::map<unsigned, mlir::BlockArgument> arguments;
  llvm::SmallVector<unsigned> participants{packet.tile};
  for (const auto &receiver : packet.receivers)
    if (!llvm::is_contained(participants, receiver.tile))
      participants.push_back(receiver.tile);
  llvm::sort(participants);
  for (unsigned index : participants) {
    mlir::ModuleOp module = modules[index];
    mlir::func::FuncOp entry;
    for (auto candidate : module.getOps<mlir::func::FuncOp>())
      if (!candidate.isPrivate() && !candidate.isExternal())
        entry = candidate;
    if (!entry)
      return fail(SharedDDRCompletionFailure::Contract,
                  "communication packet has no Tile entry");
    rewriter.setInsertionPointToStart(module.getBody());
    auto global = rewriter.create<mlir::memref::GlobalOp>(
        module.getLoc(), symbol, rewriter.getStringAttr("private"), packetType,
        mlir::Attribute{}, false, rewriter.getI64IntegerAttr(256));
    global->setAttr(kWaferDDRResourceAttrName,
                    DDRResourceAttr::get(rewriter.getContext(), resourceId));
    auto binding = DDRBindingAttr::get(
        rewriter.getContext(), resource, resourceId,
        index == packet.tile ? DDRAccess::Write : DDRAccess::Read);
    unsigned argument = entry.getNumArguments();
    entry.insertArgument(argument, packetType,
                         rewriter.getDictionaryAttr({rewriter.getNamedAttr(
                             kWaferDDRBindingAttrName, binding)}),
                         entry.getLoc());
    arguments.emplace(index, entry.getArgument(argument));
  }
  auto stride = rewriter.getDenseI64ArrayAttr({0, 0, 0});
  auto iterations = rewriter.getDenseI64ArrayAttr({1, 1, 1});
  auto sourceRegion = directRegion(packet.sender);
  auto dest = bindInput(sourceRegion, arguments.at(packet.tile));
  auto *producer = precedingBufferWrite(packet.sender, packet.source);
  if (producer)
    rewriter.setInsertionPointAfter(producer);
  else
    rewriter.setInsertionPointToStart(packet.sender->getBlock());
  rewriter.create<InstrWDMAOp>(
      packet.sender->getLoc(), packet.source, dest,
      rewriter.getI64IntegerAttr(packet.bytes),
      rewriter.getI64IntegerAttr(packet.bytes),
      rewriter.getI64IntegerAttr(packet.sourceOffset),
      rewriter.getI64IntegerAttr(0), stride, iterations,
      NCCWorkerAttr::get(rewriter.getContext(), NCCWorker::Worker0));
  for (const auto &receiver : packet.receivers) {
    auto recv = receiver.operation;
    auto source = bindInput(directRegion(recv), arguments.at(receiver.tile));
    auto *consumer = nextBufferAccess(recv, recv.getBuffer());
    if (!consumer)
      return fail(SharedDDRCompletionFailure::Contract,
                  "communication receiver has no current access cut");
    rewriter.setInsertionPoint(consumer);
    rewriter.create<InstrRDMAOp>(
        recv.getLoc(), source, recv.getBuffer(),
        rewriter.getI64IntegerAttr(packet.pieceBytes),
        rewriter.getI64IntegerAttr(packet.pieceBytes),
        rewriter.getI64IntegerAttr(receiver.packetOffset),
        rewriter.getI64IntegerAttr(recv.getBufferOffset().value_or(0)), stride,
        iterations,
        NCCWorkerAttr::get(rewriter.getContext(), NCCWorker::Worker0));
    eraseWaits(recv.getToken(), rewriter);
    rewriter.eraseOp(recv);
  }
  eraseWaits(packet.sender->getResult(0), rewriter);
  rewriter.eraseOp(packet.sender);
  return materializeSharedDDRResourceCompletion(modules, resourceId);
}

} // namespace

CommunicationConstructionResult
constructCommunication(llvm::ArrayRef<mlir::ModuleOp> modules,
                       llvm::ArrayRef<TileId> tileIds) {
  support::ScopedCompileTimingSpan timing(
      "construction", "communication-proposal", "current-instr");
  CommunicationConstructionResult result;
  if (modules.empty() || modules.size() != tileIds.size() ||
      llvm::any_of(modules, [](mlir::ModuleOp module) {
        return !module || mlir::failed(mlir::verify(module));
      })) {
    result.outcome =
        fail(SharedDDRCompletionFailure::Contract,
             "communication construction requires verified Tile modules");
    return result;
  }
  result.outcome = materializeSharedDDRNotifications(modules);
  if (!result.outcome.succeeded())
    return result;
  constexpr uint64_t maximumWork = 67108864;
  uint64_t work = 0;
  bool complete = false;
  // Only complete ready exchanges enter a proposed order. A blocked prefix
  // exposes current endpoints at which another representation may advance.
  // Representation choices are applied in batches; no wait graph is used to
  // select a message, and no incomplete order is passed to the memory leaf.
  for (unsigned branch = 0; branch < 64; ++branch) {
    auto scheduled =
        scheduleCurrentCommunication(modules, tileIds, maximumWork - work);
    work += scheduled.work;
    support::addCompileCounter("communication-construction", "frontier-queries",
                               1);
    if (scheduled.status == CommunicationSchedulingStatus::Scheduled) {
      result.statistics.issuePlacements += scheduled.movedIssues;
      complete = true;
      break;
    }
    if (scheduled.status != CommunicationSchedulingStatus::Blocked) {
      result.outcome =
          fail(scheduled.status == CommunicationSchedulingStatus::WorkLimit
                   ? SharedDDRCompletionFailure::Indeterminate
               : scheduled.status == CommunicationSchedulingStatus::Unsupported
                   ? SharedDDRCompletionFailure::Unsupported
                   : SharedDDRCompletionFailure::Contract,
               scheduled.detail);
      return result;
    }
    auto packets = collectPackets(modules, tileIds);
    if (mlir::failed(packets)) {
      result.outcome = fail(SharedDDRCompletionFailure::Contract,
                            "communication has unmatched current messages");
      return result;
    }
    llvm::DenseSet<mlir::Operation *> sources(scheduled.readySenders.begin(),
                                              scheduled.readySenders.end());
    llvm::DenseSet<mlir::Operation *> destinations(
        scheduled.readyReceivers.begin(), scheduled.readyReceivers.end());
    uint64_t expanded = 0;
    for (const Packet &packet : *packets) {
      const bool receiveReady =
          llvm::all_of(packet.receivers, [&](const Receiver &receiver) {
            return destinations.contains(receiver.operation.operator->());
          });
      if (receiveReady || sources.contains(packet.sender)) {
        auto local = restoreLocalInput(packet, modules);
        if (mlir::failed(local)) {
          result.outcome =
              fail(SharedDDRCompletionFailure::Contract,
                   "ready input representation failed materialization");
          return result;
        }
        if (*local) {
          ++result.statistics.localInputReads;
          result.statistics.replacedMessages += packet.receivers.size();
          ++expanded;
          continue;
        }
      }
      if (!sources.contains(packet.sender))
        continue;
      result.outcome = useDDRPacket(packet, modules);
      if (!result.outcome.succeeded())
        return result;
      ++result.statistics.ddrPackets;
      result.statistics.replacedMessages += packet.receivers.size();
      ++expanded;
    }
    if (!expanded) {
      result.outcome = fail(
          SharedDDRCompletionFailure::Unsupported,
          "current fixed dependencies admit no ready communication extension");
      return result;
    }
  }
  if (!complete) {
    result.outcome = fail(SharedDDRCompletionFailure::Indeterminate,
                          "communication construction branch budget exhausted");
    return result;
  }
  auto waits = rebuildRequiredDirectDTEWaits(modules);
  if (!waits.succeeded()) {
    result.outcome = fail(SharedDDRCompletionFailure::Contract, waits.detail);
    return result;
  }
  auto order = analyzeCurrentCommunicationOrder(modules, tileIds);
  if (order.status != CommunicationOrderStatus::Acyclic) {
    result.outcome = fail(
        SharedDDRCompletionFailure::Contract,
        "constructed communication order failed independent verification: " +
            order.detail);
    return result;
  }
  result.outcome = {};
  for (auto module : modules)
    if (mlir::failed(mlir::verify(module))) {
      result.outcome =
          fail(SharedDDRCompletionFailure::Contract,
               "communication construction produced invalid current IR");
      return result;
    }
  support::addCompileCounter("communication-proposal", "local-input-reads",
                             result.statistics.localInputReads);
  support::addCompileCounter("communication-proposal", "ddr-packets",
                             result.statistics.ddrPackets);
  support::addCompileCounter("communication-proposal", "replaced-messages",
                             result.statistics.replacedMessages);
  support::addCompileCounter("communication-proposal", "issue-placements",
                             result.statistics.issuePlacements);
  return result;
}

} // namespace wafer::compiler::detail
