//===- CommunicationScheduling.cpp - Ready exchange group scheduling -----===//

#include "CommunicationScheduling.h"
#include "DirectDTETransport.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/DirectDTE.h"
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace wafer::compiler::detail {
namespace {
using Status = CommunicationSchedulingStatus;
using Key = std::tuple<uint64_t, uint64_t, int64_t, int64_t, int64_t>;

Key key(uint64_t src, uint64_t dst, DTEMessageAttr message) {
  return {src, dst, message.getCommunicationId(), message.getRound(),
          message.getPayloadSlice()};
}

bool isSend(mlir::Operation *op) {
  return mlir::isa<InstrDTESendOp, InstrDTEBroadcastOp, InstrDTEScatterOp>(op);
}
bool isIssue(mlir::Operation *op) {
  return isSend(op) || mlir::isa<InstrDTERecvOp>(op);
}

struct Node {
  llvm::SmallVector<mlir::Operation *, 4> operations;
  llvm::SmallVector<unsigned> next;
  unsigned degree = 0;
  bool exchange = false;
};

struct Access {
  unsigned node;
  mlir::Operation *operation;
  mlir::Value value;
  bool write;
};

class Scheduler {
public:
  Scheduler(llvm::ArrayRef<mlir::ModuleOp> modules,
            llvm::ArrayRef<TileId> tileIds, uint64_t maximumWork)
      : modules(modules), tileIds(tileIds), maximumWork(maximumWork) {}

  CommunicationSchedulingResult run() {
    if (modules.size() != tileIds.size() || modules.empty())
      return failure(Status::Contract, "communication Tile domain differs");
    if (!collectEntries() || !collectExchanges() || !collectLocalDependencies())
      return result;
    for (auto [resource, acquire] : acquisitions) {
      auto found = publications.find(resource);
      if (found == publications.end())
        return failure(Status::Contract, "DDR acquire has no actual publisher");
      edge(found->second, nodeOf.lookup(acquire), acquire);
    }
    if (result.status == Status::WorkLimit)
      return result;

    std::set<unsigned> local, exchanges;
    for (unsigned index = 0; index < nodes.size(); ++index)
      if (!nodes[index].degree)
        (nodes[index].exchange ? exchanges : local).insert(index);
    llvm::SmallVector<bool> scheduled(nodes.size(), false);
    llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Operation *>> order;
    auto finish = [&](unsigned index) {
      scheduled[index] = true;
      for (unsigned next : nodes[index].next)
        if (--nodes[next].degree == 0)
          (nodes[next].exchange ? exchanges : local).insert(next);
    };
    size_t count = 0;
    while (!local.empty() || !exchanges.empty()) {
      while (!local.empty()) {
        unsigned index = *local.begin();
        local.erase(local.begin());
        for (auto *op : nodes[index].operations)
          order[op->getBlock()].push_back(op);
        finish(index);
        ++count;
      }
      if (exchanges.empty())
        continue;
      llvm::SmallVector<unsigned> group;
      llvm::DenseSet<unsigned> senders;
      llvm::DenseMap<unsigned, unsigned> receivers;
      llvm::DenseSet<std::pair<unsigned, unsigned>> peers;
      for (unsigned index : exchanges) {
        if (!spend())
          return result;
        auto &node = nodes[index];
        unsigned sender = tileOf.lookup(node.operations.front());
        if (senders.contains(sender))
          continue;
        bool available = true;
        for (auto *recv : llvm::drop_begin(node.operations)) {
          unsigned receiver = tileOf.lookup(recv);
          available &= receivers.lookup(receiver) <
                           TargetDirectDTEResourceLimits::receiverFSMsPerTile &&
                       !peers.contains({sender, receiver});
        }
        if (!available)
          continue;
        group.push_back(index);
        senders.insert(sender);
        for (auto *recv : llvm::drop_begin(node.operations)) {
          unsigned receiver = tileOf.lookup(recv);
          ++receivers[receiver];
          peers.insert({sender, receiver});
        }
      }
      if (group.empty())
        return failure(Status::Unsupported,
                       "ready exchange exceeds current endpoint resources");
      // Every source and every receiver is ready before any pair is committed.
      // Projecting prepare-before-issue to each block permits logical rings.
      for (unsigned index : group)
        for (auto *recv : llvm::drop_begin(nodes[index].operations))
          order[recv->getBlock()].push_back(recv);
      for (unsigned index : group) {
        auto *send = nodes[index].operations.front();
        order[send->getBlock()].push_back(send);
      }
      for (unsigned index : group) {
        exchanges.erase(index);
        finish(index);
        ++count;
      }
      ++result.groups;
    }
    if (count != nodes.size()) {
      result.status = Status::Blocked;
      result.detail = "no complete exchange or local operation is ready";
      for (const auto &node : nodes) {
        if (!node.exchange || scheduled[nodeOf.lookup(node.operations.front())])
          continue;
        for (auto *op : node.operations) {
          const auto &requirements = endpointDependencies[op];
          if (llvm::all_of(requirements, [&](unsigned predecessor) {
                return scheduled[predecessor];
              }))
            (isSend(op) ? result.readySenders : result.readyReceivers)
                .push_back(op);
        }
      }
      return result;
    }

    mlir::ModuleOp first = modules.front();
    mlir::IRRewriter rewriter(first.getContext());
    for (mlir::Block *block : blocks) {
      llvm::SmallVector<mlir::Operation *> before, after;
      for (auto &op : *block)
        if (!mlir::isa<InstrDTEWaitOp>(op) && &op != block->getTerminator())
          before.push_back(&op);
      for (auto *op : order[block])
        if (op != block->getTerminator())
          after.push_back(op);
      if (before.size() != after.size())
        return failure(Status::Contract,
                       "communication order omitted a current operation");
      for (auto [old, selected] : llvm::zip_equal(before, after))
        result.movedIssues += isIssue(selected) && old != selected;
    }
    for (auto wait : oldWaits)
      rewriter.eraseOp(wait);
    for (mlir::Block *block : blocks) {
      auto found = order.find(block);
      if (found == order.end())
        continue;
      auto *terminator = block->getTerminator();
      for (auto *op : found->second) {
        if (op == terminator)
          continue;
        rewriter.moveOpBefore(op, terminator);
      }
    }
    result.status = Status::Scheduled;
    return result;
  }

private:
  CommunicationSchedulingResult failure(Status status, llvm::StringRef detail) {
    result.status = status;
    result.detail = detail.str();
    return result;
  }
  bool spend(uint64_t amount = 1) {
    if (amount > maximumWork - std::min(result.work, maximumWork)) {
      failure(Status::WorkLimit,
              "communication construction work limit reached");
      return false;
    }
    result.work += amount;
    return true;
  }
  unsigned addNode(mlir::Operation *operation, bool exchange = false) {
    unsigned index = nodes.size();
    nodes.push_back({{}, {}, 0, exchange});
    if (operation) {
      nodes.back().operations.push_back(operation);
      nodeOf[operation] = index;
    }
    return index;
  }
  void edge(unsigned from, unsigned to, mlir::Operation *endpoint = nullptr) {
    if (!spend())
      return;
    if (endpoint && isIssue(endpoint))
      endpointDependencies[endpoint].insert(from);
    if (!llvm::is_contained(nodes[from].next, to)) {
      nodes[from].next.push_back(to);
      ++nodes[to].degree;
    }
  }
  bool collectEntries() {
    llvm::DenseSet<uint64_t> ids;
    for (auto [tile, moduleRef] : llvm::enumerate(modules)) {
      mlir::ModuleOp module = moduleRef;
      if (!ids.insert(tileIds[tile].getValue()).second) {
        failure(Status::Contract,
                "communication has duplicate Tile identities");
        return false;
      }
      mlir::func::FuncOp entry;
      for (auto function : module.getOps<mlir::func::FuncOp>())
        if (!function.isPrivate() && !function.isExternal()) {
          if (entry) {
            failure(Status::Contract, "communication requires one Tile entry");
            return false;
          }
          entry = function;
        }
      if (!entry || !entry.getBody().hasOneBlock()) {
        failure(Status::Unsupported,
                "communication requires single-block entries");
        return false;
      }
      entries.push_back(entry);
      bool valid = true;
      bool malformedToken = false;
      entry.walk([&](mlir::Operation *op) {
        tileOf[op] = tile;
        if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op)) {
          malformedToken |=
              llvm::any_of(wait.getTokens(), [](mlir::Value token) {
                auto *definition = token.getDefiningOp();
                return !definition || !isIssue(definition);
              });
          oldWaits.push_back(wait);
          return;
        }
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
          auto callee =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                  call, call.getCalleeAttr());
          if (!callee || callee.isExternal())
            valid = false;
          else
            callee.walk([&](mlir::Operation *nested) {
              if (isIssue(nested) ||
                  mlir::isa<SyncDDRPublishOp, SyncDDRAcquireOp,
                            mlir::func::CallOp>(nested))
                valid = false;
            });
        }
        if (!isIssue(op) && !mlir::isa<SyncDDRPublishOp, SyncDDRAcquireOp>(op))
          return;
        auto region = mlir::dyn_cast_or_null<TileRegionOp>(op->getParentOp());
        if (!region || region->getBlock() != &entry.getBody().front())
          valid = false;
        if (isIssue(op)) {
          if (llvm::any_of(op->getResult(0).getUsers(),
                           [](mlir::Operation *user) {
                             return !mlir::isa<InstrDTEWaitOp>(user);
                           }))
            valid = false;
          if (auto send = mlir::dyn_cast<InstrDTESendOp>(op))
            valid &= !send.getBinding() && !send.getBindingSelector();
          if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(op))
            valid &= !recv.getBinding() && !recv.getBindingSelector();
        }
      });
      if (malformedToken) {
        failure(Status::Contract, "communication wait has a non-DTE token");
        return false;
      }
      if (!valid) {
        failure(Status::Unsupported, "communication construction requires "
                                     "unbound single-execution issues");
        return false;
      }
    }
    return true;
  }
  bool collectExchanges() {
    std::map<Key, mlir::Operation *> receives;
    for (auto [tile, entryRef] : llvm::enumerate(entries)) {
      auto entry = entryRef;
      bool duplicate = false;
      entry.walk([&](InstrDTERecvOp recv) {
        duplicate |= !receives
                          .emplace(key(recv.getPeer(), tileIds[tile].getValue(),
                                       recv.getMessage()),
                                   recv)
                          .second;
      });
      if (duplicate) {
        failure(Status::Contract,
                "communication has duplicate receive identities");
        return false;
      }
    }
    for (auto [tile, entryRef] : llvm::enumerate(entries)) {
      auto entry = entryRef;
      bool valid = true;
      entry.walk([&](mlir::Operation *op) {
        if (!isSend(op) || !valid)
          return;
        unsigned index = addNode(op, true);
        llvm::SmallVector<std::pair<uint64_t, DTEMessageAttr>> endpoints;
        int64_t bytes = 0;
        if (auto send = mlir::dyn_cast<InstrDTESendOp>(op)) {
          endpoints.emplace_back(send.getPeer(), send.getMessage());
          bytes = send.getBytes();
        } else {
          auto append = [&](auto send) {
            bytes = send.getBytes();
            for (auto [peer, message] :
                 llvm::zip_equal(send.getPeers(), send.getMessages()))
              endpoints.emplace_back(peer, mlir::cast<DTEMessageAttr>(message));
          };
          if (auto send = mlir::dyn_cast<InstrDTEBroadcastOp>(op))
            append(send);
          else
            append(mlir::cast<InstrDTEScatterOp>(op));
        }
        for (auto [peer, message] : endpoints) {
          auto found =
              receives.find(key(tileIds[tile].getValue(), peer, message));
          if (found == receives.end() || nodeOf.count(found->second) ||
              mlir::cast<InstrDTERecvOp>(found->second).getBytes() !=
                  static_cast<uint64_t>(bytes)) {
            valid = false;
            break;
          }
          nodeOf[found->second] = index;
          nodes[index].operations.push_back(found->second);
        }
      });
      if (!valid) {
        failure(Status::Contract,
                "communication has unmatched endpoints or payloads");
        return false;
      }
    }
    for (auto [identity, receive] : receives)
      if (!nodeOf.count(receive)) {
        failure(Status::Contract, "communication receive has no sender");
        return false;
      }
    return true;
  }

  bool collectBlock(mlir::Block &block, unsigned begin, unsigned end) {
    blocks.push_back(&block);
    llvm::SmallVector<mlir::Operation *> operations;
    for (auto &op : block) {
      if (mlir::isa<InstrDTEWaitOp>(op))
        continue;
      operations.push_back(&op);
      if (!nodeOf.count(&op))
        addNode(&op);
    }
    unsigned previous = begin;
    for (auto *op : operations) {
      unsigned index = nodeOf.lookup(op);
      edge(begin, index, op);
      edge(index, end);
      if (!isIssue(op)) {
        edge(previous, index);
        previous = index;
      }
      if (auto publish = mlir::dyn_cast<SyncDDRPublishOp>(op)) {
        auto binding = getSharedDDRBinding(publish.getData());
        if (!binding ||
            !publications.emplace(binding.getResourceId(), index).second) {
          failure(Status::Contract, "DDR publication has no unique binding");
          return false;
        }
      }
      if (auto acquire = mlir::dyn_cast<SyncDDRAcquireOp>(op)) {
        auto binding = getSharedDDRBinding(acquire.getData());
        if (!binding) {
          failure(Status::Contract, "DDR acquisition has no binding");
          return false;
        }
        acquisitions.emplace_back(binding.getResourceId(), op);
      }
      // Nested local loops are opaque current operations. Their operand and
      // effect closure constrain issues without unrolling dynamic executions.
      op->walk([&](mlir::Operation *nested) {
        for (auto operand : nested->getOperands()) {
          auto *definition = operand.getDefiningOp();
          while (definition && definition->getBlock() != &block)
            definition = definition->getParentOp();
          if (definition && definition != op && nodeOf.count(definition))
            edge(nodeOf.lookup(definition), index, op);
        }
      });
    }
    edge(previous, end);

    struct RootAccesses {
      llvm::SmallVector<Access> fixed;
      llvm::SmallVector<Access> issues;
    };
    llvm::DenseMap<mlir::Value, RootAccesses> byRoot;
    llvm::SmallVector<mlir::Operation *> barriers;
    for (auto *op : operations) {
      if (mlir::isa<mlir::ViewLikeOpInterface>(op))
        continue;
      auto effects = mlir::getEffectsRecursively(op);
      ++result.effectSummaries;
      bool unknown = !effects;
      if (effects)
        for (const auto &effect : *effects)
          unknown |=
              !effect.getValue() &&
              effect.getResource() == mlir::SideEffects::DefaultResource::get();
      if (unknown) {
        barriers.push_back(op);
        continue;
      }
      unsigned index = nodeOf.lookup(op);
      llvm::DenseSet<std::pair<mlir::Value, unsigned>> distinctAccesses;
      for (const auto &effect : *effects) {
        auto value = effect.getValue();
        if (!value || !mlir::isa<mlir::BaseMemRefType>(value.getType()))
          continue;
        bool write = !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        if (!distinctAccesses.insert({value, write}).second)
          continue;
        for (mlir::Value root : roots.getStorageRoots(value)) {
          auto &accesses = byRoot[root];
          auto constrain = [&](llvm::ArrayRef<Access> previous) {
            for (const auto &prior : previous) {
              if (!spend())
                return false;
              if (prior.operation == op || (!write && !prior.write) ||
                  haveDisjointCommunicationRanges(prior.operation, prior.value,
                                                  op, value))
                continue;
              edge(prior.node, index, op);
            }
            return true;
          };
          // Fixed operations already have the complete original local order.
          // Only issue/fixed and issue/issue pairs add constraints; scanning
          // all fixed/fixed accesses here was quadratic on fused Regions.
          const bool issue = isIssue(op);
          if (!constrain(accesses.issues) ||
              (issue && !constrain(accesses.fixed)))
            return false;
          (issue ? accesses.issues : accesses.fixed)
              .push_back({index, op, value, write});
        }
      }
    }
    for (auto *barrier : barriers)
      for (auto *op : operations)
        if (isIssue(op)) {
          if (op->isBeforeInBlock(barrier))
            edge(nodeOf.lookup(op), nodeOf.lookup(barrier));
          else
            edge(nodeOf.lookup(barrier), nodeOf.lookup(op), op);
        }
    return result.status != Status::WorkLimit;
  }

  bool collectLocalDependencies() {
    for (auto entryRef : entries) {
      auto entry = entryRef;
      // Existing Region sequence is a scope boundary, not an invented global
      // barrier. No issue crosses its allocation's current owning Region.
      unsigned previous = addNode(nullptr);
      for (auto &op : entry.getBody().front()) {
        if (auto region = mlir::dyn_cast<TileRegionOp>(op)) {
          if (!region.getBody().hasOneBlock()) {
            failure(Status::Unsupported,
                    "communication requires single-block Regions");
            return false;
          }
          unsigned begin = addNode(nullptr), end = addNode(nullptr);
          edge(previous, begin);
          if (!collectBlock(region.getBody().front(), begin, end))
            return false;
          previous = end;
        } else {
          unsigned node = addNode(&op);
          edge(previous, node);
          previous = node;
        }
      }
    }
    return true;
  }

  llvm::ArrayRef<mlir::ModuleOp> modules;
  llvm::ArrayRef<TileId> tileIds;
  uint64_t maximumWork;
  CommunicationSchedulingResult result;
  llvm::SmallVector<mlir::func::FuncOp> entries;
  llvm::SmallVector<Node> nodes;
  llvm::DenseMap<mlir::Operation *, unsigned> nodeOf, tileOf;
  llvm::DenseMap<mlir::Operation *, llvm::SmallSetVector<unsigned, 4>>
      endpointDependencies;
  llvm::SmallVector<mlir::Block *> blocks;
  llvm::SmallVector<InstrDTEWaitOp> oldWaits;
  std::map<int64_t, unsigned> publications;
  llvm::SmallVector<std::pair<int64_t, mlir::Operation *>> acquisitions;
  StorageRootMemo roots;
};
} // namespace

CommunicationSchedulingResult
scheduleCurrentCommunication(llvm::ArrayRef<mlir::ModuleOp> modules,
                             llvm::ArrayRef<TileId> tileIds,
                             uint64_t maximumWork) {
  support::ScopedCompileTimingSpan timing("construction", "communication",
                                          "ready-groups");
  auto result = Scheduler(modules, tileIds, maximumWork).run();
  support::addCompileCounter("communication-construction", "query-work",
                             result.work);
  support::addCompileCounter("communication-construction", "ready-groups",
                             result.groups);
  support::addCompileCounter("communication-construction", "effect-summaries",
                             result.effectSummaries);
  return result;
}
} // namespace wafer::compiler::detail
