//===- SharedDDRCompletion.cpp - Shared DDR publication
//--------------------===//
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <map>
#include <set>
#include <tuple>

namespace wafer {
namespace {
using Result = SharedDDRCompletionResult;
static Result contract(llvm::StringRef detail) {
  return {SharedDDRCompletionFailure::Contract, detail.str()};
}
static Result unsupported(llvm::StringRef detail) {
  return {SharedDDRCompletionFailure::Unsupported, detail.str()};
}

static mlir::BlockArgument getEntryRoot(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (mlir::isa<mlir::func::FuncOp>(arg.getOwner()->getParentOp()))
        return arg;
      auto flow =
          analysis::getSingleExecutionRegionFlow(arg.getOwner()->getParentOp());
      if (!flow)
        return {};
      auto found = llvm::find(flow->entryArguments, arg);
      if (found == flow->entryArguments.end())
        return {};
      value = flow->entryOperands[found - flow->entryArguments.begin()];
      continue;
    }
    if (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
            value.getDefiningOp())) {
      value = view.getViewSource();
      continue;
    }
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      auto flow = analysis::getSingleExecutionRegionFlow(result.getOwner());
      if (flow) {
        value = flow->exitOperands[result.getResultNumber()];
        continue;
      }
    }
    return {};
  }
  return {};
}

static DDRBindingAttr getBinding(mlir::Value value) {
  auto arg = getEntryRoot(value);
  if (!arg)
    return {};
  return mlir::cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp())
      .getArgAttrOfType<DDRBindingAttr>(arg.getArgNumber(),
                                        kWaferDDRBindingAttrName);
}

struct Access {
  mlir::BlockArgument root;
  TileRegionOp region;
};
struct Resource {
  std::optional<Access> writer;
  llvm::SmallVector<Access> readers;
};
struct Collection {
  std::map<int64_t, Resource> resources;
  llvm::SmallVector<mlir::func::FuncOp> entries;
};

static TileRegionOp getTopLevelRegion(mlir::Operation *op) {
  auto region = op->getParentOfType<TileRegionOp>();
  auto function = op->getParentOfType<mlir::func::FuncOp>();
  if (!region || !function || !function.getBody().hasOneBlock() ||
      region->getBlock() != &function.getBody().front())
    return {};
  return region;
}

// Analyze actual DMA operands, not declared access modes. Communication state
// itself is consumed by publication ops and never masquerades as a DMA payload.
static Result collect(llvm::ArrayRef<mlir::ModuleOp> modules, Collection &out) {
  support::ScopedCompileTimingSpan timing("completion-phase", "shared-ddr",
                                          "collect-accesses");
  for (mlir::ModuleOp module : modules) {
    mlir::func::FuncOp entry;
    for (auto function : module.getOps<mlir::func::FuncOp>()) {
      if (function.isPrivate() || function.isExternal())
        continue;
      if (entry)
        return contract("shared DDR completion requires one entry per Tile");
      entry = function;
    }
    if (!entry)
      return contract("shared DDR completion has no Tile entry");
    out.entries.push_back(entry);
    Result result;
    entry.walk([&](mlir::Operation *op) {
      if (!result.succeeded())
        return;
      if (auto getGlobal = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
        auto global =
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
                op, getGlobal.getNameAttr());
        if (global && global->hasAttr(kWaferDDRResourceAttrName)) {
          result =
              unsupported("shared DDR access must use its typed entry binding");
          return;
        }
      }
      // A shared argument cannot escape into an unknown alias/effect or a
      // repeated region. Reject at the operand boundary, before losing its
      // provenance in an unrecognized block argument or result.
      for (mlir::Value operand : op->getOperands()) {
        if (!getBinding(operand))
          continue;
        if (mlir::isa<InstrWDMAOp, InstrRDMAOp, SyncDDRPublishOp,
                      SyncDDRAcquireOp, mlir::ViewLikeOpInterface>(op) ||
            analysis::getSingleExecutionRegionFlow(op) ||
            (op->hasTrait<mlir::OpTrait::IsTerminator>() &&
             analysis::getSingleExecutionRegionFlow(op->getParentOp())))
          continue;
        result = unsupported("shared DDR operand has an unsupported alias, "
                             "effect or control-flow escape");
        return;
      }
      mlir::Value buffer;
      bool write = false;
      if (auto dma = mlir::dyn_cast<InstrWDMAOp>(op)) {
        buffer = dma.getDest();
        write = true;
      } else if (auto dma = mlir::dyn_cast<InstrRDMAOp>(op)) {
        buffer = dma.getSource();
      } else {
        return;
      }
      auto binding = getBinding(buffer);
      if (!binding)
        return;
      auto root = getEntryRoot(buffer);
      auto region = getTopLevelRegion(op);
      if (!region) {
        result = unsupported("shared DDR DMA requires an unconditional, "
                             "single-execution TileRegion");
        return;
      }
      Resource &resource = out.resources[binding.getResourceId()];
      if (write) {
        if (resource.writer && (resource.writer->root != root ||
                                resource.writer->region != region)) {
          result = unsupported("shared DDR publication requires one writer "
                               "Region without later overwrites");
          return;
        }
        resource.writer = Access{root, region};
      } else if (!llvm::any_of(resource.readers, [&](const Access &access) {
                   return access.root == root && access.region == region;
                 })) {
        resource.readers.push_back({root, region});
      }
    });
    if (!result.succeeded())
      return result;
  }
  for (auto &[id, resource] : out.resources) {
    if (!resource.writer || resource.readers.empty())
      return contract(
          "shared DDR DMA resource lacks an actual writer or reader");
    for (const Access &reader : resource.readers)
      if (reader.root.getOwner() == resource.writer->root.getOwner())
        return contract("shared DDR communication resource has a local reader");
  }
  return {};
}

// Preserve actual blocking points. A DTE-connected set of Regions is not an
// atomic phase: one Tile may publish DDR while another continues that exchange.
static Result verifyOrder(const Collection &collection,
                          llvm::ArrayRef<TileId> tileIds) {
  support::ScopedCompileTimingSpan timing("completion-phase", "shared-ddr",
                                          "verify-order");
  if (tileIds.size() != collection.entries.size())
    return contract("shared DDR completion Tile identity domain differs");
  if (collection.resources.empty())
    return {};
  auto isBlockingPoint = [](mlir::Operation *op) {
    return mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEBroadcastOp,
                     InstrDTEScatterOp, InstrDTEWaitOp, SyncDDRPublishOp,
                     SyncDDRAcquireOp>(op);
  };
  llvm::SmallVector<mlir::Operation *> operations;
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>> successors;
  llvm::SmallVector<unsigned> degree;
  auto edge = [&](unsigned from, unsigned to) {
    if (!llvm::is_contained(successors[from], to)) {
      successors[from].push_back(to);
      ++degree[to];
    }
  };
  using Message = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
  std::map<Message, unsigned> sends, receives;
  std::map<int64_t, unsigned> publishers;
  llvm::SmallVector<std::pair<int64_t, unsigned>> acquisitions;
  for (auto [index, entryRef] : llvm::enumerate(collection.entries)) {
    mlir::func::FuncOp entry = entryRef;
    std::optional<unsigned> previous;
    auto endpoint = [&](unsigned node, int64_t peer, DTEMessageAttr message,
                        bool send) {
      const int64_t local = tileIds[index].getValue();
      Message key{send ? local : peer, send ? peer : local,
                  message.getCommunicationId(), message.getRound(),
                  message.getPayloadSlice()};
      return (send ? sends : receives).try_emplace(key, node).second;
    };
    std::function<Result(mlir::Block &)> visit =
        [&](mlir::Block &block) -> Result {
      for (mlir::Operation &op : block) {
        if (isBlockingPoint(&op)) {
          unsigned node = operations.size();
          operations.push_back(&op);
          successors.emplace_back();
          degree.push_back(0);
          if (previous)
            edge(*previous, node);
          previous = node;
          bool unique = true;
          if (auto send = mlir::dyn_cast<InstrDTESendOp>(op))
            unique = endpoint(node, send.getPeer(), send.getMessage(), true);
          if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(op))
            unique = endpoint(node, recv.getPeer(), recv.getMessage(), false);
          auto multiSend = [&](auto send) {
            for (auto [peer, message] :
                 llvm::zip_equal(send.getPeers(), send.getMessages()))
              unique &= endpoint(node, peer,
                                 mlir::cast<DTEMessageAttr>(message), true);
          };
          if (auto send = mlir::dyn_cast<InstrDTEBroadcastOp>(op))
            multiSend(send);
          if (auto send = mlir::dyn_cast<InstrDTEScatterOp>(op))
            multiSend(send);
          if (!unique)
            return unsupported("shared DDR order requires unique "
                               "single-execution DTE messages");
          if (auto publish = mlir::dyn_cast<SyncDDRPublishOp>(op)) {
            auto binding = getBinding(publish.getData());
            if (!binding ||
                !publishers.try_emplace(binding.getResourceId(), node).second)
              return contract(
                  "shared DDR order requires a unique bound publisher");
          }
          if (auto acquire = mlir::dyn_cast<SyncDDRAcquireOp>(op)) {
            auto binding = getBinding(acquire.getData());
            if (!binding)
              return contract("shared DDR order has an unbound acquire");
            acquisitions.emplace_back(binding.getResourceId(), node);
          }
          continue;
        }
        if (auto flow = analysis::getSingleExecutionRegionFlow(&op)) {
          Result nested = visit(flow->region->front());
          if (!nested.succeeded())
            return nested;
          continue;
        }
        bool containsBlocking = false;
        op.walk([&](mlir::Operation *nested) {
          containsBlocking |=
              isBlockingPoint(nested) ||
              (nested != &op && mlir::isa<mlir::func::CallOp>(nested));
        });
        if (containsBlocking)
          return unsupported("shared DDR order cannot flatten repeated or "
                             "conditional communication");
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
          auto callee =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                  call, call.getCalleeAttr());
          if (!callee || callee.isExternal())
            return unsupported("shared DDR order has an unresolved call");
          callee.walk([&](mlir::Operation *nested) {
            containsBlocking |= isBlockingPoint(nested) ||
                                mlir::isa<mlir::func::CallOp>(nested);
          });
          if (containsBlocking)
            return unsupported(
                "shared DDR order requires inlined communication calls");
        }
      }
      return {};
    };
    Result result = visit(entry.getBody().front());
    if (!result.succeeded())
      return result;
  }
  if (sends.size() != receives.size())
    return contract("shared DDR order has unmatched Direct DTE endpoints");
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
      prerequisites;
  for (const auto &[message, send] : sends) {
    auto receive = receives.find(message);
    if (receive == receives.end())
      return contract("shared DDR order has an unmatched Direct DTE endpoint");
    unsigned recv = receive->second;
    // CRT send issue consumes peer-ready; waits need both endpoint issues.
    // Match the existing DirectDTETransport wait-graph contract exactly.
    edge(recv, send);
    for (unsigned node : {send, recv}) {
      prerequisites[operations[node]].push_back(send);
      prerequisites[operations[node]].push_back(recv);
    }
  }
  for (auto [index, op] : llvm::enumerate(operations)) {
    auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op);
    if (!wait)
      continue;
    for (mlir::Value token : wait.getTokens()) {
      llvm::DenseSet<mlir::Value> visited;
      while (token && visited.insert(token).second &&
             !prerequisites.count(token.getDefiningOp())) {
        if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(token))
          token = analysis::getSingleExecutionRegionEntryOperand(argument);
        else if (auto result = mlir::dyn_cast<mlir::OpResult>(token))
          token = analysis::getSingleExecutionRegionExitOperand(result);
        else
          token = {};
      }
      auto found = prerequisites.find(token ? token.getDefiningOp() : nullptr);
      if (found == prerequisites.end())
        return unsupported("shared DDR order cannot resolve a DTE wait token");
      for (unsigned issue : found->second)
        edge(issue, index);
    }
  }
  for (auto [resource, acquire] : acquisitions) {
    auto publish = publishers.find(resource);
    if (publish == publishers.end())
      return contract("shared DDR order has an acquire without publication");
    edge(publish->second, acquire);
  }
  llvm::SmallVector<unsigned> ready;
  for (unsigned node = 0; node < degree.size(); ++node)
    if (!degree[node])
      ready.push_back(node);
  for (unsigned index = 0; index < ready.size(); ++index)
    for (unsigned next : successors[ready[index]])
      if (--degree[next] == 0)
        ready.push_back(next);
  return ready.size() == operations.size()
             ? Result{}
             : unsupported("shared DDR and DTE completion dependencies contain "
                           "a cycle");
}

static bool isZeroInitialized(mlir::BlockArgument argument,
                              mlir::SymbolTableCollection &symbols) {
  auto binding = getBinding(argument);
  if (!binding)
    return false;
  auto global = symbols.lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
      argument.getOwner()->getParentOp(), binding.getResource());
  auto initial = global ? global.getInitialValue() : std::nullopt;
  auto dense = initial ? mlir::dyn_cast<mlir::DenseIntElementsAttr>(*initial)
                       : mlir::DenseIntElementsAttr{};
  return dense && dense.isSplat() &&
         dense.getSplatValue<llvm::APInt>().isZero();
}
} // namespace

Result verifySharedDDRCompletion(llvm::ArrayRef<mlir::ModuleOp> modules,
                                 llvm::ArrayRef<TileId> tileIds) {
  Collection collection;
  Result result = collect(modules, collection);
  if (!result.succeeded())
    return result;
  result = verifyOrder(collection, tileIds);
  if (!result.succeeded())
    return result;
  std::set<int64_t> readyResources;
  support::ScopedCompileTimingSpan timing(
      "completion-phase", "shared-ddr", "verify-publications",
      llvm::formatv("resources={0}", collection.resources.size()).str());
  // These indices describe only this read-only IR epoch. In particular, keep
  // duplicate operations/bindings so indexing cannot turn an invalid program
  // into an apparently unique publication.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<SyncDDRPublishOp, 1>> publishes;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<SyncDDRAcquireOp, 1>> acquires;
  using ResourceBindings =
      std::map<int64_t, llvm::SmallVector<mlir::BlockArgument, 1>>;
  llvm::SmallVector<ResourceBindings> entryBindings;
  mlir::SymbolTableCollection symbols;
  uint64_t indexedOperations = 0, indexedArguments = 0;
  for (auto entry : collection.entries) {
    entry.walk([&](mlir::Operation *op) {
      ++indexedOperations;
      if (auto publish = mlir::dyn_cast<SyncDDRPublishOp>(op))
        publishes[getEntryRoot(publish.getData())].push_back(publish);
      if (auto acquire = mlir::dyn_cast<SyncDDRAcquireOp>(op))
        acquires[acquire.getData()].push_back(acquire);
    });
    auto &bindings = entryBindings.emplace_back();
    for (auto argument : entry.getArguments()) {
      ++indexedArguments;
      if (auto binding = getBinding(argument))
        bindings[binding.getResourceId()].push_back(argument);
    }
  }
  support::addCompileCounter("shared-ddr", "indexed-operations",
                             indexedOperations);
  support::addCompileCounter("shared-ddr", "indexed-arguments",
                             indexedArguments);
  llvm::DenseSet<mlir::Operation *> checked;
  for (auto &[id, resource] : collection.resources) {
    auto publications = publishes.find(resource.writer->root);
    if (publications == publishes.end() || publications->second.size() != 1)
      return contract("shared DDR writer requires exactly one publication "
                      "after its writes");
    SyncDDRPublishOp publish = publications->second.front();
    if (publish->getBlock() != resource.writer->region->getBlock() ||
        !resource.writer->region->isBeforeInBlock(publish))
      return contract("shared DDR writer requires exactly one publication "
                      "after its writes");
    auto ready = getEntryRoot(publish.getReady());
    auto readyBinding = getBinding(ready);
    if (!ready || !readyBinding || !isZeroInitialized(ready, symbols) ||
        readyBinding.getResourceId() == id ||
        !readyResources.insert(readyBinding.getResourceId()).second)
      return contract("shared DDR publication requires distinct "
                      "zero-initialized completion storage");
    if (publish.getData() != resource.writer->root ||
        publish.getReady() != ready ||
        collection.resources.count(readyBinding.getResourceId()))
      return contract("shared DDR publication must use whole, disjoint data "
                      "and completion resources");
    checked.insert(publish);
    llvm::DenseSet<mlir::Block *> readersChecked;
    for (const Access &reader : resource.readers) {
      if (!readersChecked.insert(reader.root.getOwner()).second)
        continue;
      unsigned acquired = 0;
      auto acquisitions = acquires.find(reader.root);
      if (acquisitions == acquires.end())
        return contract("shared DDR reader has no matching acquisition before "
                        "its first DMA");
      for (SyncDDRAcquireOp op : acquisitions->second) {
        auto binding = getBinding(op.getReady());
        bool beforeAllReads =
            llvm::all_of(resource.readers, [&](const Access &access) {
              return access.root.getOwner() != reader.root.getOwner() ||
                     (op->getBlock() == access.region->getBlock() &&
                      op->isBeforeInBlock(access.region));
            });
        if (op.getData() == reader.root && binding &&
            op.getReady() == getEntryRoot(op.getReady()) &&
            binding.getResourceId() == readyBinding.getResourceId() &&
            isZeroInitialized(getEntryRoot(op.getReady()), symbols) &&
            beforeAllReads) {
          ++acquired;
          checked.insert(op);
        }
      }
      if (acquired != 1)
        return contract("shared DDR reader has no matching acquisition before "
                        "its first DMA");
    }
    for (const auto &bindings : entryBindings) {
      auto found = bindings.find(readyBinding.getResourceId());
      if (found == bindings.end() || found->second.size() != 1)
        return contract(
            "shared DDR completion requires one binding on every Tile");
      for (auto argument : found->second) {
        if (argument.getType() != ready.getType() ||
            !isZeroInitialized(argument, symbols))
          return contract(
              "shared DDR completion initialization differs across Tiles");
        for (auto *user : argument.getUsers())
          if (!checked.contains(user))
            return contract(
                "shared DDR completion storage has an unrelated access");
      }
    }
  }
  for (auto entry : collection.entries)
    entry.walk([&](mlir::Operation *op) {
      if (mlir::isa<SyncDDRPublishOp, SyncDDRAcquireOp>(op) &&
          !checked.contains(op))
        result = contract("shared DDR completion has an orphan or extra "
                          "publication/acquisition");
    });
  if (!result.succeeded())
    return result;
  return {};
}

Result materializeSharedDDRCompletion(llvm::ArrayRef<mlir::ModuleOp> modules,
                                      llvm::ArrayRef<TileId> tileIds) {
  Collection collection;
  Result result = collect(modules, collection);
  if (!result.succeeded() || collection.resources.empty())
    return result;
  bool existing = false;
  for (auto entry : collection.entries)
    entry.walk([&](mlir::Operation *op) {
      existing |= mlir::isa<SyncDDRPublishOp, SyncDDRAcquireOp>(op);
    });
  if (existing)
    return verifySharedDDRCompletion(modules, tileIds);
  {
    support::ScopedCompileTimingSpan timing(
        "completion-phase", "shared-ddr", "materialize-publications",
        llvm::formatv("resources={0}", collection.resources.size()).str());
    int64_t nextResource = 0;
    for (auto module : modules)
      for (auto global : module.getOps<mlir::memref::GlobalOp>())
        if (auto resource = global->getAttrOfType<DDRResourceAttr>(
                kWaferDDRResourceAttrName))
          nextResource = std::max(nextResource, resource.getResourceId() + 1);
    for (auto &[id, resource] : collection.resources) {
      auto *context = mlir::ModuleOp(modules.front()).getContext();
      mlir::OpBuilder builder(context);
      auto type = mlir::MemRefType::get(
          {64}, builder.getI8Type(), mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(context, MemorySpace::DDR, MemLayout::Tensor));
      auto initial = mlir::DenseIntElementsAttr::get(
          mlir::RankedTensorType::get({64}, builder.getI8Type()),
          llvm::ArrayRef<int8_t>{0});
      std::string symbol =
          "__wafer_ddr_completion_" + std::to_string(nextResource);
      for (auto [moduleRef, entry] :
           llvm::zip_equal(modules, collection.entries)) {
        mlir::ModuleOp module = moduleRef;
        builder.setInsertionPointToStart(module.getBody());
        auto global = builder.create<mlir::memref::GlobalOp>(
            module.getLoc(), symbol, builder.getStringAttr("private"), type,
            initial, false, builder.getI64IntegerAttr(64));
        global->setAttr(kWaferDDRResourceAttrName,
                        DDRResourceAttr::get(context, nextResource));
        auto dataRoot = resource.writer->root;
        bool writes = dataRoot.getOwner() == &entry.getBody().front();
        TileRegionOp firstRead;
        for (const Access &reader : resource.readers)
          if (reader.root.getOwner() == &entry.getBody().front() &&
              (!firstRead || reader.region->isBeforeInBlock(firstRead))) {
            firstRead = reader.region;
            dataRoot = reader.root;
          }
        DDRAccess access = writes      ? DDRAccess::Write
                           : firstRead ? DDRAccess::Read
                                       : DDRAccess::None;
        auto binding = DDRBindingAttr::get(
            context, mlir::FlatSymbolRefAttr::get(context, symbol),
            nextResource, access);
        unsigned index = entry.getNumArguments();
        entry.insertArgument(index, type,
                             builder.getDictionaryAttr({builder.getNamedAttr(
                                 kWaferDDRBindingAttrName, binding)}),
                             entry.getLoc());
        auto ready = entry.getArgument(index);
        if (writes) {
          builder.setInsertionPointAfter(resource.writer->region);
          builder.create<SyncDDRPublishOp>(entry.getLoc(), dataRoot, ready);
        }
        if (firstRead) {
          builder.setInsertionPoint(firstRead);
          builder.create<SyncDDRAcquireOp>(entry.getLoc(), dataRoot, ready);
        }
      }
      ++nextResource;
    }
  }
  for (auto module : modules)
    if (mlir::failed(mlir::verify(module)))
      return contract(
          "shared DDR completion materialization produced invalid IR");
  return verifySharedDDRCompletion(modules, tileIds);
}
} // namespace wafer
