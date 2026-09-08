//===- SharedDDRCompletion.cpp - Shared DDR publication
//--------------------===//
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <set>

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

// A DTE-connected residency component must progress within one device phase.
// Collapse its Regions before adding DDR publication edges and local order.
// This prevents a DDR acquire from blocking a Tile needed by a peer issue.
static Result verifyOrder(const Collection &collection,
                          llvm::ArrayRef<TileId> tileIds) {
  if (tileIds.size() != collection.entries.size())
    return contract("shared DDR completion Tile identity domain differs");
  llvm::DenseMap<mlir::Operation *, unsigned> nodes;
  llvm::SmallVector<TileRegionOp> regions;
  for (auto entry : collection.entries)
    for (auto region : entry.getBody().front().getOps<TileRegionOp>()) {
      nodes[region] = regions.size();
      regions.push_back(region);
    }
  llvm::SmallVector<unsigned> parent(regions.size());
  for (unsigned i = 0; i < parent.size(); ++i)
    parent[i] = i;
  auto root = [&](unsigned i) {
    while (parent[i] != i)
      i = parent[i];
    return i;
  };
  struct Endpoint {
    int64_t tile;
    int64_t peer;
    DTEMessageAttr message;
    unsigned node;
  };
  llvm::SmallVector<Endpoint> sends, receives;
  bool unknownDTERegion = false;
  for (auto [index, entryRef] : llvm::enumerate(collection.entries)) {
    mlir::func::FuncOp entry = entryRef;
    auto add = [&](mlir::Operation *op, int64_t peer, DTEMessageAttr message,
                   bool send) {
      auto region = getTopLevelRegion(op);
      if (region)
        (send ? sends : receives)
            .push_back({tileIds[index].getValue(), peer, message,
                        nodes.lookup(region)});
      else
        unknownDTERegion = true;
    };
    entry.walk([&](InstrDTESendOp op) {
      add(op, op.getPeer(), op.getMessage(), true);
    });
    entry.walk([&](InstrDTERecvOp op) {
      add(op, op.getPeer(), op.getMessage(), false);
    });
    auto addMultiSend = [&](auto op) {
      for (auto [peer, message] :
           llvm::zip_equal(op.getPeers(), op.getMessages()))
        add(op, peer, mlir::cast<DTEMessageAttr>(message), true);
    };
    entry.walk([&](InstrDTEBroadcastOp op) { addMultiSend(op); });
    entry.walk([&](InstrDTEScatterOp op) { addMultiSend(op); });
  }
  if (unknownDTERegion && !collection.resources.empty())
    return unsupported(
        "shared DDR order cannot resolve a Direct DTE execution region");
  for (const Endpoint &send : sends) {
    auto receive = llvm::find_if(receives, [&](const Endpoint &recv) {
      return recv.tile == send.peer && recv.peer == send.tile &&
             recv.message == send.message;
    });
    if (receive == receives.end())
      return contract("shared DDR order has an unmatched Direct DTE endpoint");
    parent[root(send.node)] = root(receive->node);
  }
  std::set<std::pair<unsigned, unsigned>> edges;
  auto edge = [&](TileRegionOp a, TileRegionOp b, bool strict) {
    unsigned from = root(nodes.lookup(a)), to = root(nodes.lookup(b));
    if (from == to)
      return !strict;
    edges.insert({from, to});
    return true;
  };
  for (auto entry : collection.entries) {
    TileRegionOp previous;
    for (auto region : entry.getBody().front().getOps<TileRegionOp>()) {
      if (previous)
        edge(previous, region, false);
      previous = region;
    }
  }
  for (const auto &[id, resource] : collection.resources)
    for (const Access &reader : resource.readers)
      if (!edge(resource.writer->region, reader.region, true))
        return unsupported("shared DDR publication crosses a mutually "
                           "dependent DTE component");
  llvm::SmallVector<unsigned> degree(regions.size(), 0);
  for (auto [from, to] : edges)
    ++degree[to];
  llvm::SmallVector<unsigned> ready;
  unsigned count = 0;
  for (unsigned i = 0; i < parent.size(); ++i)
    if (root(i) == i) {
      ++count;
      if (!degree[i])
        ready.push_back(i);
    }
  for (unsigned i = 0; i < ready.size(); ++i)
    for (auto [from, to] : edges)
      if (from == ready[i] && --degree[to] == 0)
        ready.push_back(to);
  return ready.size() == count ? Result{}
                               : unsupported("shared DDR and DTE completion "
                                             "dependencies contain a cycle");
}

static bool isZeroInitialized(mlir::BlockArgument argument) {
  auto binding = getBinding(argument);
  if (!binding)
    return false;
  auto global =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
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
  llvm::DenseSet<mlir::Operation *> checked;
  for (auto &[id, resource] : collection.resources) {
    SyncDDRPublishOp publish;
    auto writer = mlir::cast<mlir::func::FuncOp>(
        resource.writer->root.getOwner()->getParentOp());
    unsigned count = 0;
    writer.walk([&](SyncDDRPublishOp op) {
      if (getEntryRoot(op.getData()) == resource.writer->root) {
        publish = op;
        ++count;
      }
    });
    if (count != 1 ||
        publish->getBlock() != resource.writer->region->getBlock() ||
        !resource.writer->region->isBeforeInBlock(publish))
      return contract("shared DDR writer requires exactly one publication "
                      "after its writes");
    auto ready = getEntryRoot(publish.getReady());
    auto readyBinding = getBinding(ready);
    if (!ready || !readyBinding || !isZeroInitialized(ready) ||
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
      auto function =
          mlir::cast<mlir::func::FuncOp>(reader.root.getOwner()->getParentOp());
      unsigned acquired = 0;
      function.walk([&](SyncDDRAcquireOp op) {
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
            isZeroInitialized(getEntryRoot(op.getReady())) && beforeAllReads) {
          ++acquired;
          checked.insert(op);
        }
      });
      if (acquired != 1)
        return contract("shared DDR reader has no matching acquisition before "
                        "its first DMA");
    }
    for (auto entry : collection.entries) {
      unsigned bindings = 0;
      for (auto argument : entry.getArguments()) {
        auto binding = getBinding(argument);
        if (!binding || binding.getResourceId() != readyBinding.getResourceId())
          continue;
        ++bindings;
        if (argument.getType() != ready.getType() ||
            !isZeroInitialized(argument))
          return contract(
              "shared DDR completion initialization differs across Tiles");
        for (auto *user : argument.getUsers())
          if (!checked.contains(user))
            return contract(
                "shared DDR completion storage has an unrelated access");
      }
      if (bindings != 1)
        return contract(
            "shared DDR completion requires one binding on every Tile");
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
  result = verifyOrder(collection, tileIds);
  if (!result.succeeded())
    return result;
  bool existing = false;
  for (auto entry : collection.entries)
    entry.walk([&](mlir::Operation *op) {
      existing |= mlir::isa<SyncDDRPublishOp, SyncDDRAcquireOp>(op);
    });
  if (existing)
    return verifySharedDDRCompletion(modules, tileIds);
  int64_t nextResource = 0;
  for (auto module : modules)
    for (auto global : module.getOps<mlir::memref::GlobalOp>())
      if (auto resource =
              global->getAttrOfType<DDRResourceAttr>(kWaferDDRResourceAttrName))
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
          context, mlir::FlatSymbolRefAttr::get(context, symbol), nextResource,
          access);
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
  for (auto module : modules)
    if (mlir::failed(mlir::verify(module)))
      return contract(
          "shared DDR completion materialization produced invalid IR");
  return verifySharedDDRCompletion(modules, tileIds);
}
} // namespace wafer
