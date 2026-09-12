//===- ReadOnlyInputSharing.cpp - Actual external input reuse ----------===//

#include "ReadOnlyInputSharing.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <optional>

namespace wafer::compiler::detail {
namespace {

struct InputWindow {
  int64_t argument;
  mlir::MemRefType sourceType;
  mlir::MemRefType payloadType;
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  int64_t bytes;
};

struct Endpoint {
  TileModuleOp tile;
  StorageLoadOp load;
};

struct SharingGroup {
  InputWindow window;
  llvm::SmallVector<Endpoint> endpoints;
};

// Follow only explicit aliases. An escaping value, unknown effect or any
// second writer excludes the choice; this is not an alias-repair pass.
bool hasOnlyReads(mlir::Value value, mlir::Operation *writer,
                  llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *op = use.getOwner();
    if (op == writer)
      continue;
    if (auto region = mlir::dyn_cast<TileRegionOp>(op)) {
      if (!hasOnlyReads(
              region.getBody().front().getArgument(use.getOperandNumber()),
              writer, visited))
        return false;
      continue;
    }
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
      if (view.getViewSource() != value)
        return false;
      for (mlir::Value result : op->getResults())
        if (!hasOnlyReads(result, writer, visited))
          return false;
      continue;
    }
    auto effects = mlir::getEffectsRecursively(op);
    if (!effects || op->hasTrait<mlir::OpTrait::IsTerminator>() ||
        op->getNumRegions())
      return false;
    for (mlir::Value result : op->getResults()) {
      if (!mlir::isa<mlir::ShapedType>(result.getType()))
        continue;
      // A materializing layout conversion or compute result has an explicit
      // allocation effect. It cannot alias this input. Unclassified results
      // remain excluded, and metadata aliases were followed above.
      if (!mlir::isa<mlir::MemRefType>(result.getType()) ||
          !llvm::any_of(*effects, [&](const auto &effect) {
            return effect.getValue() == result &&
                   mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect());
          }))
        return false;
    }
    bool reads = false;
    for (const auto &effect : *effects) {
      if (effect.getValue() != value)
        continue;
      if (!mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
        return false;
      reads = true;
    }
    if (!reads)
      return false;
  }
  return true;
}

mlir::BlockArgument getProgramSource(mlir::Value value) {
  while (true) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (auto region = mlir::dyn_cast<TileRegionOp>(
              argument.getOwner()->getParentOp())) {
        value = region.getInputs()[argument.getArgNumber()];
        continue;
      }
      return mlir::isa<mlir::func::FuncOp>(argument.getOwner()->getParentOp())
                 ? argument
                 : mlir::BlockArgument{};
    }
    if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      if (llvm::any_of(view.getStaticOffsets(), mlir::ShapedType::isDynamic) ||
          llvm::any_of(view.getStaticSizes(), mlir::ShapedType::isDynamic) ||
          llvm::any_of(view.getStaticStrides(), mlir::ShapedType::isDynamic))
        return {};
      value = view.getSource();
      continue;
    }
    return {};
  }
}

std::optional<InputWindow> getInputWindow(StorageLoadOp load) {
  auto region = mlir::dyn_cast<TileRegionOp>(load->getParentOp());
  if (!region || !mlir::isa<mlir::func::FuncOp>(region->getParentOp()))
    return std::nullopt;
  auto argument = getProgramSource(load.getSource());
  if (!argument)
    return std::nullopt;
  auto function =
      mlir::cast<mlir::func::FuncOp>(argument.getOwner()->getParentOp());
  auto identity = function.getArgAttrOfType<ProgramArgumentAttr>(
      argument.getArgNumber(), kWaferProgramArgumentAttrName);
  if (!identity || identity.getIndex() != argument.getArgNumber())
    return std::nullopt;
  auto original = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
  auto source = mlir::cast<mlir::MemRefType>(load.getSource().getType());
  auto payload = mlir::cast<mlir::MemRefType>(load.getDest().getType());
  if (!original || !original.getLayout().isIdentity() ||
      !isWaferDDRMemRefType(original) ||
      getWaferMemoryAttr(original).getLayout() != MemLayout::Tensor ||
      !source.hasStaticShape() || !payload.hasStaticShape() ||
      !load.getDest().getDefiningOp<mlir::memref::AllocOp>())
    return std::nullopt;
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(mlir::getStridesAndOffset(source, strides, offset)) ||
      mlir::ShapedType::isDynamic(offset) || offset < 0 ||
      llvm::any_of(strides, [](int64_t stride) { return stride < 0; }))
    return std::nullopt;
  auto physical = computeWaferPhysicalTensorInfo(payload);
  if (!physical || physical->physicalBytes <= 0)
    return std::nullopt;
  llvm::DenseSet<mlir::Value> reads;
  if (!hasOnlyReads(load.getDest(), load, reads))
    return std::nullopt;
  return InputWindow{identity.getIndex(), source, payload,
                     std::move(strides),  offset, physical->physicalBytes};
}

llvm::SmallVector<SharingGroup> collectGroups(mlir::ModuleOp module) {
  llvm::SmallVector<TileModuleOp> tiles;
  for (auto tile : module.getOps<TileModuleOp>())
    tiles.push_back(tile);
  llvm::sort(tiles, [](TileModuleOp a, TileModuleOp b) {
    return std::make_pair(a.getCardId(), a.getTileId()) <
           std::make_pair(b.getCardId(), b.getTileId());
  });
  // A program argument denotes the same external resource across these
  // isolated Tile entries. A writer in any binding invalidates read-only
  // sharing, including Tiles that have no otherwise eligible load.
  std::map<std::pair<int64_t, int64_t>, bool> readOnlyInputs;
  bool completeBindings = true;
  for (auto tile : tiles)
    tile.walk([&](mlir::func::FuncOp function) {
      if (function.isExternal())
        return;
      for (auto argument : function.getArguments()) {
        auto identity = function.getArgAttrOfType<ProgramArgumentAttr>(
            argument.getArgNumber(), kWaferProgramArgumentAttrName);
        if (!identity) {
          if (mlir::isa<mlir::BaseMemRefType>(argument.getType()) &&
              !function.getArgAttrOfType<DDRBindingAttr>(
                  argument.getArgNumber(), kWaferDDRBindingAttrName))
            completeBindings = false;
          continue;
        }
        auto [entry, inserted] = readOnlyInputs.try_emplace(
            std::make_pair(tile.getCardId(), identity.getIndex()), true);
        llvm::DenseSet<mlir::Value> visited;
        entry->second =
            entry->second && hasOnlyReads(argument, nullptr, visited);
      }
    });
  if (!completeBindings)
    return {};
  llvm::SmallVector<SharingGroup> groups;
  for (auto tile : tiles)
    tile.walk([&](StorageLoadOp load) {
      auto window = getInputWindow(load);
      if (!window || !readOnlyInputs.at({tile.getCardId(), window->argument}))
        return;
      auto group = llvm::find_if(groups, [&](const SharingGroup &group) {
        auto firstTile = group.endpoints.front().tile;
        return group.window.argument == window->argument &&
               group.window.sourceType == window->sourceType &&
               group.window.payloadType == window->payloadType &&
               group.window.strides == window->strides &&
               group.window.offset == window->offset &&
               firstTile.getCardId() == tile.getCardId() &&
               !llvm::any_of(group.endpoints, [&](const Endpoint &endpoint) {
                 return endpoint.tile == tile;
               });
      });
      if (group == groups.end()) {
        groups.push_back({std::move(*window), {}});
        group = std::prev(groups.end());
      }
      group->endpoints.push_back({tile, load});
    });
  llvm::erase_if(groups, [](const SharingGroup &group) {
    return group.endpoints.size() < 2;
  });
  return groups;
}

} // namespace

bool hasReadOnlyInputSharing(mlir::ModuleOp module) {
  return !collectGroups(module).empty();
}

BoundaryMovementResult
materializeReadOnlyInputSharing(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations) {
  BoundaryMovementResult result;
  if (mlir::failed(verifyTileModuleCollection(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations))) {
    result.failure = BoundaryMovementFailureKind::BrokenContract;
    result.detail = "input sharing requires verified current buffer owners";
    return result;
  }
  auto groups = collectGroups(module);
  if (groups.empty()) {
    result.failure = BoundaryMovementFailureKind::Unsupported;
    result.detail = "no proven equal read-only program windows";
    return result;
  }
  int64_t next = 0;
  module.walk([&](mlir::Operation *operation) {
    for (auto attribute : operation->getAttrs())
      attribute.getValue().walk([&](DTEMessageAttr message) {
        next = std::max(next, message.getCommunicationId());
      });
  });
  if (groups.size() >= uint64_t(std::numeric_limits<int64_t>::max() - next)) {
    result.failure = BoundaryMovementFailureKind::Unsupported;
    result.detail = "input sharing exhausts message identities";
    return result;
  }
  mlir::IRRewriter rewriter(module.getContext());
  for (auto &group : groups) {
    auto donor = group.endpoints.front();
    ++next;
    mlir::Operation *lastSend = donor.load;
    for (auto [index, endpoint] :
         llvm::enumerate(llvm::ArrayRef(group.endpoints).drop_front())) {
      Endpoint receiver = endpoint;
      auto message = DTEMessageAttr::get(module.getContext(), next, 0, index);
      rewriter.setInsertionPointAfter(lastSend);
      lastSend = rewriter.create<CommPeerSendOp>(
          donor.load.getLoc(), donor.load.getDest(), receiver.tile.getTileId(),
          group.window.bytes, message);
      rewriter.setInsertionPoint(receiver.load);
      rewriter.create<CommPeerRecvOp>(
          receiver.load.getLoc(), receiver.load.getDest(),
          donor.tile.getTileId(), group.window.bytes, message);
      rewriter.eraseOp(receiver.load);
      ++result.statistics.peerSends;
      ++result.statistics.peerReceives;
    }
  }
  rebuildCurrentBufferOwnerRelations(module, relations);
  if (mlir::failed(verifyPhysicalTileDataflow(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations))) {
    result.failure = BoundaryMovementFailureKind::CompilerFailure;
    result.detail = "input sharing produced invalid physical dataflow";
  }
  return result;
}

} // namespace wafer::compiler::detail
