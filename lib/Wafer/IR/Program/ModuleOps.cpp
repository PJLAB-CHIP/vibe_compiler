//===- ModuleOps.cpp - Wafer target module verification ------------------===//

#include "Wafer/IR/WaferDialect.h"

#include "../Common/WaferIRVerification.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <map>
using namespace wafer;

namespace {

template <typename ProgramOp>
static mlir::LogicalResult
verifyOnlyIdentifierAttribute(ProgramOp op, mlir::StringAttr identifierName) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (attr.getName() != identifierName &&
        !wafer::detail::isExternalDiscardableAttribute(op, attr))
      return op.emitOpError("does not accept semantic attribute '")
             << attr.getName().getValue() << "'";
  }
  return mlir::success();
}

static bool isCardSharedDeclaration(mlir::Operation &operation) {
  if (!mlir::isa<mlir::SymbolOpInterface>(operation))
    return false;
  return llvm::all_of(operation.getRegions(),
                      [](mlir::Region &region) { return region.empty(); });
}

} // namespace

mlir::LogicalResult CardModuleOp::verify() {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(getOperation()->getParentOp());
  if (!module)
    return emitOpError("must be directly nested under a builtin.module");
  if (!getBody().hasOneBlock())
    return emitOpError("must contain exactly one body block");

  int64_t cardId = getCardIdAttr().getInt();
  if (cardId < 0)
    return emitOpError("card_id must be non-negative");

  llvm::DenseSet<int64_t> seenTileIds;
  for (mlir::Operation &operation : getBody().front()) {
    auto tile = mlir::dyn_cast<TileModuleOp>(operation);
    if (!tile) {
      if (!isCardSharedDeclaration(operation))
        return emitOpError("body may contain only card-shared declarations and "
                           "wafer.tile.module operations");
      continue;
    }
    int64_t tileId = tile.getTileIdAttr().getInt();
    if (!seenTileIds.insert(tileId).second)
      return emitOpError("tile_id must be unique in a card module; duplicate ")
             << tileId;
  }

  std::map<int64_t, mlir::memref::GlobalOp> cardDDRResources;
  for (mlir::memref::GlobalOp global :
       getBody().front().getOps<mlir::memref::GlobalOp>()) {
    auto resource = global->getAttrOfType<CardDDRResourceAttr>(
        kWaferCardDDRResourceAttrName);
    if (!resource)
      continue;
    auto type = global.getType();
    if (resource.getResourceId() < 0 || !type || !type.hasStaticShape() ||
        !isWaferDDRMemRefType(type) || global.getInitialValue())
      return emitOpError(
          "card DDR declaration must be external, static, and DDR-typed");
    if (!cardDDRResources.try_emplace(resource.getResourceId(), global).second)
      return emitOpError("card DDR resource_id must be unique: ")
             << resource.getResourceId();
  }
  for (TileModuleOp tile : getBody().front().getOps<TileModuleOp>()) {
    std::map<int64_t, size_t> tileBindings;
    mlir::LogicalResult valid = mlir::success();
    tile.walk([&](mlir::func::FuncOp function) {
      if (mlir::failed(valid))
        return;
      for (unsigned argument = 0; argument < function.getNumArguments();
           ++argument) {
        auto binding = function.getArgAttrOfType<CardDDRBindingAttr>(
            argument, kWaferCardDDRBindingAttrName);
        if (!binding)
          continue;
        auto resource = cardDDRResources.find(binding.getResourceId());
        mlir::Operation *declaration =
            mlir::SymbolTable::lookupSymbolIn(*this, binding.getResource());
        auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(
            function.getArgument(argument).getType());
        if (resource == cardDDRResources.end() ||
            declaration != resource->second.getOperation() || !tensorType ||
            tensorType.getShape() != resource->second.getType().getShape() ||
            tensorType.getElementType() !=
                resource->second.getType().getElementType() ||
            ++tileBindings[binding.getResourceId()] != 1) {
          function.emitOpError(
              "has an invalid or duplicate card DDR argument binding");
          valid = mlir::failure();
          return;
        }
      }
    });
    if (mlir::failed(valid))
      return mlir::failure();
  }

  if (mlir::failed(verifyOnlyIdentifierAttribute(*this, getCardIdAttrName())))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult TileModuleOp::verify() {
  CardModuleOp card =
      mlir::dyn_cast<CardModuleOp>(getOperation()->getParentOp());
  if (!card)
    return emitOpError("must be directly nested under wafer.card.module");
  if (!getBody().hasOneBlock())
    return emitOpError("must contain exactly one body block");

  if (!mlir::isa<mlir::ModuleOp>(card->getParentOp()))
    return emitOpError(
        "requires its wafer.card.module to be directly nested under a "
        "builtin.module");

  // The parent CardModule verifier owns topology construction and checks the
  // complete child domain once. Keep this leaf verifier local so verifying a
  // card does not rebuild identical module-wide topology for every Tile.
  if (getTileIdAttr().getInt() < 0)
    return emitOpError("tile_id must be non-negative");

  if (mlir::failed(verifyOnlyIdentifierAttribute(*this, getTileIdAttrName())))
    return mlir::failure();
  return mlir::success();
}
