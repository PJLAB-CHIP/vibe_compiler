//===- TileModuleOps.cpp - Wafer target module verification
//------------------===//

#include "Wafer/IR/WaferDialect.h"

#include "../Common/WaferIRVerification.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>

using namespace wafer;

namespace {

static mlir::LogicalResult verifyOnlyIdentifierAttributes(
    TileModuleOp op, llvm::ArrayRef<mlir::StringAttr> identifierNames) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!llvm::is_contained(identifierNames, attr.getName()) &&
        !wafer::detail::isExternalDiscardableAttribute(op, attr))
      return op.emitOpError("does not accept semantic attribute '")
             << attr.getName().getValue() << "'";
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult TileModuleOp::verify() {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(getOperation()->getParentOp());
  if (!module)
    return emitOpError("must be directly nested under a builtin.module");
  if (!getBody().hasOneBlock())
    return emitOpError("must contain exactly one body block");

  // Keep this verifier local. Complete topology and Tile-domain checks belong
  // to the builtin.module stage that consumes the Tile module collection.
  if (getCardIdAttr().getInt() < 0 || getTileIdAttr().getInt() < 0)
    return emitOpError("card_id and tile_id must be non-negative");

  if (mlir::failed(verifyOnlyIdentifierAttributes(
          *this, {getCardIdAttrName(), getTileIdAttrName()})))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult wafer::verifyTileModuleCollection(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();

  std::set<std::pair<int64_t, int64_t>> identities;
  for (TileModuleOp tile : module.getOps<TileModuleOp>()) {
    std::pair<int64_t, int64_t> identity{tile.getCardIdAttr().getInt(),
                                         tile.getTileIdAttr().getInt()};
    if (!identities.insert(identity).second)
      return tile.emitOpError("duplicates (card_id, tile_id) ")
             << '(' << identity.first << ", " << identity.second << ')';
  }

  std::map<int64_t, mlir::memref::GlobalOp> resources;
  for (mlir::memref::GlobalOp global :
       module.getOps<mlir::memref::GlobalOp>()) {
    auto resource =
        global->getAttrOfType<DDRResourceAttr>(kWaferDDRResourceAttrName);
    if (!resource)
      continue;
    mlir::MemRefType type = global.getType();
    if (resource.getResourceId() < 0 || !type || !type.hasStaticShape() ||
        !isWaferDDRMemRefType(type))
      return global.emitOpError("DDR resource must be static and DDR-typed");
    if (auto initial = global.getInitialValue()) {
      auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(*initial);
      if (!dense || !dense.isSplat() ||
          !dense.getSplatValue<llvm::APInt>().isZero())
        return global.emitOpError(
            "DDR initializer must be an integer zero splat");
    }
    if (!resources.try_emplace(resource.getResourceId(), global).second)
      return global.emitOpError("DDR resource_id must be unique: ")
             << resource.getResourceId();
  }

  for (TileModuleOp tile : module.getOps<TileModuleOp>()) {
    std::map<int64_t, size_t> bindings;
    mlir::LogicalResult valid = mlir::success();
    tile.walk([&](mlir::func::FuncOp function) {
      if (mlir::failed(valid))
        return;
      for (unsigned argument = 0; argument < function.getNumArguments();
           ++argument) {
        auto binding = function.getArgAttrOfType<DDRBindingAttr>(
            argument, kWaferDDRBindingAttrName);
        if (!binding)
          continue;
        auto resource = resources.find(binding.getResourceId());
        mlir::Operation *declaration =
            mlir::SymbolTable::lookupSymbolIn(module, binding.getResource());
        auto shapedType = mlir::dyn_cast<mlir::ShapedType>(
            function.getArgument(argument).getType());
        if (resource == resources.end() ||
            declaration != resource->second.getOperation() || !shapedType ||
            (!mlir::isa<mlir::RankedTensorType, mlir::MemRefType>(
                shapedType)) ||
            shapedType.getShape() != resource->second.getType().getShape() ||
            shapedType.getElementType() !=
                resource->second.getType().getElementType() ||
            ++bindings[binding.getResourceId()] != 1) {
          function.emitOpError(
              "has an invalid or duplicate shared DDR argument binding");
          valid = mlir::failure();
          return;
        }
      }
    });
    if (mlir::failed(valid))
      return mlir::failure();
  }
  return mlir::success();
}
