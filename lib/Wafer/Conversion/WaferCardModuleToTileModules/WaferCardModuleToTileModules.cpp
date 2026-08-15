//===- WaferCardModuleToTileModules.cpp - Per-Tile modules -------------===//

#include "Wafer/Conversion/WaferCardModuleToTileModules/WaferCardModuleToTileModules.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace wafer {
namespace {

using TileModuleList = llvm::SmallVector<TileModule, 16>;

static mlir::FailureOr<TileModuleList> failSplit(std::string *failureReason,
                                                 llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

static bool isCardSharedDeclaration(mlir::Operation &operation) {
  if (!mlir::isa<mlir::SymbolOpInterface>(&operation))
    return false;
  return llvm::all_of(operation.getRegions(),
                      [](mlir::Region &region) { return region.empty(); });
}

static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
createTileModule(mlir::ModuleOp sourceModule, CardModuleOp cardModule,
                 TileModuleOp sourceTileModule,
                 const StructuredMaterializationRelations *sourceRelations,
                 StructuredMaterializationRelations &tileRelations,
                 std::string *failureReason) {
  mlir::OwningOpRef<mlir::ModuleOp> resultModule =
      mlir::ModuleOp::create(sourceModule.getLoc());
  resultModule->getOperation()->setAttrs(sourceModule->getAttrDictionary());

  mlir::OpBuilder builder(resultModule->getBodyRegion());
  mlir::IRMapping mapping;

  for (TargetTopologyOp topology : sourceModule.getOps<TargetTopologyOp>())
    builder.clone(*topology.getOperation(), mapping);
  for (ExecutionMeshOp mesh : sourceModule.getOps<ExecutionMeshOp>())
    builder.clone(*mesh.getOperation(), mapping);

  for (mlir::Operation &operation : cardModule.getBody().front()) {
    if (mlir::isa<TileModuleOp>(operation))
      continue;
    if (!isCardSharedDeclaration(operation)) {
      if (failureReason)
        *failureReason =
            "wafer.card.module contains a non-declaration operation outside "
            "a wafer.tile.module";
      return mlir::failure();
    }
    builder.clone(operation, mapping);
  }

  for (mlir::Operation &operation : sourceTileModule.getBody().front())
    builder.clone(operation, mapping);

  if (sourceRelations) {
    auto remap = [&](const auto &source, auto &destination) {
      for (const auto &relation : source) {
        mlir::Value buffer = relation.buffer;
        mlir::Operation *parent =
            buffer ? buffer.getParentRegion()->getParentOp() : nullptr;
        const bool belongsToTile =
            parent && (parent == sourceTileModule.getOperation() ||
                       sourceTileModule->isProperAncestor(parent));
        if (mlir::Value mapped = mapping.lookupOrNull(buffer)) {
          auto copy = relation;
          copy.buffer = mapped;
          destination.push_back(copy);
          continue;
        }
        if (belongsToTile && failureReason)
          *failureReason = "Tile module failed to remap a current-IR buffer "
                           "relation";
      }
    };
    remap(sourceRelations->operationResultBuffers,
          tileRelations.operationResultBuffers);
    remap(sourceRelations->operandBuffers, tileRelations.operandBuffers);
    remap(sourceRelations->outputBuffers, tileRelations.outputBuffers);
    if (failureReason && !failureReason->empty())
      return mlir::failure();
  }

  if (mlir::failed(mlir::verify(*resultModule))) {
    if (failureReason)
      *failureReason = "Tile module is not verifier-legal";
    return mlir::failure();
  }
  return std::move(resultModule);
}

} // namespace

mlir::FailureOr<llvm::SmallVector<TileModule, 16>>
splitCardModuleIntoTileModules(
    mlir::ModuleOp sourceModule, std::string *failureReason,
    const StructuredMaterializationRelations *materializationRelations) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule)
    return failSplit(failureReason, "source module is null");
  if (mlir::failed(mlir::verify(sourceModule)))
    return failSplit(failureReason, "source module is not verifier-legal");

  llvm::SmallVector<CardModuleOp, 2> cardModules(
      sourceModule.getOps<CardModuleOp>());
  if (cardModules.size() != 1)
    return failSplit(
        failureReason,
        "expected exactly one direct wafer.card.module in source module");
  CardModuleOp cardModule = cardModules.front();

  llvm::SmallVector<TileModuleOp, 16> sourceTileModules(
      cardModule.getBody().front().getOps<TileModuleOp>());
  llvm::sort(sourceTileModules, [](TileModuleOp lhs, TileModuleOp rhs) {
    return lhs.getTileIdAttr().getInt() < rhs.getTileIdAttr().getInt();
  });

  TileModuleList resultModules;
  resultModules.reserve(sourceTileModules.size());
  for (TileModuleOp sourceTileModule : sourceTileModules) {
    StructuredMaterializationRelations tileRelations;
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> resultModule =
        createTileModule(sourceModule, cardModule, sourceTileModule,
                         materializationRelations, tileRelations,
                         failureReason);
    if (mlir::failed(resultModule))
      return mlir::failure();
    resultModules.push_back(
        TileModule{CardId(cardModule.getCardIdAttr().getInt()),
                   TileId(sourceTileModule.getTileIdAttr().getInt()),
                   std::move(*resultModule), std::move(tileRelations)});
  }
  return std::move(resultModules);
}

} // namespace wafer
