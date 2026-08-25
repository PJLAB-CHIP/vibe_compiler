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
    builder.clone(operation, mapping);
  }

  if (sourceRelations) {
    for (const StructuredOperationEmissionRelation &relation :
         sourceRelations->operationEmissions) {
      mlir::Operation *operation = relation.operation;
      if (operation && (operation == sourceTileModule.getOperation() ||
                        sourceTileModule->isProperAncestor(operation)))
        tileRelations.operationEmissions.push_back(relation);
    }
    auto retain = [&](const auto &source, auto &destination) {
      for (const auto &relation : source) {
        mlir::Value buffer = relation.buffer;
        mlir::Operation *parent =
            buffer ? buffer.getParentRegion()->getParentOp() : nullptr;
        const bool belongsToTile =
            parent && (parent == sourceTileModule.getOperation() ||
                       sourceTileModule->isProperAncestor(parent));
        if (belongsToTile)
          destination.push_back(relation);
      }
    };
    retain(sourceRelations->operationResultBuffers,
           tileRelations.operationResultBuffers);
    retain(sourceRelations->operandBuffers, tileRelations.operandBuffers);
    retain(sourceRelations->scratchBuffers, tileRelations.scratchBuffers);
    retain(sourceRelations->outputBuffers, tileRelations.outputBuffers);
    retain(sourceRelations->cardDDRBuffers, tileRelations.cardDDRBuffers);
    retain(sourceRelations->partialReductionContributions,
           tileRelations.partialReductionContributions);
    retain(sourceRelations->partialReductionMergeInputs,
           tileRelations.partialReductionMergeInputs);
  }

  // TileModuleOp is IsolatedFromAbove, so its body owns a closed SSA graph.
  // Move that graph into the standalone module instead of cloning it. The
  // current-IR relation values above therefore remain the same SSA values.
  mlir::Block &sourceBody = sourceTileModule.getBody().front();
  mlir::Block *destinationBody = resultModule->getBody();
  while (!sourceBody.empty())
    sourceBody.front().moveBefore(destinationBody, destinationBody->end());

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
    mlir::OwningOpRef<mlir::ModuleOp> sourceModuleOwner,
    std::string *failureReason,
    const StructuredMaterializationRelations *materializationRelations) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModuleOwner)
    return failSplit(failureReason, "source module is null");
  mlir::ModuleOp sourceModule = *sourceModuleOwner;
  if (mlir::failed(mlir::verify(sourceModule)))
    return failSplit(failureReason, "source module is not verifier-legal");

  llvm::SmallVector<CardModuleOp, 2> cardModules(
      sourceModule.getOps<CardModuleOp>());
  if (cardModules.size() != 1)
    return failSplit(
        failureReason,
        "expected exactly one direct wafer.card.module in source module");
  CardModuleOp cardModule = cardModules.front();

  for (mlir::Operation &operation : cardModule.getBody().front()) {
    if (mlir::isa<TileModuleOp>(operation) ||
        isCardSharedDeclaration(operation))
      continue;
    return failSplit(
        failureReason,
        "wafer.card.module contains a non-declaration operation outside a "
        "wafer.tile.module");
  }

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
