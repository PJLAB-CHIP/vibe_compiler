//===- StandaloneTileModules.cpp - Create standalone Tile modules --------===//

#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <tuple>
#include <utility>

namespace wafer {
namespace {

using StandaloneTileModuleList = llvm::SmallVector<StandaloneTileModule, 16>;

static mlir::FailureOr<StandaloneTileModuleList>
failCreate(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

static bool isSharedDeclaration(mlir::Operation &operation) {
  if (!mlir::isa<mlir::SymbolOpInterface>(&operation))
    return false;
  return llvm::all_of(operation.getRegions(),
                      [](mlir::Region &region) { return region.empty(); });
}

static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
createStandaloneTileModule(
    mlir::ModuleOp sourceModule, TileModuleOp sourceTileModule,
    const StructuredMaterializationRelations *sourceRelations,
    StructuredMaterializationRelations &tileRelations,
    std::string *failureReason) {
  mlir::OwningOpRef<mlir::ModuleOp> resultModule =
      mlir::ModuleOp::create(sourceModule.getLoc());
  resultModule->getOperation()->setAttrs(sourceModule->getAttrDictionary());

  mlir::OpBuilder builder(resultModule->getBodyRegion());
  mlir::IRMapping mapping;

  for (mlir::Operation &operation :
       sourceModule.getBody()->without_terminator()) {
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
    retain(sourceRelations->ddrBuffers, tileRelations.ddrBuffers);
    TileId tileId(sourceTileModule.getTileIdAttr().getInt());
    for (const DDRTransferRelation &relation : sourceRelations->ddrTransfers)
      if (relation.producerTile == tileId || relation.consumerTile == tileId)
        tileRelations.ddrTransfers.push_back(relation);
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

mlir::FailureOr<llvm::SmallVector<StandaloneTileModule, 16>>
createStandaloneTileModules(
    mlir::OwningOpRef<mlir::ModuleOp> sourceModuleOwner,
    std::string *failureReason,
    const StructuredMaterializationRelations *materializationRelations) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModuleOwner)
    return failCreate(failureReason, "source module is null");
  mlir::ModuleOp sourceModule = *sourceModuleOwner;
  if (mlir::failed(mlir::verify(sourceModule)))
    return failCreate(failureReason, "source module is not verifier-legal");
  if (mlir::failed(verifyTileModuleCollection(sourceModule)))
    return failCreate(failureReason, "source Tile module set is invalid");

  for (mlir::Operation &operation :
       sourceModule.getBody()->without_terminator()) {
    if (mlir::isa<TileModuleOp>(operation) || isSharedDeclaration(operation) ||
        mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation))
      continue;
    return failCreate(
        failureReason,
        "source module contains a non-declaration operation outside a "
        "wafer.tile.module");
  }

  llvm::SmallVector<TileModuleOp, 16> sourceTileModules(
      sourceModule.getOps<TileModuleOp>());
  if (sourceTileModules.empty())
    return failCreate(failureReason,
                      "source module contains no top-level wafer.tile.module");
  llvm::sort(sourceTileModules, [](TileModuleOp lhs, TileModuleOp rhs) {
    return std::tuple(lhs.getCardIdAttr().getInt(),
                      lhs.getTileIdAttr().getInt()) <
           std::tuple(rhs.getCardIdAttr().getInt(),
                      rhs.getTileIdAttr().getInt());
  });
  for (auto [lhs, rhs] :
       llvm::zip(sourceTileModules, llvm::drop_begin(sourceTileModules)))
    if (lhs.getCardIdAttr() == rhs.getCardIdAttr() &&
        lhs.getTileIdAttr() == rhs.getTileIdAttr())
      return failCreate(failureReason,
                        "source module repeats one (card_id, tile_id)");

  StandaloneTileModuleList resultModules;
  resultModules.reserve(sourceTileModules.size());
  for (TileModuleOp sourceTileModule : sourceTileModules) {
    StructuredMaterializationRelations tileRelations;
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> resultModule =
        createStandaloneTileModule(sourceModule, sourceTileModule,
                                   materializationRelations, tileRelations,
                                   failureReason);
    if (mlir::failed(resultModule))
      return mlir::failure();
    resultModules.push_back(StandaloneTileModule{
        CardId(sourceTileModule.getCardIdAttr().getInt()),
        TileId(sourceTileModule.getTileIdAttr().getInt()),
        std::move(*resultModule), std::move(tileRelations)});
  }
  return std::move(resultModules);
}

} // namespace wafer
