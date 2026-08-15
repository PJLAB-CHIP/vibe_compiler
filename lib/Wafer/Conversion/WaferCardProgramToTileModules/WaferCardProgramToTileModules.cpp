//===- WaferCardProgramToTileModules.cpp - Per-Tile modules -------------===//

#include "Wafer/Conversion/WaferCardProgramToTileModules/WaferCardProgramToTileModules.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace wafer {
namespace {

using PhysicalTileModuleList = llvm::SmallVector<PhysicalTileModule, 16>;

static mlir::FailureOr<PhysicalTileModuleList>
failSplit(std::string *failureReason, llvm::StringRef message) {
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
createPhysicalTileModule(
    mlir::ModuleOp sourceModule, CardProgramOp cardProgram,
    TileProgramOp tileProgram,
    const StructuredMaterializationRelations *sourceRelations,
    StructuredMaterializationRelations &tileRelations,
    std::string *failureReason) {
  mlir::OwningOpRef<mlir::ModuleOp> tileModule =
      mlir::ModuleOp::create(sourceModule.getLoc());
  tileModule->getOperation()->setAttrs(sourceModule->getAttrDictionary());

  mlir::OpBuilder builder(tileModule->getBodyRegion());
  mlir::IRMapping mapping;

  for (TargetTopologyOp topology : sourceModule.getOps<TargetTopologyOp>())
    builder.clone(*topology.getOperation(), mapping);
  for (ExecutionMeshOp mesh : sourceModule.getOps<ExecutionMeshOp>())
    builder.clone(*mesh.getOperation(), mapping);

  for (mlir::Operation &operation : cardProgram.getBody().front()) {
    if (mlir::isa<TileProgramOp>(operation))
      continue;
    if (!isCardSharedDeclaration(operation)) {
      if (failureReason)
        *failureReason =
            "wafer.card.program contains a non-declaration operation outside "
            "a wafer.tile.program";
      return mlir::failure();
    }
    builder.clone(operation, mapping);
  }

  for (mlir::Operation &operation : tileProgram.getBody().front())
    builder.clone(operation, mapping);

  if (sourceRelations) {
    auto remap = [&](const auto &source, auto &destination) {
      for (const auto &relation : source) {
        mlir::Value buffer = relation.buffer;
        mlir::Operation *parent =
            buffer ? buffer.getParentRegion()->getParentOp() : nullptr;
        const bool belongsToTile =
            parent && (parent == tileProgram.getOperation() ||
                       tileProgram->isProperAncestor(parent));
        if (mlir::Value mapped = mapping.lookupOrNull(buffer)) {
          auto copy = relation;
          copy.buffer = mapped;
          destination.push_back(copy);
          continue;
        }
        if (belongsToTile && failureReason)
          *failureReason =
              "physical Tile module failed to remap a current-IR buffer "
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

  if (mlir::failed(mlir::verify(*tileModule))) {
    if (failureReason)
      *failureReason = "physical Tile module is not verifier-legal";
    return mlir::failure();
  }
  return std::move(tileModule);
}

} // namespace

mlir::FailureOr<llvm::SmallVector<PhysicalTileModule, 16>>
splitCardProgramIntoPhysicalTileModules(
    mlir::ModuleOp sourceModule, std::string *failureReason,
    const StructuredMaterializationRelations *materializationRelations) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule)
    return failSplit(failureReason, "source module is null");
  if (mlir::failed(mlir::verify(sourceModule)))
    return failSplit(failureReason, "source module is not verifier-legal");

  llvm::SmallVector<CardProgramOp, 2> cardPrograms(
      sourceModule.getOps<CardProgramOp>());
  if (cardPrograms.size() != 1)
    return failSplit(
        failureReason,
        "expected exactly one direct wafer.card.program in source module");
  CardProgramOp cardProgram = cardPrograms.front();

  llvm::SmallVector<TileProgramOp, 16> tilePrograms(
      cardProgram.getBody().front().getOps<TileProgramOp>());
  llvm::sort(tilePrograms, [](TileProgramOp lhs, TileProgramOp rhs) {
    return lhs.getTileIdAttr().getInt() < rhs.getTileIdAttr().getInt();
  });

  PhysicalTileModuleList tileModules;
  tileModules.reserve(tilePrograms.size());
  for (TileProgramOp tileProgram : tilePrograms) {
    StructuredMaterializationRelations tileRelations;
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> tileModule =
        createPhysicalTileModule(sourceModule, cardProgram, tileProgram,
                                 materializationRelations, tileRelations,
                                 failureReason);
    if (mlir::failed(tileModule))
      return mlir::failure();
    tileModules.push_back(
        PhysicalTileModule{PhysicalCardId(cardProgram.getCardIdAttr().getInt()),
                           PhysicalTileId(tileProgram.getTileIdAttr().getInt()),
                           std::move(*tileModule), std::move(tileRelations)});
  }
  return std::move(tileModules);
}

} // namespace wafer
