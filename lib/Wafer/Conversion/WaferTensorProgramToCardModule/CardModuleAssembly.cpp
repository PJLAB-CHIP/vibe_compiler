//===- CardModuleAssembly.cpp - Final CardModule assembly -----------===//

#include "Internal.h"

#include "Wafer/Support/BoundedParallel.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

namespace wafer::tensor_program_to_card_module {

mlir::LogicalResult lowerPreparedTensorProgramToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    const TileMaterializationPreparation &preparation,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule, std::string *failureReason,
    StructuredMaterializationRelations *materializationRelations,
    CardModuleMaterializationStatistics *statistics) {
  mlir::OwningOpRef<mlir::ModuleOp> result =
      mlir::ModuleOp::create(sourceModule.getLoc());
  result->getOperation()->setAttrs(sourceModule->getAttrDictionary());
  cloneModuleFacts(sourceModule, *result);

  mlir::OpBuilder moduleBuilder(result->getBodyRegion());
  auto card = moduleBuilder.create<CardModuleOp>(
      sourceModule.getLoc(),
      moduleBuilder.getI64IntegerAttr(cardId.getValue()));
  card.getBody().push_back(new mlir::Block());
  mlir::Block &cardBody = card.getBody().front();
  mlir::OpBuilder cardBuilder(&cardBody, cardBody.end());
  mlir::IRMapping declarationMapping;
  for (mlir::Operation &operation :
       sourceModule.getBody()->without_terminator()) {
    if (isCardSharedDeclaration(operation) &&
        !mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation))
      cardBuilder.clone(operation, declarationMapping);
  }

  struct MaterializedTileEntry {
    mlir::OwningOpRef<mlir::ModuleOp> owner;
    mlir::func::FuncOp entry;
    StructuredMaterializationRelations relations;
    llvm::SmallVector<int64_t, 4> coveredShardExtents;
    std::string failureReason;
    bool succeeded = false;
  };
  std::vector<MaterializedTileEntry> tileEntries(
      preparation.availableTiles.size());
  unsigned materializationWorkers = 1;
  {
    mlir::ParallelDiagnosticHandler parallelDiagnostics(
        sourceModule.getContext());
    materializationWorkers = wafer::support::runBoundedParallelWork(
        sourceModule.getContext(), preparation.availableTiles.size(),
        [&](size_t tileIndex) {
          parallelDiagnostics.setOrderIDForThread(tileIndex);
          auto eraseDiagnosticOrder = llvm::make_scope_exit(
              [&] { parallelDiagnostics.eraseOrderIDForThread(); });
          const TileId tileId = preparation.availableTiles[tileIndex];
          MaterializedTileEntry &result = tileEntries[tileIndex];
          result.coveredShardExtents.assign(preparation.outputDomains.size(),
                                            0);
          llvm::SmallVector<SpatialOutputShard, 4> tileShards;
          collectTileOutputShards(preparation, tileId, tileShards,
                                  result.coveredShardExtents);
          mlir::FailureOr<mlir::func::FuncOp> entry;
          if (mapping.materializationMode ==
              SpatialDataflowMaterializationMode::IndependentDDRStages)
            entry = lowerBaselineTileEntry(preparation, cardId, tileId, mapping,
                                           tileShards, result.relations,
                                           &result.failureReason);
          else
            entry =
                lowerTileEntry(preparation, cardId, tileId, mapping, tileShards,
                               /*materializeTile=*/true, &result.failureReason,
                               &result.relations);
          if (mlir::failed(entry))
            return;
          result.owner = mlir::ModuleOp::create(sourceModule.getLoc());
          result.owner->getBody()->push_back(entry->getOperation());
          result.entry = *entry;
          result.succeeded = true;
        });
  }

  llvm::SmallVector<int64_t, 4> coveredShardExtents(
      preparation.outputDomains.size(), 0);
  StructuredMaterializationRelations resultRelations;
  for (auto [tileIndex, materialized] : llvm::enumerate(tileEntries)) {
    const TileId tileId = preparation.availableTiles[tileIndex];
    if (!materialized.succeeded) {
      if (failureReason)
        *failureReason = materialized.failureReason;
      return mlir::failure();
    }
    for (auto [outputIndex, covered] :
         llvm::enumerate(materialized.coveredShardExtents))
      coveredShardExtents[outputIndex] += covered;
    auto tile = cardBuilder.create<TileModuleOp>(
        sourceModule.getLoc(),
        cardBuilder.getI64IntegerAttr(tileId.getValue()));
    tile.getBody().push_back(new mlir::Block());
    mlir::Block &tileBody = tile.getBody().front();

    materialized.entry->remove();
    tileBody.push_back(materialized.entry.getOperation());
    // Peer endpoint verification resolves physical identities through the
    // enclosing module topology, so transform and verify the entry only after
    // it has been attached to its final Card/Tile IR scope.
    if (mlir::failed(removeTileOutputDestinations(
            materialized.entry, preparation.sourceArgumentCount,
            materialized.relations, failureReason)))
      return mlir::failure();
    resultRelations.operationResultBuffers.append(
        materialized.relations.operationResultBuffers.begin(),
        materialized.relations.operationResultBuffers.end());
    resultRelations.operandBuffers.append(
        materialized.relations.operandBuffers.begin(),
        materialized.relations.operandBuffers.end());
    resultRelations.scratchBuffers.append(
        materialized.relations.scratchBuffers.begin(),
        materialized.relations.scratchBuffers.end());
    resultRelations.outputBuffers.append(
        materialized.relations.outputBuffers.begin(),
        materialized.relations.outputBuffers.end());
  }

  if (!relationsBelongTo(result->getOperation(), resultRelations))
    return failCardModule(
        failureReason,
        "CardModule materialization produced a buffer relation outside the "
        "current IR");

  for (auto [outputIndex, covered] : llvm::enumerate(coveredShardExtents)) {
    int64_t expected = 1;
    for (int64_t extent : preparation.outputDomains[outputIndex])
      expected *= extent;
    if (covered != expected)
      return failCardModule(failureReason,
                            "card spatial shards do not cover output "
                            "domain exactly");
  }
  {
    mlir::ScopedDiagnosticHandler suppress(
        sourceModule.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    if (mlir::failed(mlir::verify(*result)))
      return failCardModule(failureReason,
                            "materialized CardModule is not verifier-legal");
  }

  if (materializationRelations)
    *materializationRelations = std::move(resultRelations);
  if (statistics) {
    statistics->tileEntryMaterializations = tileEntries.size();
    statistics->maximumTileMaterializationWorkers = materializationWorkers;
  }
  cardModule = std::move(result);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_card_module
