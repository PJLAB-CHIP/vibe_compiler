//===- CardModuleAssembly.cpp - Final CardModule assembly -----------===//

#include "Internal.h"

#include "Wafer/Support/BoundedParallel.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"

#include <string>

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
  int64_t previousCardDDRResource = -1;
  for (const CardDDRResource &resource : mapping.cardDDRResources) {
    if (resource.resourceId < 0 ||
        resource.resourceId <= previousCardDDRResource ||
        !resource.tensorType || !resource.tensorType.hasStaticShape())
      return failCardModule(
          failureReason,
          "card DDR resources must have ordered IDs and static tensor types");
    previousCardDDRResource = resource.resourceId;
    auto memrefType = mlir::MemRefType::get(
        resource.tensorType.getShape(), resource.tensorType.getElementType(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(sourceModule.getContext(), MemorySpace::DDR,
                        MemLayout::Tensor));
    std::string symbol =
        (llvm::Twine("card_ddr_") + llvm::Twine(resource.resourceId)).str();
    auto declaration = cardBuilder.create<mlir::memref::GlobalOp>(
        sourceModule.getLoc(), symbol, cardBuilder.getStringAttr("private"),
        memrefType,
        /*initial_value=*/mlir::Attribute{}, /*constant=*/false,
        /*alignment=*/mlir::IntegerAttr{});
    declaration->setAttr(kWaferCardDDRResourceAttrName,
                         CardDDRResourceAttr::get(sourceModule.getContext(),
                                                  resource.resourceId));
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
          llvm::DenseSet<int64_t> existingCardDDRResources;
          for (unsigned argument = 0; argument < entry->getNumArguments();
               ++argument)
            if (auto binding = entry->getArgAttrOfType<CardDDRBindingAttr>(
                    argument, kWaferCardDDRBindingAttrName))
              existingCardDDRResources.insert(binding.getResourceId());
          for (const CardDDRResource &resource : mapping.cardDDRResources) {
            if (existingCardDDRResources.contains(resource.resourceId))
              continue;
            const unsigned argumentIndex = entry->getNumArguments();
            std::string symbol =
                (llvm::Twine("card_ddr_") + llvm::Twine(resource.resourceId))
                    .str();
            auto binding = CardDDRBindingAttr::get(
                entry->getContext(),
                mlir::FlatSymbolRefAttr::get(entry->getContext(), symbol),
                resource.resourceId, CardDDRAccess::None);
            entry->insertArgument(
                argumentIndex, resource.tensorType,
                mlir::DictionaryAttr::get(
                    entry->getContext(),
                    {mlir::NamedAttribute(
                        mlir::StringAttr::get(entry->getContext(),
                                              kWaferCardDDRBindingAttrName),
                        binding)}),
                entry->getLoc());
          }
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
    uint64_t cardDDRBoundaryCount = 0;
    for (unsigned argument = 0; argument < materialized.entry.getNumArguments();
         ++argument)
      cardDDRBoundaryCount += static_cast<bool>(
          materialized.entry.getArgAttrOfType<CardDDRBindingAttr>(
              argument, kWaferCardDDRBindingAttrName));
    if (mlir::failed(removeTileOutputDestinations(
            materialized.entry, preparation.sourceArgumentCount,
            materialized.relations, failureReason, cardDDRBoundaryCount)))
      return mlir::failure();
    resultRelations.operationEmissions.append(
        materialized.relations.operationEmissions.begin(),
        materialized.relations.operationEmissions.end());
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
    resultRelations.cardDDRBuffers.append(
        materialized.relations.cardDDRBuffers.begin(),
        materialized.relations.cardDDRBuffers.end());
    resultRelations.partialReductionContributions.append(
        materialized.relations.partialReductionContributions.begin(),
        materialized.relations.partialReductionContributions.end());
    resultRelations.partialReductionMergeInputs.append(
        materialized.relations.partialReductionMergeInputs.begin(),
        materialized.relations.partialReductionMergeInputs.end());
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
