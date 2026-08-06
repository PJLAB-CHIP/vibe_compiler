//===- RankTileMaterialization.cpp - Complete-rank Tile IR -------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

namespace wafer::tensor_program_scheduling {

namespace {

static mlir::Operation *getTopLevelOperation(mlir::Operation *operation,
                                             mlir::Block *entry) {
  if (!operation || !entry)
    return nullptr;
  mlir::Operation *current = operation;
  while (current && current->getBlock() != entry)
    current = current->getParentOp();
  return current;
}

static bool isAtOrBefore(mlir::Operation *lhs, mlir::Operation *rhs) {
  return lhs == rhs || lhs->isBeforeInBlock(rhs);
}

static bool isShapedDDR(mlir::Type type) {
  return !mlir::isa<mlir::ShapedType>(type) ||
         wafer::isWaferDDRMemRefType(type);
}

static mlir::FailureOr<llvm::SmallVector<mlir::OpOperand *, 8>>
analyzeSelectiveTileSpill(TileRegionOp region, mlir::Value root,
                          mlir::Operation *storeAfter,
                          mlir::Operation *reloadBefore) {
  if (!region || region.getBody().empty() || !root || !storeAfter ||
      !reloadBefore)
    return mlir::failure();
  mlir::Operation *rootDefinition = root.getDefiningOp();
  auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
  mlir::Block *block = rootDefinition ? rootDefinition->getBlock() : nullptr;
  if (!rootType || !rootType.hasStaticShape() ||
      !wafer::isWaferSPMMemRefType(rootType) || !rootDefinition || !block ||
      storeAfter->getBlock() != block || reloadBefore->getBlock() != block ||
      rootDefinition->getParentOfType<TileRegionOp>() != region ||
      storeAfter->getParentOfType<TileRegionOp>() != region ||
      reloadBefore->getParentOfType<TileRegionOp>() != region ||
      storeAfter == block->getTerminator() ||
      reloadBefore == block->getTerminator() ||
      !isAtOrBefore(rootDefinition, storeAfter) ||
      !storeAfter->isBeforeInBlock(reloadBefore))
    return mlir::failure();

  bool initialized = !mlir::isa<mlir::memref::AllocOp>(rootDefinition);
  llvm::SmallVector<mlir::OpOperand *, 8> lateUses;
  for (mlir::OpOperand &use : root.getUses()) {
    mlir::Operation *top = getTopLevelOperation(use.getOwner(), block);
    if (!top)
      return mlir::failure();
    if (isAtOrBefore(top, storeAfter)) {
      if (auto effects =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(use.getOwner()))
        initialized |= static_cast<bool>(
            effects.getEffectOnValue<mlir::MemoryEffects::Write>(root));
      continue;
    }
    if (isAtOrBefore(reloadBefore, top)) {
      lateUses.push_back(&use);
      continue;
    }
    // A use in the selected dead interval would require an additional root or
    // a different action; do not silently move it across the spill.
    return mlir::failure();
  }
  if (!initialized || lateUses.empty())
    return mlir::failure();
  return lateUses;
}

} // namespace

bool canMaterializeSelectiveTileSpill(TileRegionOp region, mlir::Value root,
                                      mlir::Operation *storeAfter,
                                      mlir::Operation *reloadBefore) {
  return mlir::succeeded(analyzeSelectiveTileSpill(
      region, root, storeAfter, reloadBefore));
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankTileProgram(
    mlir::ModuleOp sourceModule, int64_t logicalRank,
    unsigned *materializedRegionCount) {
  if (materializedRegionCount)
    *materializedRegionCount = 0;
  if (!sourceModule || logicalRank < 0) {
    if (sourceModule)
      sourceModule.emitError(
          "complete-rank Tile materialization requires a non-negative "
          "logical rank");
    return mlir::failure();
  }

  mlir::OwningOpRef<mlir::ModuleOp> rankModule;
  std::string failureReason;
  if (mlir::failed(lowerCompleteRankTensorProgramToTileRegionModule(
          sourceModule, rankModule, &failureReason, logicalRank))) {
    sourceModule.emitError()
        << "cannot materialize complete-rank structured IR as Tile IR"
        << (failureReason.empty() ? "" : ": ") << failureReason;
    return mlir::failure();
  }

  unsigned regionCount = 0;
  (*rankModule)->walk([&](wafer::TileRegionOp) { ++regionCount; });
  if (regionCount != 1) {
    (*rankModule).emitError()
        << "conservative complete-rank Tile materialization requires one "
           "outer residency region, got "
        << regionCount;
    return mlir::failure();
  }
  if (materializedRegionCount)
    *materializedRegionCount = regionCount;
  return std::move(rankModule);
}

mlir::FailureOr<SelectiveSpillMaterialization>
materializeSelectiveTileSpill(TileRegionOp region, mlir::Value root,
                              mlir::Operation *storeAfter,
                              mlir::Operation *reloadBefore) {
  mlir::FailureOr<llvm::SmallVector<mlir::OpOperand *, 8>> lateUses =
      analyzeSelectiveTileSpill(region, root, storeAfter, reloadBefore);
  if (mlir::failed(lateUses))
    return mlir::failure();
  auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());

  auto ddrType = mlir::MemRefType::get(
      rootType.getShape(), rootType.getElementType(),
      mlir::MemRefLayoutAttrInterface{},
      wafer::MemoryAttr::get(root.getContext(), wafer::MemorySpace::DDR,
                             wafer::MemLayout::Tensor));
  mlir::OpBuilder storeBuilder(root.getContext());
  storeBuilder.setInsertionPointAfter(storeAfter);
  auto ddrAllocation =
      storeBuilder.create<mlir::memref::AllocOp>(root.getLoc(), ddrType);
  auto store = storeBuilder.create<wafer::StorageStoreOp>(
      root.getLoc(), root, ddrAllocation.getResult());

  mlir::OpBuilder reloadBuilder(reloadBefore);
  auto reloadAllocation =
      reloadBuilder.create<mlir::memref::AllocOp>(root.getLoc(), rootType);
  auto load = reloadBuilder.create<wafer::StorageLoadOp>(
      root.getLoc(), ddrAllocation.getResult(), reloadAllocation.getResult());
  for (mlir::OpOperand *use : *lateUses)
    use->set(reloadAllocation.getResult());

  return SelectiveSpillMaterialization{
      ddrAllocation.getResult(), reloadAllocation.getResult(), store, load,
      reloadAllocation.getOperation()};
}

mlir::FailureOr<TileRegionPartition>
partitionTileRegionAtDDRBoundary(TileRegionOp region,
                                 mlir::Operation *tailBegin) {
  if (!region || region.getBody().empty() || !tailBegin)
    return mlir::failure();
  mlir::Block &entry = region.getBody().front();
  if (tailBegin->getBlock() != &entry ||
      tailBegin == entry.getTerminator())
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *, 16> headOperations;
  llvm::SmallVector<mlir::Operation *, 16> tailOperations;
  bool inTail = false;
  for (mlir::Operation &operation : entry.without_terminator()) {
    inTail |= &operation == tailBegin;
    (inTail ? tailOperations : headOperations).push_back(&operation);
  }
  if (!inTail || headOperations.empty() || tailOperations.empty())
    return mlir::failure();

  llvm::DenseSet<mlir::Operation *> headSet(headOperations.begin(),
                                            headOperations.end());
  llvm::DenseSet<mlir::Operation *> tailSet(tailOperations.begin(),
                                            tailOperations.end());
  llvm::SmallVector<bool, 8> headInputUsed(entry.getNumArguments(), false);
  llvm::SmallVector<bool, 8> tailInputUsed(entry.getNumArguments(), false);
  llvm::SmallVector<mlir::Value, 8> crossingValues;
  llvm::DenseSet<mlir::Value> seenCrossings;

  auto classifyOperand = [&](mlir::Value value, bool tail) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() == &entry) {
        (tail ? tailInputUsed : headInputUsed)[argument.getArgNumber()] = true;
        return true;
      }
      mlir::Operation *owner = argument.getOwner()
                                   ? argument.getOwner()->getParentOp()
                                   : nullptr;
      mlir::Operation *top = getTopLevelOperation(owner, &entry);
      return top && (tail ? tailSet.contains(top) : headSet.contains(top));
    }
    mlir::Operation *definition = value.getDefiningOp();
    mlir::Operation *top = getTopLevelOperation(definition, &entry);
    if (!top)
      return false;
    if (tailSet.contains(top))
      return tail;
    if (!headSet.contains(top))
      return false;
    if (!tail)
      return true;
    if (!isShapedDDR(value.getType()))
      return false;
    if (seenCrossings.insert(value).second)
      crossingValues.push_back(value);
    return true;
  };

  auto scanOperation = [&](mlir::Operation *top, bool tail) {
    bool valid = true;
    top->walk([&](mlir::Operation *nested) {
      for (mlir::Value operand : nested->getOperands())
        valid &= classifyOperand(operand, tail);
    });
    return valid;
  };
  for (mlir::Operation *operation : headOperations)
    if (!scanOperation(operation, /*tail=*/false))
      return mlir::failure();
  for (mlir::Operation *operation : tailOperations)
    if (!scanOperation(operation, /*tail=*/true))
      return mlir::failure();
  auto originalYield =
      mlir::dyn_cast<wafer::TileYieldOp>(entry.getTerminator());
  if (!originalYield)
    return mlir::failure();
  for (mlir::Value value : originalYield.getValues())
    if (!classifyOperand(value, /*tail=*/true))
      return mlir::failure();

  llvm::SmallVector<unsigned, 8> headInputIndices;
  llvm::SmallVector<unsigned, 8> tailInputIndices;
  for (unsigned index = 0; index < entry.getNumArguments(); ++index) {
    if (headInputUsed[index])
      headInputIndices.push_back(index);
    if (tailInputUsed[index])
      tailInputIndices.push_back(index);
  }

  llvm::SmallVector<mlir::Value, 8> headInputs;
  for (unsigned index : headInputIndices)
    headInputs.push_back(region.getInputs()[index]);
  llvm::SmallVector<mlir::Type, 8> headResultTypes;
  for (mlir::Value crossing : crossingValues)
    headResultTypes.push_back(crossing.getType());

  mlir::OpBuilder outerBuilder(region);
  auto head = outerBuilder.create<wafer::TileRegionOp>(
      region.getLoc(), headResultTypes, headInputs);
  mlir::Block *headBlock = new mlir::Block();
  head.getBody().push_back(headBlock);
  mlir::IRMapping headMapping;
  for (auto [newIndex, oldIndex] : llvm::enumerate(headInputIndices)) {
    mlir::Value input = headInputs[newIndex];
    mlir::BlockArgument argument =
        headBlock->addArgument(input.getType(), input.getLoc());
    headMapping.map(entry.getArgument(oldIndex), argument);
  }
  mlir::OpBuilder headBuilder = mlir::OpBuilder::atBlockEnd(headBlock);
  for (mlir::Operation *operation : headOperations)
    headBuilder.clone(*operation, headMapping);
  llvm::SmallVector<mlir::Value, 8> mappedCrossings;
  for (mlir::Value crossing : crossingValues)
    mappedCrossings.push_back(headMapping.lookup(crossing));
  headBuilder.create<wafer::TileYieldOp>(region.getLoc(), mappedCrossings);

  llvm::SmallVector<mlir::Value, 8> tailInputs;
  for (unsigned index : tailInputIndices)
    tailInputs.push_back(region.getInputs()[index]);
  tailInputs.append(head.getResults().begin(), head.getResults().end());
  llvm::SmallVector<mlir::Type, 8> tailResultTypes(region.getResultTypes());
  outerBuilder.setInsertionPointAfter(head);
  auto tail = outerBuilder.create<wafer::TileRegionOp>(
      region.getLoc(), tailResultTypes, tailInputs);
  mlir::Block *tailBlock = new mlir::Block();
  tail.getBody().push_back(tailBlock);
  mlir::IRMapping tailMapping;
  unsigned tailArgumentIndex = 0;
  for (unsigned oldIndex : tailInputIndices) {
    mlir::Value input = tailInputs[tailArgumentIndex++];
    mlir::BlockArgument argument =
        tailBlock->addArgument(input.getType(), input.getLoc());
    tailMapping.map(entry.getArgument(oldIndex), argument);
  }
  for (auto [crossing, headResult] :
       llvm::zip_equal(crossingValues, head.getResults())) {
    mlir::BlockArgument argument = tailBlock->addArgument(
        headResult.getType(), headResult.getLoc());
    tailMapping.map(crossing, argument);
    ++tailArgumentIndex;
  }
  mlir::OpBuilder tailBuilder = mlir::OpBuilder::atBlockEnd(tailBlock);
  for (mlir::Operation *operation : tailOperations)
    tailBuilder.clone(*operation, tailMapping);
  llvm::SmallVector<mlir::Value, 8> mappedResults;
  for (mlir::Value value : originalYield.getValues())
    mappedResults.push_back(tailMapping.lookup(value));
  tailBuilder.create<wafer::TileYieldOp>(region.getLoc(), mappedResults);

  if (mlir::failed(mlir::verify(head.getOperation())) ||
      mlir::failed(mlir::verify(tail.getOperation()))) {
    tail.erase();
    head.erase();
    return mlir::failure();
  }
  for (auto [oldResult, newResult] :
       llvm::zip_equal(region.getResults(), tail.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  region.erase();
  return TileRegionPartition{head, tail};
}

} // namespace wafer::tensor_program_scheduling
