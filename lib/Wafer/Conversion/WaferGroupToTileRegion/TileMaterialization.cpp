//===- TileMaterialization.cpp - Candidate root tile materialization -===//

#include "Internal.h"

using namespace wafer;

namespace wafer::group_to_tile_region {

static std::optional<ComputeReduceKind>
inferCandidateReduceKind(mlir::linalg::GenericOp generic,
                         std::string *failureReason) {
  if (generic.getRegionInputArgs().size() != 1 ||
      generic.getRegionOutputArgs().size() != 1) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires one input and one accumulator");
    return std::nullopt;
  }
  return matchExactReductionKind(generic.getRegionOutputArgs(), /*redPos=*/0,
                                 generic.getRegionInputArgs().front(),
                                 "candidate reduction split", failureReason);
}

static mlir::FailureOr<ComputeReduceKind>
getCandidateCombineKind(mlir::linalg::LinalgOp root,
                        std::string *failureReason) {
  if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp>(
          root.getOperation()))
    return ComputeReduceKind::Sum;
  if (auto generic =
          mlir::dyn_cast<mlir::linalg::GenericOp>(root.getOperation())) {
    std::optional<ComputeReduceKind> kind =
        inferCandidateReduceKind(generic, failureReason);
    if (!kind)
      return mlir::failure();
    return *kind;
  }

  setFailureReason(
      failureReason,
      "candidate reduction split requires matmul, batch_matmul, or generic "
      "reduction root");
  return mlir::failure();
}

static mlir::FailureOr<mlir::TypedAttr>
getNeutralScalarAttr(mlir::Type elementType, ComputeReduceKind kind,
                     std::string *failureReason) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    switch (kind) {
    case ComputeReduceKind::Sum:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(floatType, 0.0));
    case ComputeReduceKind::Max:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(
          floatType, llvm::APFloat::getInf(floatType.getFloatSemantics(),
                                           /*Negative=*/true)));
    case ComputeReduceKind::Min:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(
          floatType, llvm::APFloat::getInf(floatType.getFloatSemantics(),
                                           /*Negative=*/false)));
    case ComputeReduceKind::Avg:
      break;
    }
  }

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    unsigned width = intType.getWidth();
    switch (kind) {
    case ComputeReduceKind::Sum:
      return mlir::cast<mlir::TypedAttr>(
          mlir::IntegerAttr::get(intType, llvm::APInt(width, 0)));
    case ComputeReduceKind::Max:
      return mlir::cast<mlir::TypedAttr>(mlir::IntegerAttr::get(
          intType, llvm::APInt::getSignedMinValue(width)));
    case ComputeReduceKind::Min:
      return mlir::cast<mlir::TypedAttr>(mlir::IntegerAttr::get(
          intType, llvm::APInt::getSignedMaxValue(width)));
    case ComputeReduceKind::Avg:
      break;
    }
  }

  setFailureReason(
      failureReason,
      "candidate reduction split requires float or integer accumulator type");
  return mlir::failure();
}

static mlir::FailureOr<mlir::Value>
createNeutralInitTensor(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::RankedTensorType resultType,
                        ComputeReduceKind kind, std::string *failureReason) {
  mlir::FailureOr<mlir::TypedAttr> attr =
      getNeutralScalarAttr(resultType.getElementType(), kind, failureReason);
  if (mlir::failed(attr))
    return mlir::failure();

  auto constant = builder.create<mlir::arith::ConstantOp>(loc, *attr);
  auto empty = builder.create<mlir::tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());
  auto fill = builder.create<mlir::linalg::FillOp>(loc, constant.getResult(),
                                                   empty.getResult());
  return fill.getResult(0);
}

static mlir::FailureOr<mlir::Value>
createPartialCombine(mlir::OpBuilder &builder, mlir::Location loc,
                     ComputeReduceKind kind, mlir::Value accumulator,
                     mlir::Value partial, mlir::Value outputInit,
                     std::string *failureReason) {
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(partial.getType());
  if (!resultType || accumulator.getType() != partial.getType() ||
      outputInit.getType() != partial.getType()) {
    setFailureReason(failureReason,
                     "candidate reduction split accumulator type mismatch");
    return mlir::failure();
  }

  mlir::MLIRContext *context = builder.getContext();
  if (!mlir::isa<mlir::FloatType, mlir::IntegerType>(
          resultType.getElementType())) {
    setFailureReason(failureReason,
                     "candidate reduction split requires float or integer "
                     "accumulator element type");
    return mlir::failure();
  }

  mlir::AffineMap identity =
      mlir::AffineMap::getMultiDimIdentityMap(resultType.getRank(), context);
  llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {identity, identity,
                                                        identity};
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      resultType.getRank(), mlir::utils::IteratorType::parallel);

  auto add = builder.create<mlir::linalg::GenericOp>(
      loc, resultType, mlir::ValueRange{accumulator, partial},
      mlir::ValueRange{outputInit}, indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange blockArgs) {
        mlir::Value value;
        mlir::Type elementType = resultType.getElementType();
        bool isFloat = mlir::isa<mlir::FloatType>(elementType);
        bool isInteger = mlir::isa<mlir::IntegerType>(elementType);
        if (!isFloat && !isInteger)
          return;

        switch (kind) {
        case ComputeReduceKind::Sum:
          if (isFloat) {
            value = nestedBuilder.create<mlir::arith::AddFOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          } else {
            value = nestedBuilder.create<mlir::arith::AddIOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          }
          break;
        case ComputeReduceKind::Max:
          if (isFloat) {
            value = nestedBuilder.create<mlir::arith::MaximumFOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          } else {
            value = nestedBuilder.create<mlir::arith::MaxSIOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          }
          break;
        case ComputeReduceKind::Min:
          if (isFloat) {
            value = nestedBuilder.create<mlir::arith::MinimumFOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          } else {
            value = nestedBuilder.create<mlir::arith::MinSIOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          }
          break;
        case ComputeReduceKind::Avg:
          return;
        }
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, value);
      });

  return add->getResult(0);
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    GroupOp group, mlir::linalg::LinalgOp root, unsigned outputIndex,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  if (root.getNumDpsInits() != 1 || root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires one DPS output");
    return mlir::failure();
  }
  if (!root.hasOnlyProjectedPermutations()) {
    setFailureReason(
        failureReason,
        "candidate tile materialization requires permutation-only maps");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(root);
  bool hasDirectOutputInit =
      isGroupOutputBoundary(group, root.getDpsInits().front(), outputIndex);
  if (!hasDirectOutputInit && reductionLoopDims.empty()) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires direct output "
                     "boundary init for non-reduction roots");
    return mlir::failure();
  }

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType) {
    setFailureReason(failureReason,
                     "candidate tile materialization result is not ranked");
    return mlir::failure();
  }
  if (mlir::failed(validateCandidateTile(resultType, candidateTileOffsets,
                                         candidateTileSizes, failureReason)))
    return mlir::failure();

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      root.getIndexingMapsArray();
  unsigned outputMapIndex = static_cast<unsigned>(root.getNumDpsInputs());
  if (outputMapIndex >= indexingMaps.size()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output map");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 2> rootReductionTileSizes;
  if (!reductionLoopDims.empty())
    rootReductionTileSizes.assign(candidateReductionTileSizes.begin(),
                                  candidateReductionTileSizes.end());

  std::optional<ComputeReduceKind> combineKind;
  if (!rootReductionTileSizes.empty()) {
    mlir::FailureOr<ComputeReduceKind> kind =
        getCandidateCombineKind(root, failureReason);
    if (mlir::failed(kind))
      return mlir::failure();
    combineKind = *kind;
  }

  mlir::FailureOr<llvm::SmallVector<ReductionChunk, 8>> reductionChunks =
      buildReductionChunks(root, rootReductionTileSizes, failureReason);
  if (mlir::failed(reductionChunks))
    return mlir::failure();

  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::Value, 4> valuesToTile(root->operand_begin(),
                                                 root->operand_end());
  unsigned initOperandIndex = static_cast<unsigned>(root.getNumDpsInputs());
  mlir::Value outputInitTile;
  mlir::Value neutralInitTile;
  mlir::Value accumulator;

  for (const ReductionChunk &chunk : *reductionChunks) {
    CandidateLoopTile loopTile;
    if (mlir::failed(buildCandidateLoopTile(
            builder, root.getLoc(), root, indexingMaps[outputMapIndex],
            candidateTileOffsets, candidateTileSizes, chunk.offsets,
            chunk.sizes, loopTile, failureReason)))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> tiledOperands =
        mlir::linalg::makeTiledShapes(builder, root.getLoc(), root,
                                      valuesToTile, loopTile.ivs,
                                      loopTile.tileSizes, loopTile.sizeBounds,
                                      /*omitPartialTileCheck=*/true);
    if (!outputInitTile) {
      outputInitTile = tiledOperands[initOperandIndex];
    } else if (initOperandIndex < tiledOperands.size()) {
      mlir::Operation *unusedInitSlice =
          tiledOperands[initOperandIndex].getDefiningOp();
      if (!neutralInitTile) {
        auto initType =
            mlir::dyn_cast<mlir::RankedTensorType>(outputInitTile.getType());
        if (!initType || !combineKind) {
          setFailureReason(
              failureReason,
              "candidate reduction split neutral init type mismatch");
          return mlir::failure();
        }
        mlir::FailureOr<mlir::Value> neutral = createNeutralInitTensor(
            builder, root.getLoc(), initType, *combineKind, failureReason);
        if (mlir::failed(neutral))
          return mlir::failure();
        neutralInitTile = *neutral;
      }
      tiledOperands[initOperandIndex] = neutralInitTile;
      if (unusedInitSlice && unusedInitSlice->use_empty())
        unusedInitSlice->erase();
    }

    llvm::SmallVector<mlir::Type, 2> resultTypes =
        mlir::linalg::getTensorOutputTypes(root, tiledOperands);
    if (resultTypes.size() != 1) {
      setFailureReason(
          failureReason,
          "candidate tile materialization expected one tiled result type");
      return mlir::failure();
    }

    mlir::Operation *tiled =
        mlir::clone(builder, root.getOperation(), resultTypes, tiledOperands);
    auto tiledLinalg = mlir::cast<mlir::linalg::LinalgOp>(tiled);
    mlir::linalg::offsetIndices(builder, tiledLinalg, loopTile.loopOffsets);

    builder.setInsertionPointAfter(tiled);
    mlir::Value partial = tiled->getResult(0);
    if (!accumulator) {
      accumulator = partial;
      continue;
    }

    if (!combineKind) {
      setFailureReason(failureReason,
                       "candidate reduction split missing combine kind");
      return mlir::failure();
    }
    mlir::FailureOr<mlir::Value> combined =
        createPartialCombine(builder, root.getLoc(), *combineKind, accumulator,
                             partial, outputInitTile, failureReason);
    if (mlir::failed(combined))
      return mlir::failure();
    accumulator = *combined;
    builder.setInsertionPointAfter(accumulator.getDefiningOp());
  }

  if (!accumulator) {
    setFailureReason(failureReason,
                     "candidate tile materialization produced no tiled result");
    return mlir::failure();
  }
  return accumulator;
}

mlir::FailureOr<mlir::Value>
getCandidateOutputBoundary(GroupOp group, unsigned outputIndex,
                           std::string *failureReason) {
  unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
  mlir::Block &body = group.getBody().front();
  if (inputCount + outputIndex >= body.getNumArguments()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output boundary");
    return mlir::failure();
  }
  return body.getArgument(inputCount + outputIndex);
}

mlir::Value
insertCandidateRootTile(mlir::linalg::LinalgOp root, mlir::Value tileValue,
                        mlir::Value outputDestination,
                        llvm::ArrayRef<int64_t> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes) {
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  offsets.reserve(candidateTileOffsets.size());
  sizes.reserve(candidateTileSizes.size());
  strides.reserve(candidateTileSizes.size());
  for (auto [offset, size] :
       llvm::zip(candidateTileOffsets, candidateTileSizes)) {
    offsets.push_back(builder.getIndexAttr(offset));
    sizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }

  auto inserted = builder.create<mlir::tensor::InsertSliceOp>(
      root.getLoc(), tileValue, outputDestination, offsets, sizes, strides);
  return inserted.getResult();
}

mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
collectCandidateRoots(GroupOp group, bool rejectProducerChains,
                      std::string *failureReason) {
  auto yield =
      mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
  if (!yield || yield.getValues().empty()) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires group results");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::linalg::LinalgOp, 4> roots;
  llvm::DenseSet<mlir::Operation *> seenRoots;
  for (auto [index, value] : llvm::enumerate(yield.getValues())) {
    auto root =
        mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(value.getDefiningOp());
    if (!root) {
      setFailureReason(failureReason,
                       "candidate tile materialization requires linalg roots");
      return mlir::failure();
    }
    if (root->getBlock() != &group.getBody().front() ||
        !seenRoots.insert(root.getOperation()).second ||
        root->getNumResults() != 1 || root.getNumDpsInits() != 1 ||
        root->getResult(0) != value) {
      setFailureReason(failureReason,
                       "candidate multi-output coverage requires distinct "
                       "single-result yielded roots");
      return mlir::failure();
    }

    unsigned matchingYieldUses = 0;
    for (mlir::OpOperand &use : value.getUses()) {
      if (use.getOwner() == yield.getOperation() &&
          use.getOperandNumber() == index) {
        ++matchingYieldUses;
        continue;
      }
      setFailureReason(failureReason, "candidate multi-output coverage "
                                      "requires independent yielded roots");
      return mlir::failure();
    }
    if (matchingYieldUses != 1) {
      setFailureReason(failureReason, "candidate multi-output coverage "
                                      "requires independent yielded roots");
      return mlir::failure();
    }

    if (rejectProducerChains) {
      for (mlir::Value input : root.getDpsInputs()) {
        if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
          continue;
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(input);
        if (blockArg && blockArg.getOwner() == &group.getBody().front())
          continue;
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support tensor producer "
            "chains");
        return mlir::failure();
      }

      mlir::Value init = root.getDpsInits().front();
      bool hasDirectOutputInit =
          isGroupOutputBoundary(group, init, static_cast<unsigned>(index));
      bool hasReduction = !getReductionLoopDims(root).empty();
      if (!hasDirectOutputInit &&
          (!hasReduction || !init.getDefiningOp<mlir::linalg::FillOp>())) {
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support output producer "
            "chains");
        return mlir::failure();
      }
    }
    roots.push_back(root);
  }

  return roots;
}

mlir::LogicalResult materializeCandidateTileSlices(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      collectCandidateRoots(group, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  auto yield =
      mlir::cast<GroupYieldOp>(group.getBody().front().getTerminator());

  llvm::SmallVector<mlir::Value, 4> insertedValues;
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> tileValue = materializeCandidateRootTileValue(
        group, root, static_cast<unsigned>(index), candidateTileOffsets,
        candidateTileSizes, candidateReductionTileSizes, failureReason);
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        group, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(tileValue) || mlir::failed(outputBoundary))
      return mlir::failure();
    insertedValues.push_back(
        insertCandidateRootTile(root, *tileValue, *outputBoundary,
                                candidateTileOffsets, candidateTileSizes));
  }

  for (auto [index, inserted] : llvm::enumerate(insertedValues))
    yield->setOperand(index, inserted);
  for (mlir::linalg::LinalgOp root : *roots)
    root->erase();
  return mlir::success();
}

} // namespace wafer::group_to_tile_region
