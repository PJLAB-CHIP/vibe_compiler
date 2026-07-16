//===- NamedComputeLowering.cpp - Named compute lowering ------------===//

#include "Internal.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

mlir::LogicalResult
TileRegionBodyEmitter::verifyNamedLinalgPayloads(TensorProgramScope scope) {
  mlir::WalkResult result = scope.getFunction().walk([&](mlir::Operation *op) {
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op)) {
      if (mlir::failed(verifyExactFillPayload(fill)))
        return mlir::WalkResult::interrupt();
    } else if (mlir::isa<mlir::linalg::MatmulOp>(op)) {
      if (mlir::failed(verifyExactGemmPayload(
              mlir::cast<mlir::linalg::LinalgOp>(op), "matmul")))
        return mlir::WalkResult::interrupt();
    } else if (mlir::isa<mlir::linalg::BatchMatmulOp>(op)) {
      if (mlir::failed(verifyExactGemmPayload(
              mlir::cast<mlir::linalg::LinalgOp>(op), "batch matmul")))
        return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::verifyExactFillPayload(mlir::linalg::FillOp fill) {
  mlir::linalg::LinalgOp op = fill;
  if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
      op.getRegionInputArgs().size() != 1 ||
      op.getRegionOutputArgs().size() != 1)
    return fail("linalg.fill requires the canonical scalar payload");

  mlir::Block &body = op->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != op.getRegionInputArgs().front() ||
      !body.without_terminator().empty())
    return fail("linalg.fill requires the canonical scalar payload");
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::verifyExactGemmPayload(mlir::linalg::LinalgOp op,
                                              llvm::StringRef subject) {
  if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
      op.getRegionInputArgs().size() != 2 ||
      op.getRegionOutputArgs().size() != 1)
    return fail(
        (subject + " requires an exact multiply-accumulate payload").str());

  mlir::Block &body = op->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  llvm::SmallVector<mlir::Operation *, 2> payloadOps;
  for (mlir::Operation &payloadOp : body.without_terminator())
    payloadOps.push_back(&payloadOp);
  if (!yield || yield.getValues().size() != 1 || payloadOps.size() != 2)
    return fail(
        (subject + " requires an exact multiply-accumulate payload").str());

  mlir::Value lhs = op.getRegionInputArgs()[0];
  mlir::Value rhs = op.getRegionInputArgs()[1];
  mlir::Value accumulator = op.getRegionOutputArgs()[0];
  auto matchesPair = [](mlir::Value first, mlir::Value second,
                        mlir::Value expectedFirst, mlir::Value expectedSecond) {
    return (first == expectedFirst && second == expectedSecond) ||
           (first == expectedSecond && second == expectedFirst);
  };

  mlir::Value sum;
  if (auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(payloadOps[0])) {
    auto add = mlir::dyn_cast<mlir::arith::AddFOp>(payloadOps[1]);
    if (!add || mul.getFastmath() != mlir::arith::FastMathFlags::none ||
        add.getFastmath() != mlir::arith::FastMathFlags::none ||
        !matchesPair(mul.getLhs(), mul.getRhs(), lhs, rhs) ||
        !matchesPair(add.getLhs(), add.getRhs(), mul.getResult(), accumulator))
      return fail(
          (subject + " requires an exact multiply-accumulate payload").str());
    sum = add.getResult();
  } else if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(payloadOps[0])) {
    auto add = mlir::dyn_cast<mlir::arith::AddIOp>(payloadOps[1]);
    if (!add ||
        mul.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none ||
        add.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none ||
        !matchesPair(mul.getLhs(), mul.getRhs(), lhs, rhs) ||
        !matchesPair(add.getLhs(), add.getRhs(), mul.getResult(), accumulator))
      return fail(
          (subject + " requires an exact multiply-accumulate payload").str());
    sum = add.getResult();
  } else {
    return fail(
        (subject + " requires an exact multiply-accumulate payload").str());
  }
  if (yield.getValues().front() != sum)
    return fail(
        (subject + " requires an exact multiply-accumulate payload").str());
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertFill(mlir::linalg::FillOp fill,
                                   mlir::OpBuilder &builder) {
  mlir::linalg::LinalgOp op = fill;
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      fill->getNumResults() != 1)
    return fail("unsupported linalg.fill arity");
  if (mlir::failed(verifyExactFillPayload(fill)))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> value = getScalarValue(op.getDpsInputs()[0]);
  if (mlir::failed(value))
    return mlir::failure();

  fillInitScalars[fill.getResult(0)] = *value;
  if (auto attrIt = scalarAttrs.find(op.getDpsInputs()[0]);
      attrIt != scalarAttrs.end())
    fillInitAttrs[fill.getResult(0)] = attrIt->second;

  // The target GEMM is overwrite-only. Preserve its source-level identity
  // proof without issuing a dead fill or loading an output buffer that the
  // successful GEMM path cannot consume.
  if (onlyFeedsGemmOverwriteInit(fill.getResult(0)))
    return mlir::success();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(fill.getResult(0).getType());
  if (!resultTensorType)
    return fail("linalg.fill result is not a ranked tensor");
  auto result = builder.create<mlir::memref::AllocOp>(
      fill.getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Tensor));
  builder.create<ComputeFillOp>(fill.getLoc(), result.getResult(), *value);
  record(fill.getResult(0), MemLayout::Tensor, result.getResult());
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::requireZeroFilledGemmInit(mlir::linalg::LinalgOp op,
                                                 llvm::StringRef subject) {
  if (op.getNumDpsInits() != 1)
    return fail((subject + " requires exactly one DPS init").str());

  auto isPositiveZero = [](mlir::Attribute attr) {
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
      const llvm::APFloat &value = floatAttr.getValue();
      return value.isZero() && !value.isNegative();
    }
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getValue().isZero();
    if (auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
      if (!elements.isSplat())
        return false;
      if (mlir::isa<mlir::FloatType>(elements.getElementType())) {
        llvm::APFloat value = elements.getSplatValue<mlir::APFloat>();
        return value.isZero() && !value.isNegative();
      }
      if (mlir::isa<mlir::IntegerType>(elements.getElementType()))
        return elements.getSplatValue<mlir::APInt>().isZero();
    }
    return false;
  };

  auto attrIt = fillInitAttrs.find(op.getDpsInits().front());
  if (attrIt == fillInitAttrs.end() || !isPositiveZero(attrIt->second)) {
    return fail((subject + " requires a provable zero-filled DPS init because "
                           "wafer.tile.gemm has overwrite semantics")
                    .str());
  }
  return mlir::success();
}

bool TileRegionBodyEmitter::hasOrderedGemmChunkInit(
    mlir::linalg::LinalgOp op) const {
  if (op.getNumDpsInits() != 1 || op->getNumResults() != 1)
    return false;

  mlir::Value init = op.getDpsInits().front();
  mlir::Operation *producer = init.getDefiningOp();
  if (!producer || producer->getNumResults() != 1 ||
      producer->getResult(0) != init ||
      init.getType() != op->getResult(0).getType())
    return false;

  if (mlir::isa<mlir::linalg::MatmulOp>(op.getOperation()))
    return mlir::isa<mlir::linalg::MatmulOp>(producer);
  if (mlir::isa<mlir::linalg::BatchMatmulOp>(op.getOperation()))
    return mlir::isa<mlir::linalg::BatchMatmulOp>(producer);
  return false;
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::createOrderedChunkCombine(
    mlir::Location loc, ComputeReduceKind kind, mlir::Value accumulator,
    mlir::Value partial, mlir::RankedTensorType resultTensorType,
    mlir::OpBuilder &builder) {
  ComputeElementwiseKind elementwiseKind;
  switch (kind) {
  case ComputeReduceKind::Sum:
    elementwiseKind = ComputeElementwiseKind::Add;
    break;
  case ComputeReduceKind::Max:
    elementwiseKind = ComputeElementwiseKind::Max;
    break;
  case ComputeReduceKind::Min:
    elementwiseKind = ComputeElementwiseKind::Min;
    break;
  case ComputeReduceKind::Avg:
    return failValue("ordered reduction chunks do not support average");
  }

  mlir::FailureOr<mlir::Value> previous =
      getOrMaterialize(accumulator, MemLayout::Tensor, builder);
  if (mlir::failed(previous))
    return mlir::failure();

  mlir::Type tensorBufferType =
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  mlir::Value partialTensor = partial;
  if (partialTensor.getType() != tensorBufferType) {
    partialTensor =
        builder
            .create<LayoutMaterializeOp>(loc, tensorBufferType, partialTensor)
            .getResult();
  }
  if ((*previous).getType() != tensorBufferType)
    return failValue("ordered reduction chunk accumulator type mismatch");

  auto kindAttr =
      ComputeElementwiseKindAttr::get(builder.getContext(), elementwiseKind);
  auto combined = builder.create<ComputeElementwiseOp>(
      loc, tensorBufferType, kindAttr,
      mlir::ValueRange{*previous, partialTensor});
  return combined.getResult();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertMatmul(mlir::linalg::LinalgOp op,
                                     mlir::OpBuilder &builder) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return fail("unsupported matmul arity");
  if (mlir::failed(verifyExactGemmPayload(op, "matmul")))
    return mlir::failure();
  bool orderedChunk = hasOrderedGemmChunkInit(op);
  if (!orderedChunk && mlir::failed(requireZeroFilledGemmInit(op, "matmul")))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> lhs =
      getOrMaterialize(op.getDpsInputs()[0], MemLayout::Cx, builder);
  mlir::FailureOr<mlir::Value> rhs =
      getOrMaterialize(op.getDpsInputs()[1], MemLayout::Cx, builder);
  if (mlir::failed(lhs) || mlir::failed(rhs))
    return mlir::failure();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultTensorType)
    return fail("matmul result is not a ranked tensor");

  auto gemm = builder.create<ComputeGemmOp>(
      op->getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Cx), *lhs,
      *rhs);
  if (orderedChunk) {
    mlir::FailureOr<mlir::Value> combined = createOrderedChunkCombine(
        op->getLoc(), ComputeReduceKind::Sum, op.getDpsInits().front(),
        gemm.getResult(), resultTensorType, builder);
    if (mlir::failed(combined))
      return mlir::failure();
    record(op->getResult(0), MemLayout::Tensor, *combined);
    return mlir::success();
  }
  record(op->getResult(0), MemLayout::Cx, gemm.getResult());
  return mlir::success();
}

mlir::FailureOr<unsigned> TileRegionBodyEmitter::findOperandDimForLoop(
    mlir::AffineMap map, unsigned loopDim, llvm::StringRef role) {
  for (auto [operandDim, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr)
      return failUnsigned("batch matmul requires projected permutation " +
                          role.str() + " indexing map");
    if (dimExpr.getPosition() == loopDim)
      return static_cast<unsigned>(operandDim);
  }
  return failUnsigned("batch matmul indexing map is missing " + role.str() +
                      " loop dimension");
}

bool TileRegionBodyEmitter::mapContainsLoopDim(mlir::AffineMap map,
                                               unsigned loopDim) const {
  for (mlir::AffineExpr expr : map.getResults()) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (dimExpr && dimExpr.getPosition() == loopDim)
      return true;
  }
  return false;
}

mlir::FailureOr<BatchedGemmAttrs>
TileRegionBodyEmitter::inferBatchMatmulAttrs(mlir::linalg::LinalgOp op) {
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      op.getIteratorTypesArray();
  llvm::SmallVector<unsigned, 1> reductionLoops;
  for (auto [index, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      reductionLoops.push_back(static_cast<unsigned>(index));
  }
  if (reductionLoops.size() != 1)
    return mlir::failure();

  llvm::SmallVector<mlir::AffineMap, 4> maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return mlir::failure();
  mlir::AffineMap lhsMap = maps[0];
  mlir::AffineMap rhsMap = maps[1];
  mlir::AffineMap resultMap = maps[2];

  std::optional<unsigned> mLoop;
  std::optional<unsigned> nLoop;
  llvm::SmallVector<unsigned, 4> batchLoops;
  unsigned reductionLoop = reductionLoops.front();
  for (unsigned loopDim = 0; loopDim < iteratorTypes.size(); ++loopDim) {
    if (loopDim == reductionLoop)
      continue;
    bool inLhs = mapContainsLoopDim(lhsMap, loopDim);
    bool inRhs = mapContainsLoopDim(rhsMap, loopDim);
    bool inResult = mapContainsLoopDim(resultMap, loopDim);
    if (inLhs && !inRhs && inResult) {
      if (mLoop)
        return mlir::failure();
      mLoop = loopDim;
      continue;
    }
    if (!inLhs && inRhs && inResult) {
      if (nLoop)
        return mlir::failure();
      nLoop = loopDim;
      continue;
    }
    if (inLhs && inRhs && inResult) {
      batchLoops.push_back(loopDim);
      continue;
    }
    return mlir::failure();
  }
  if (!mLoop || !nLoop || batchLoops.empty())
    return mlir::failure();

  BatchedGemmAttrs attrs;
  auto lhsMDim = findOperandDimForLoop(lhsMap, *mLoop, "lhs M");
  auto lhsKDim =
      findOperandDimForLoop(lhsMap, reductionLoop, "lhs contracting");
  auto rhsKDim =
      findOperandDimForLoop(rhsMap, reductionLoop, "rhs contracting");
  auto rhsNDim = findOperandDimForLoop(rhsMap, *nLoop, "rhs N");
  auto resultMDim = findOperandDimForLoop(resultMap, *mLoop, "result M");
  auto resultNDim = findOperandDimForLoop(resultMap, *nLoop, "result N");
  if (mlir::failed(lhsMDim) || mlir::failed(lhsKDim) || mlir::failed(rhsKDim) ||
      mlir::failed(rhsNDim) || mlir::failed(resultMDim) ||
      mlir::failed(resultNDim))
    return mlir::failure();
  attrs.lhsMDim = *lhsMDim;
  attrs.lhsContractingDim = *lhsKDim;
  attrs.rhsContractingDim = *rhsKDim;
  attrs.rhsNDim = *rhsNDim;
  attrs.resultMDim = *resultMDim;
  attrs.resultNDim = *resultNDim;

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultTensorType || !resultTensorType.hasStaticShape())
    return mlir::failure();

  attrs.batchCount = 1;
  for (unsigned batchLoop : batchLoops) {
    auto lhsBatchDim = findOperandDimForLoop(lhsMap, batchLoop, "lhs batch");
    auto rhsBatchDim = findOperandDimForLoop(rhsMap, batchLoop, "rhs batch");
    auto resultBatchDim =
        findOperandDimForLoop(resultMap, batchLoop, "result batch");
    if (mlir::failed(lhsBatchDim) || mlir::failed(rhsBatchDim) ||
        mlir::failed(resultBatchDim))
      return mlir::failure();
    attrs.lhsBatchDims.push_back(*lhsBatchDim);
    attrs.rhsBatchDims.push_back(*rhsBatchDim);
    attrs.resultBatchDims.push_back(*resultBatchDim);
    int64_t dimSize = resultTensorType.getDimSize(*resultBatchDim);
    if (mlir::ShapedType::isDynamic(dimSize) || dimSize <= 0)
      return mlir::failure();
    if (attrs.batchCount > std::numeric_limits<int64_t>::max() / dimSize)
      return mlir::failure();
    attrs.batchCount *= dimSize;
  }
  return attrs;
}

mlir::LogicalResult
TileRegionBodyEmitter::convertBatchMatmul(mlir::linalg::LinalgOp op,
                                          mlir::OpBuilder &builder) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return fail("unsupported batch matmul arity");
  if (mlir::failed(verifyExactGemmPayload(op, "batch matmul")))
    return mlir::failure();
  bool orderedChunk = hasOrderedGemmChunkInit(op);
  if (!orderedChunk &&
      mlir::failed(requireZeroFilledGemmInit(op, "batch matmul")))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> lhs =
      getOrMaterialize(op.getDpsInputs()[0], MemLayout::Cx, builder);
  mlir::FailureOr<mlir::Value> rhs =
      getOrMaterialize(op.getDpsInputs()[1], MemLayout::Cx, builder);
  if (mlir::failed(lhs) || mlir::failed(rhs))
    return mlir::failure();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultTensorType)
    return fail("batch matmul result is not a ranked tensor");

  mlir::FailureOr<BatchedGemmAttrs> attrs = inferBatchMatmulAttrs(op);
  if (mlir::failed(attrs))
    return fail("unsupported batch matmul indexing");

  auto gemm = builder.create<ComputeGemmOp>(
      op->getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Cx), *lhs,
      *rhs);
  gemm->setAttr("batch_count", builder.getI64IntegerAttr(attrs->batchCount));
  gemm->setAttr("lhs_batch_dims",
                builder.getDenseI64ArrayAttr(attrs->lhsBatchDims));
  gemm->setAttr("lhs_m_dim", builder.getI64IntegerAttr(attrs->lhsMDim));
  gemm->setAttr("lhs_contracting_dim",
                builder.getI64IntegerAttr(attrs->lhsContractingDim));
  gemm->setAttr("rhs_batch_dims",
                builder.getDenseI64ArrayAttr(attrs->rhsBatchDims));
  gemm->setAttr("rhs_contracting_dim",
                builder.getI64IntegerAttr(attrs->rhsContractingDim));
  gemm->setAttr("rhs_n_dim", builder.getI64IntegerAttr(attrs->rhsNDim));
  gemm->setAttr("result_batch_dims",
                builder.getDenseI64ArrayAttr(attrs->resultBatchDims));
  gemm->setAttr("result_m_dim", builder.getI64IntegerAttr(attrs->resultMDim));
  gemm->setAttr("result_n_dim", builder.getI64IntegerAttr(attrs->resultNDim));
  if (orderedChunk) {
    mlir::FailureOr<mlir::Value> combined = createOrderedChunkCombine(
        op->getLoc(), ComputeReduceKind::Sum, op.getDpsInits().front(),
        gemm.getResult(), resultTensorType, builder);
    if (mlir::failed(combined))
      return mlir::failure();
    record(op->getResult(0), MemLayout::Tensor, *combined);
    return mlir::success();
  }
  record(op->getResult(0), MemLayout::Cx, gemm.getResult());
  return mlir::success();
}

std::optional<ComputeElementwiseKind>
TileRegionBodyEmitter::inferCompareKind(mlir::arith::CmpFPredicate predicate) {
  switch (predicate) {
  case mlir::arith::CmpFPredicate::OEQ:
    return ComputeElementwiseKind::Eq;
  case mlir::arith::CmpFPredicate::UNE:
    return ComputeElementwiseKind::Ne;
  case mlir::arith::CmpFPredicate::OLT:
    return ComputeElementwiseKind::Lt;
  case mlir::arith::CmpFPredicate::OLE:
    return ComputeElementwiseKind::Le;
  case mlir::arith::CmpFPredicate::OGT:
    return ComputeElementwiseKind::Gt;
  case mlir::arith::CmpFPredicate::OGE:
    return ComputeElementwiseKind::Ge;
  case mlir::arith::CmpFPredicate::UEQ:
  case mlir::arith::CmpFPredicate::ONE:
  case mlir::arith::CmpFPredicate::ULT:
  case mlir::arith::CmpFPredicate::ULE:
  case mlir::arith::CmpFPredicate::UGT:
  case mlir::arith::CmpFPredicate::UGE:
  case mlir::arith::CmpFPredicate::AlwaysFalse:
  case mlir::arith::CmpFPredicate::ORD:
  case mlir::arith::CmpFPredicate::UNO:
  case mlir::arith::CmpFPredicate::AlwaysTrue:
    (void)fail("arith.cmpf predicate does not match the current target "
               "comparison NaN semantics");
    return std::nullopt;
  }
  llvm_unreachable("unknown cmpf predicate");
}

std::optional<ComputeElementwiseKind>
TileRegionBodyEmitter::inferCompareKind(mlir::arith::CmpIPredicate predicate) {
  switch (predicate) {
  case mlir::arith::CmpIPredicate::eq:
    return ComputeElementwiseKind::Eq;
  case mlir::arith::CmpIPredicate::ne:
    return ComputeElementwiseKind::Ne;
  case mlir::arith::CmpIPredicate::slt:
    return ComputeElementwiseKind::Lt;
  case mlir::arith::CmpIPredicate::sle:
    return ComputeElementwiseKind::Le;
  case mlir::arith::CmpIPredicate::sgt:
    return ComputeElementwiseKind::Gt;
  case mlir::arith::CmpIPredicate::sge:
    return ComputeElementwiseKind::Ge;
  case mlir::arith::CmpIPredicate::ult:
  case mlir::arith::CmpIPredicate::ule:
  case mlir::arith::CmpIPredicate::ugt:
  case mlir::arith::CmpIPredicate::uge:
    (void)fail("unsigned arith.cmpi predicate cannot be represented by the "
               "current signed target comparison kind");
    return std::nullopt;
  }
  llvm_unreachable("unknown cmpi predicate");
}

std::optional<ComputeReduceKind>
TileRegionBodyEmitter::inferReduceKind(mlir::linalg::GenericOp generic) {
  if (generic.getRegionInputArgs().size() != 1 ||
      generic.getRegionOutputArgs().size() != 1) {
    (void)fail("linalg.generic reduction requires one input and one "
               "accumulator");
    return std::nullopt;
  }
  return matchExactReductionKind(generic.getRegionOutputArgs(), /*redPos=*/0,
                                 generic.getRegionInputArgs().front(),
                                 "linalg.generic reduction", failureReason);
}

mlir::FailureOr<bool> TileRegionBodyEmitter::hasOrderedReduceChunkInit(
    mlir::linalg::GenericOp generic, ComputeReduceKind kind) {
  mlir::Value init = generic.getDpsInits().front();
  auto producer = init.getDefiningOp<mlir::linalg::GenericOp>();
  if (!producer || !hasReductionIterator(producer))
    return false;
  if (producer.getNumDpsInputs() != 1 || producer.getNumDpsInits() != 1 ||
      producer->getNumResults() != 1 || producer->getResult(0) != init ||
      init.getType() != generic->getResult(0).getType()) {
    (void)fail("ordered reduction chunk init must be the direct prior "
               "single-result reduction");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 2> currentDims;
  llvm::SmallVector<int64_t, 2> producerDims;
  if (mlir::failed(getReductionInputDims(generic, currentDims)) ||
      mlir::failed(getReductionInputDims(producer, producerDims)))
    return mlir::failure();
  if (currentDims.size() != 1 || producerDims.size() != 1) {
    (void)fail("ordered reduction chunk chain requires exactly one "
               "reduction axis");
    return mlir::failure();
  }

  std::optional<ComputeReduceKind> producerKind = inferReduceKind(producer);
  if (!producerKind)
    return mlir::failure();
  if (*producerKind != kind) {
    (void)fail("ordered reduction chunks require the same exact combiner");
    return mlir::failure();
  }
  return true;
}

mlir::FailureOr<mlir::TypedAttr>
TileRegionBodyEmitter::getNeutralReduceInit(mlir::Type elementType,
                                            ComputeReduceKind kind) {
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

  (void)fail(
      "ordered reduction chunks require float or integer accumulator type");
  return mlir::failure();
}

bool TileRegionBodyEmitter::hasReductionIterator(
    mlir::linalg::GenericOp generic) const {
  for (mlir::utils::IteratorType iteratorType :
       generic.getIteratorTypesArray()) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      return true;
  }
  return false;
}

mlir::LogicalResult TileRegionBodyEmitter::getReductionInputDims(
    mlir::linalg::GenericOp generic,
    llvm::SmallVectorImpl<int64_t> &inputDims) {
  inputDims.clear();
  if (generic.getNumDpsInputs() != 1)
    return fail("unsupported reduction input arity");

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      generic.getIndexingMapsArray();
  if (indexingMaps.empty())
    return fail("reduction generic has no indexing map");
  mlir::AffineMap inputMap = indexingMaps[0];

  llvm::SmallVector<unsigned, 4> reductionLoopDims;
  for (auto [index, iteratorType] :
       llvm::enumerate(generic.getIteratorTypesArray())) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      reductionLoopDims.push_back(static_cast<unsigned>(index));
  }
  if (reductionLoopDims.empty())
    return fail("reduction generic has no reduction dimensions");

  for (unsigned loopDim : reductionLoopDims) {
    std::optional<int64_t> inputDim;
    for (auto [dimIndex, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr)
        return fail("unsupported reduction input indexing map");
      if (dimExpr.getPosition() == loopDim) {
        inputDim = static_cast<int64_t>(dimIndex);
        break;
      }
    }
    if (!inputDim)
      return fail("reduction dimension is not present in input map");
    inputDims.push_back(*inputDim);
  }
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::createReduceOp(
    mlir::Location loc, mlir::Type resultType, ComputeReduceKindAttr kindAttr,
    mlir::Value input, llvm::ArrayRef<int64_t> dims, mlir::Value init,
    mlir::Attribute initAttr, mlir::OpBuilder &builder, mlir::Value &result) {
  mlir::OperationState state(loc, ComputeReduceOp::getOperationName());
  state.addAttribute("kind", kindAttr);
  state.addAttribute("dimensions",
                     mlir::DenseI64ArrayAttr::get(builder.getContext(), dims));
  if (initAttr)
    state.addAttribute("init_value", initAttr);
  state.addOperands(input);
  if (init)
    state.addOperands(init);
  state.addTypes(resultType);
  mlir::Operation *op = builder.create(state);
  result = op->getResult(0);
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertReduceGeneric(mlir::linalg::GenericOp generic,
                                            mlir::OpBuilder &builder) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1)
    return fail("unsupported reduction generic arity");

  std::optional<ComputeReduceKind> kind = inferReduceKind(generic);
  if (!kind)
    return mlir::failure();

  llvm::SmallVector<int64_t, 4> reduceDims;
  if (mlir::failed(getReductionInputDims(generic, reduceDims)))
    return mlir::failure();

  mlir::FailureOr<bool> orderedChunk =
      hasOrderedReduceChunkInit(generic, *kind);
  if (mlir::failed(orderedChunk))
    return mlir::failure();

  mlir::Value initTensor = generic.getDpsInits()[0];
  mlir::Value initScalar;
  mlir::Attribute initAttr;

  auto inputTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
      generic.getDpsInputs()[0].getType());
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!inputTensorType || !resultTensorType)
    return fail("reduction generic operands/results must be ranked tensors");

  if (*orderedChunk) {
    mlir::FailureOr<mlir::TypedAttr> neutral =
        getNeutralReduceInit(resultTensorType.getElementType(), *kind);
    if (mlir::failed(neutral))
      return mlir::failure();
    initAttr = *neutral;
  } else {
    if (auto attrIt = fillInitAttrs.find(initTensor);
        attrIt != fillInitAttrs.end())
      initAttr = attrIt->second;
    if (auto scalarIt = fillInitScalars.find(initTensor);
        scalarIt != fillInitScalars.end())
      initScalar = scalarIt->second;
    if (initAttr)
      initScalar = {};
    if (!initAttr && !initScalar)
      return fail("missing reduction init scalar");
    if (!initAttr) {
      auto constant = initScalar.getDefiningOp<mlir::arith::ConstantOp>();
      auto typedValue =
          constant ? mlir::dyn_cast<mlir::TypedAttr>(constant.getValue())
                   : mlir::TypedAttr{};
      if (!constant || !typedValue)
        return fail(
            "reduction init must be an arith.constant or typed init_value");
    }
  }

  mlir::FailureOr<mlir::Value> input =
      getOrMaterialize(generic.getDpsInputs()[0],
                       alignedLayoutForTensor(inputTensorType), builder);
  if (mlir::failed(input))
    return mlir::failure();

  mlir::Type reduceResultType = makeSPMMemRefType(
      resultTensorType, alignedLayoutForTensor(resultTensorType));
  mlir::Value reduceResult;
  auto kindAttr = ComputeReduceKindAttr::get(generic.getContext(), *kind);
  if (mlir::failed(createReduceOp(generic.getLoc(), reduceResultType, kindAttr,
                                  *input, reduceDims, initScalar, initAttr,
                                  builder, reduceResult)))
    return mlir::failure();

  if (*orderedChunk) {
    mlir::FailureOr<mlir::Value> combined =
        createOrderedChunkCombine(generic.getLoc(), *kind, initTensor,
                                  reduceResult, resultTensorType, builder);
    if (mlir::failed(combined))
      return mlir::failure();
    record(generic->getResult(0), MemLayout::Tensor, *combined);
    return mlir::success();
  }

  record(generic->getResult(0), alignedLayoutForTensor(resultTensorType),
         reduceResult);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
