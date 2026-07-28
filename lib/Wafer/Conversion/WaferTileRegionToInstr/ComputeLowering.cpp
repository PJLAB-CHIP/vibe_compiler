//===- ComputeLowering.cpp - Tile-region compute lowering --------------===//

#include "Internal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

static mlir::FailureOr<int64_t>
getStaticDim(mlir::PatternRewriter &rewriter, mlir::Operation *op,
             mlir::RankedTensorType type, int64_t dim,
             std::string *failureReason, llvm::StringRef role) {
  int64_t value = type.getDimSize(dim);
  if (value == mlir::ShapedType::kDynamic)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine(role).concat(" requires static GEMM dimensions").str());
  return value;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 3>>
inferGemmMKN(ComputeGemmOp op, mlir::PatternRewriter &rewriter,
             std::string *failureReason) {
  std::optional<mlir::RankedTensorType> lhs =
      getLogicalTensorTypeFromMemRef(op.getLhs().getType());
  std::optional<mlir::RankedTensorType> rhs =
      getLogicalTensorTypeFromMemRef(op.getRhs().getType());
  std::optional<mlir::RankedTensorType> result =
      getLogicalTensorTypeFromMemRef(op.getResult().getType());
  if (!lhs || !rhs || !result)
    return failFailureOr<llvm::SmallVector<int64_t, 3>>(
        rewriter, op, failureReason,
        "tile.gemm lowering requires Wafer memref operands");

  if (lhs->getRank() != 2 || rhs->getRank() != 2 || result->getRank() != 2) {
    mlir::FailureOr<int64_t> lhsMDim =
        readRequiredI64Attr(rewriter, op, "lhs_m_dim", failureReason);
    mlir::FailureOr<int64_t> lhsKDim =
        readRequiredI64Attr(rewriter, op, "lhs_contracting_dim", failureReason);
    mlir::FailureOr<int64_t> rhsNDim =
        readRequiredI64Attr(rewriter, op, "rhs_n_dim", failureReason);
    if (mlir::failed(lhsMDim) || mlir::failed(lhsKDim) || mlir::failed(rhsNDim))
      return mlir::failure();
    mlir::FailureOr<int64_t> m = getStaticDim(
        rewriter, op, *lhs, *lhsMDim, failureReason, "batched tile.gemm");
    mlir::FailureOr<int64_t> k = getStaticDim(
        rewriter, op, *lhs, *lhsKDim, failureReason, "batched tile.gemm");
    mlir::FailureOr<int64_t> n = getStaticDim(
        rewriter, op, *rhs, *rhsNDim, failureReason, "batched tile.gemm");
    if (mlir::failed(m) || mlir::failed(k) || mlir::failed(n))
      return mlir::failure();
    return llvm::SmallVector<int64_t, 3>{*m, *k, *n};
  }

  GemmOrientation lhsOrientation =
      op.getLhsOrientation().value_or(GemmOrientation::Normal);
  GemmOrientation rhsOrientation =
      op.getRhsOrientation().value_or(GemmOrientation::Normal);
  int64_t lhsMDim = lhsOrientation == GemmOrientation::Normal ? 0 : 1;
  int64_t lhsKDim = lhsOrientation == GemmOrientation::Normal ? 1 : 0;
  int64_t rhsNDim = rhsOrientation == GemmOrientation::Normal ? 1 : 0;
  mlir::FailureOr<int64_t> m = getStaticDim(rewriter, op, *lhs, lhsMDim,
                                            failureReason, "rank-2 tile.gemm");
  mlir::FailureOr<int64_t> k = getStaticDim(rewriter, op, *lhs, lhsKDim,
                                            failureReason, "rank-2 tile.gemm");
  mlir::FailureOr<int64_t> n = getStaticDim(rewriter, op, *rhs, rhsNDim,
                                            failureReason, "rank-2 tile.gemm");
  if (mlir::failed(m) || mlir::failed(k) || mlir::failed(n))
    return mlir::failure();
  return llvm::SmallVector<int64_t, 3>{*m, *k, *n};
}

static mlir::FailureOr<InstrElementwiseKindAttr> getInstrElementwiseKindAttr(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeElementwiseKindAttr computeKind, std::string *failureReason);

class FillLowering : public mlir::OpRewritePattern<ComputeFillOp> {
public:
  using mlir::OpRewritePattern<ComputeFillOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ComputeFillOp op,
                  mlir::PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<InstrFillOp>(op, op.getDest(), op.getValue(),
                                             op.getFillDomainAttr());
    return mlir::success();
  }
};

static std::optional<InstrConvertKind>
resolveInstrConvertKind(mlir::Type sourceType, mlir::Type resultType) {
  for (uint32_t raw = 0; raw <= getMaxEnumValForInstrConvertKind(); ++raw) {
    std::optional<InstrConvertKind> kind = symbolizeInstrConvertKind(raw);
    if (!kind)
      continue;
    auto [expectedSource, expectedResult] =
        getInstrConvertTypePair(sourceType.getContext(), *kind);
    if (sourceType == expectedSource && resultType == expectedResult)
      return kind;
  }
  return std::nullopt;
}

class ConvertLowering : public mlir::OpRewritePattern<ComputeConvertOp> {
public:
  ConvertLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ComputeConvertOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeConvertOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.compute.convert requires memref storage");
    std::optional<InstrConvertKind> kind = resolveInstrConvertKind(
        sourceType.getElementType(), resultType.getElementType());
    if (!kind)
      return failPattern(
          rewriter, op, failureReason,
          "tile.compute.convert has no target instruction route");

    mlir::IntegerAttr zeroPoint;
    mlir::IntegerAttr roundingMode;
    switch (getInstrConvertParameterKind(*kind)) {
    case InstrConvertParameterKind::None:
      break;
    case InstrConvertParameterKind::RoundingMode:
      // Target rounding mode zero is the canonical nearest-mode route used by
      // framework floating truncation until board calibration proves a more
      // specific source-level rounding contract is required.
      roundingMode = rewriter.getI64IntegerAttr(0);
      break;
    case InstrConvertParameterKind::ZeroPoint:
      return failPattern(rewriter, op, failureReason,
                         "tile.compute.convert floating route unexpectedly "
                         "requires zero_point");
    }

    mlir::Value dest =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), resultType)
            .getResult();
    rewriter.create<InstrConvertOp>(
        op.getLoc(), InstrConvertKindAttr::get(rewriter.getContext(), *kind),
        op.getSource(), dest, zeroPoint, roundingMode,
        getDefaultNCCWorkerAttr(rewriter));
    rewriter.replaceOp(op, dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

// A select fed by a private, constant-filled predicate does not require a
// target boolean fill or mask operation.  Canonicalize that exact tile-level
// dataflow to an explicit fresh copy before conversion patterns can lower the
// fill independently.  Keeping this as a separate typed prepass makes the
// rewrite independent of cross-root dialect-conversion pattern ordering.
class ConstantPredicateSelectToCopy
    : public mlir::OpRewritePattern<ComputeElementwiseOp> {
public:
  using mlir::OpRewritePattern<ComputeElementwiseOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ComputeElementwiseOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (op.getKind() != ComputeElementwiseKind::Select ||
        op.getInputs().size() != 3)
      return mlir::failure();

    mlir::Value predicate = op.getInputs().front();
    auto predicateAlloc = predicate.getDefiningOp<mlir::memref::AllocOp>();
    if (!predicateAlloc)
      return mlir::failure();

    mlir::OpOperand *selectPredicateUse = &op->getOpOperand(0);
    ComputeFillOp predicateFill;
    unsigned selectUseCount = 0;
    unsigned fillUseCount = 0;
    for (mlir::OpOperand &use : predicate.getUses()) {
      if (&use == selectPredicateUse) {
        ++selectUseCount;
        continue;
      }
      auto fill = mlir::dyn_cast<ComputeFillOp>(use.getOwner());
      if (!fill || &use != &fill.getDestMutable())
        return mlir::failure();
      predicateFill = fill;
      ++fillUseCount;
    }
    if (selectUseCount != 1 || fillUseCount != 1 || !predicateFill)
      return mlir::failure();

    if (predicateFill->getBlock() != op->getBlock() ||
        !predicateFill->isBeforeInBlock(op))
      return mlir::failure();

    auto constant =
        predicateFill.getValue().getDefiningOp<mlir::arith::ConstantOp>();
    auto valueAttr =
        constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                 : mlir::IntegerAttr{};
    if (!constant || !constant.getType().isInteger(1) || !valueAttr ||
        !valueAttr.getType().isInteger(1))
      return mlir::failure();

    unsigned selectedInputIndex = valueAttr.getValue().isZero() ? 2 : 1;
    mlir::Value selected = op.getInputs()[selectedInputIndex];
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    auto selectedType = mlir::dyn_cast<mlir::MemRefType>(selected.getType());
    if (!resultType || !selectedType || selectedType != resultType)
      return mlir::failure();

    if (mlir::Attribute rawMaps = op->getAttr("indexing_maps")) {
      auto maps = mlir::dyn_cast<mlir::ArrayAttr>(rawMaps);
      if (!maps || maps.size() != op.getInputs().size() + 1)
        return mlir::failure();
      auto hasIdentityMap = [&](unsigned mapIndex) {
        auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(maps[mapIndex]);
        if (!mapAttr)
          return false;
        mlir::AffineMap map = mapAttr.getValue();
        return map.getNumDims() == resultType.getRank() &&
               map.getNumSymbols() == 0 &&
               map.getNumResults() == resultType.getRank() && map.isIdentity();
      };
      if (!hasIdentityMap(selectedInputIndex) ||
          !hasIdentityMap(maps.size() - 1))
        return mlir::failure();
    }

    auto copy = rewriter.create<MoveCopyOp>(op.getLoc(), resultType, selected);
    rewriter.replaceOp(op, copy.getResult());
    rewriter.eraseOp(predicateFill);
    rewriter.eraseOp(predicateAlloc);
    if (constant->use_empty())
      rewriter.eraseOp(constant);
    return mlir::success();
  }
};

class ElementwiseLowering
    : public mlir::OpRewritePattern<ComputeElementwiseOp> {
public:
  ElementwiseLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ComputeElementwiseOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeElementwiseOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (op.getKind() == ComputeElementwiseKind::Select) {
      if (op.getInputs().size() != 3)
        return failPattern(
            rewriter, op, failureReason,
            "target select lowering requires predicate, true and false "
            "operands");
      auto destType =
          mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
      if (!destType || !mlir::isa<mlir::FloatType>(destType.getElementType()))
        return failPattern(
            rewriter, op, failureReason,
            "target select lowering currently requires floating-point values");
    }

    struct InputMovementPlan {
      mlir::Value source;
      mlir::MemRefType materializedType;
      llvm::SmallVector<LogicalMovementSegment> segments;
    };

    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.elementwise lowering requires a memref result");

    // Preflight every map before creating an allocation or an instruction.
    // A failed conversion therefore cannot leave a partially materialized
    // operand sequence in the pattern rewriter.
    llvm::SmallVector<InputMovementPlan, 3> movementPlans;
    movementPlans.reserve(op.getInputs().size());
    mlir::Attribute rawIndexingMaps = op->getAttr("indexing_maps");
    mlir::ArrayAttr indexingMaps;
    if (rawIndexingMaps) {
      indexingMaps = mlir::dyn_cast<mlir::ArrayAttr>(rawIndexingMaps);
      if (!indexingMaps)
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise indexing_maps must be an array attribute");
      if (indexingMaps.size() != op.getInputs().size() + 1)
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise indexing map count must match inputs plus "
            "result");

      auto resultMapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(
          indexingMaps[indexingMaps.size() - 1]);
      if (!resultMapAttr ||
          resultMapAttr.getValue().getNumDims() != resultType.getRank() ||
          resultMapAttr.getValue().getNumSymbols() != 0 ||
          !resultMapAttr.getValue().isIdentity())
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise result indexing map must be identity");
    }

    for (auto [index, input] : llvm::enumerate(op.getInputs())) {
      InputMovementPlan plan;
      plan.source = input;
      if (!indexingMaps) {
        movementPlans.push_back(std::move(plan));
        continue;
      }

      auto sourceType = mlir::dyn_cast<mlir::MemRefType>(input.getType());
      auto inputMapAttr =
          mlir::dyn_cast<mlir::AffineMapAttr>(indexingMaps[index]);
      if (!sourceType || !inputMapAttr)
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise indexing map materialization requires memref "
            "inputs and affine maps");

      mlir::AffineMap inputMap = inputMapAttr.getValue();
      if (inputMap.getNumDims() != resultType.getRank() ||
          inputMap.getNumSymbols() != 0 ||
          inputMap.getNumResults() != sourceType.getRank() ||
          !inputMap.isProjectedPermutation())
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise input indexing map must be a projected "
            "permutation of result dimensions");

      if (inputMap.isIdentity() &&
          sourceType.getShape() == resultType.getShape()) {
        movementPlans.push_back(std::move(plan));
        continue;
      }

      plan.materializedType = mlir::MemRefType::get(
          resultType.getShape(), sourceType.getElementType(),
          resultType.getLayout(), resultType.getMemorySpace());
      auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices,
                               llvm::SmallVectorImpl<int64_t> &sourceIndices) {
        sourceIndices.reserve(inputMap.getNumResults());
        for (mlir::AffineExpr expr : inputMap.getResults()) {
          auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
          if (!dimExpr || dimExpr.getPosition() >= resultIndices.size())
            return mlir::failure();
          sourceIndices.push_back(resultIndices[dimExpr.getPosition()]);
        }
        return mlir::success();
      };
      auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices,
                            llvm::SmallVectorImpl<int64_t> &destIndices) {
        destIndices.assign(resultIndices.begin(), resultIndices.end());
        return mlir::success();
      };
      mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
          getStaticMappedMovementSegments(
              rewriter, op, sourceType, plan.materializedType,
              resultType.getShape(), sourceIndexFn, destIndexFn, failureReason,
              "tile.elementwise indexing map materialization");
      if (mlir::failed(segments))
        return mlir::failure();
      plan.segments = std::move(*segments);
      movementPlans.push_back(std::move(plan));
    }

    InstrElementwiseKindAttr instrKind;
    if (op.getKind() != ComputeElementwiseKind::Select) {
      mlir::FailureOr<InstrElementwiseKindAttr> resolvedInstrKind =
          getInstrElementwiseKindAttr(rewriter, op, op.getKindAttr(),
                                      failureReason);
      if (mlir::failed(resolvedInstrKind))
        return mlir::failure();
      instrKind = *resolvedInstrKind;
    }

    // Select has a movement-based target sequence. Preflight its copy
    // descriptors as well, still before emitting any effect.
    mlir::FailureOr<MovementDescriptor> falseDescriptor;
    mlir::FailureOr<MovementDescriptor> selectDestDescriptor;
    if (op.getKind() == ComputeElementwiseKind::Select) {
      mlir::Type falseType = movementPlans[2].source.getType();
      if (movementPlans[2].materializedType)
        falseType = movementPlans[2].materializedType;
      falseDescriptor =
          getContiguousDescriptor(rewriter, op, falseType, failureReason);
      selectDestDescriptor =
          getContiguousDescriptor(rewriter, op, resultType, failureReason);
      if (mlir::failed(falseDescriptor) || mlir::failed(selectDestDescriptor) ||
          falseDescriptor->byteCount != selectDestDescriptor->byteCount)
        return failPattern(
            rewriter, op, failureReason,
            "target select lowering requires equal contiguous false and "
            "destination payloads");
    }

    llvm::SmallVector<mlir::Value, 3> inputs;
    inputs.reserve(movementPlans.size());
    bool materializedMappedInput = false;
    for (const InputMovementPlan &plan : movementPlans) {
      if (!plan.materializedType) {
        inputs.push_back(plan.source);
        continue;
      }
      mlir::Value materialized =
          rewriter
              .create<mlir::memref::AllocOp>(op.getLoc(), plan.materializedType)
              .getResult();
      createGatherScatterSegments(rewriter, op.getLoc(), plan.source,
                                  materialized, plan.segments);
      inputs.push_back(materialized);
      materializedMappedInput = true;
    }
    mlir::Value dest =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), resultType)
            .getResult();

    if (op.getKind() == ComputeElementwiseKind::Select) {
      createGatherScatter(rewriter, op.getLoc(), inputs[2], dest,
                          *falseDescriptor, *selectDestDescriptor);
      mlir::Value mask =
          rewriter.create<mlir::memref::AllocOp>(op.getLoc(), resultType)
              .getResult();
      rewriter.create<InstrBit2FpOp>(op.getLoc(), inputs[0], mask);
      rewriter.create<InstrMaskMoveOp>(op.getLoc(), inputs[1], mask, dest);
      rewriter.replaceOp(op, dest);
      return mlir::success();
    }

    rewriter.create<InstrElementwiseOp>(
        op.getLoc(), instrKind, inputs, dest,
        getDefaultNCCWorkerAttr(rewriter));
    rewriter.replaceOp(op, dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class ReduceLowering : public mlir::OpRewritePattern<ComputeReduceOp> {
public:
  ReduceLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ComputeReduceOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeReduceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto inputType = mlir::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce lowering requires memref operands");
    if (inputType.getElementType() != resultType.getElementType())
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires matching element types");

    std::optional<int64_t> inputElementCount =
        getStaticPositiveElementCount(inputType.getShape());
    std::optional<int64_t> resultElementCount =
        getStaticPositiveElementCount(resultType.getShape());
    if (!inputElementCount || !resultElementCount)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires static positive input/result shapes");
    if (static_cast<uint64_t>(*resultElementCount) >
        std::numeric_limits<uint32_t>::max())
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering result element count exceeds uint32 target "
          "field");

    mlir::Type elementType = inputType.getElementType();
    unsigned scalarWidth = 0;
    if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
      scalarWidth = integerType.getWidth();
    else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
      scalarWidth = floatType.getWidth();
    else
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires target-encodable integer or float "
          "elements");
    if (scalarWidth > 32)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering element width exceeds uint32 scalar ABI");
    std::optional<WaferPhysicalTensorInfo> inputPhysical =
        wafer::computeWaferPhysicalTensorInfo(inputType);
    std::optional<WaferPhysicalTensorInfo> resultPhysical =
        wafer::computeWaferPhysicalTensorInfo(resultType);
    if (!inputPhysical || !resultPhysical || inputPhysical->bitPackedElement ||
        resultPhysical->bitPackedElement || inputPhysical->elementBytes <= 0 ||
        resultPhysical->elementBytes <= 0)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires byte-addressable elements");

    bool targetEncodableElement =
        (mlir::isa<mlir::IntegerType>(elementType) &&
         (scalarWidth == 8 || scalarWidth == 16 || scalarWidth == 32)) ||
        mlir::isa<mlir::Float16Type, mlir::BFloat16Type, mlir::Float32Type>(
            elementType);
    if (!targetEncodableElement)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce element type is not encodable by the target "
          "data-format ABI");

    mlir::FailureOr<InstrElementwiseKindAttr> accumulationKind =
        getAccumulationElementwiseKind(rewriter, op, op.getKindAttr(),
                                       failureReason, "tile.reduce");
    if (mlir::failed(accumulationKind))
      return mlir::failure();

    mlir::Attribute initValue = op->getAttr("init_value");
    bool hasInitOperand = static_cast<bool>(op.getInit());
    if (hasInitOperand == static_cast<bool>(initValue))
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires exactly one constant init source");
    mlir::TypedAttr typedInit;
    if (initValue) {
      typedInit = mlir::dyn_cast<mlir::TypedAttr>(initValue);
      if (!typedInit || typedInit.getType() != elementType)
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce init_value type must match input element type");
    } else {
      auto constant = op.getInit().getDefiningOp<mlir::arith::ConstantOp>();
      typedInit = constant
                      ? mlir::dyn_cast<mlir::TypedAttr>(constant.getValue())
                      : mlir::TypedAttr{};
      if (!constant || !typedInit || typedInit.getType() != elementType)
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce init operand must be a matching arith.constant");
    }

    auto dimensionsAttr =
        op->getAttrOfType<mlir::DenseI64ArrayAttr>("dimensions");
    if (!dimensionsAttr)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce lowering requires dimensions attr");

    llvm::SmallVector<int64_t, 4> reducedDims(
        dimensionsAttr.asArrayRef().begin(), dimensionsAttr.asArrayRef().end());
    if (reducedDims.empty())
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce lowering requires non-empty dimensions");
    llvm::sort(reducedDims);
    if (std::adjacent_find(reducedDims.begin(), reducedDims.end()) !=
        reducedDims.end())
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce lowering dimensions must be unique");
    for (int64_t dim : reducedDims)
      if (dim < 0 || dim >= inputType.getRank())
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce lowering dimension is outside input rank");

    llvm::SmallVector<int64_t, 4> nonReducedDims;
    llvm::SmallVector<int64_t, 4> reductionShape;
    for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
      if (llvm::is_contained(reducedDims, inputDim))
        reductionShape.push_back(inputType.getDimSize(inputDim));
      else
        nonReducedDims.push_back(inputDim);
    }
    if (static_cast<int64_t>(nonReducedDims.size()) != resultType.getRank())
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce result rank does not match non-reduced dimensions");
    for (auto [resultDim, inputDim] : llvm::enumerate(nonReducedDims))
      if (resultType.getDimSize(resultDim) != inputType.getDimSize(inputDim))
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce result shape does not match non-reduced dimensions");

    std::optional<int64_t> reductionTupleCount =
        getStaticPositiveElementCount(reductionShape);
    if (!reductionTupleCount)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce reduction tuple count overflows or is not positive");
    constexpr uint64_t budget = wafer::detail::kStaticTerminalOperationBudget;
    if (static_cast<uint64_t>(*reductionTupleCount) > (budget - 4) / 4) {
      // The terminal CT reduction encodes a complete logical reduction rather
      // than one scalar tuple at a time.  Select it only when the source init
      // is the exact identity, the logical dimensions have one typed target
      // selector, and both layouts already satisfy that instruction's rank
      // contract.  Small reductions deliberately retain the ordered baseline
      // below so this scale path cannot silently change existing semantics.
      std::optional<int64_t> targetDim;
      for (int64_t candidate = 0; candidate <= 5; ++candidate) {
        llvm::SmallVector<int64_t, 3> candidateDims =
            wafer::getInstrReduceLogicalDims(candidate, inputType.getRank());
        llvm::sort(candidateDims);
        if (candidateDims == reducedDims) {
          targetDim = candidate;
          break;
        }
      }
      bool isPositiveZeroIdentity = false;
      if (auto floatInit = mlir::dyn_cast<mlir::FloatAttr>(typedInit))
        isPositiveZeroIdentity =
            floatInit.getValue().isZero() && !floatInit.getValue().isNegative();
      else if (auto integerInit = mlir::dyn_cast<mlir::IntegerAttr>(typedInit))
        isPositiveZeroIdentity = integerInit.getValue().isZero();

      MemLayout expectedInputLayout =
          inputType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
      MemLayout expectedResultLayout =
          resultType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
      MemoryAttr inputMemory = wafer::getWaferMemoryAttr(inputType);
      MemoryAttr resultMemory = wafer::getWaferMemoryAttr(resultType);
      if (op.getKind() == ComputeReduceKind::Sum &&
          mlir::isa<mlir::Float32Type>(elementType) && isPositiveZeroIdentity &&
          targetDim && inputMemory &&
          inputMemory.getLayout() == expectedInputLayout && resultMemory &&
          resultMemory.getLayout() == expectedResultLayout) {
        mlir::FailureOr<mlir::Value> dest = createDestAlloc(
            op.getLoc(), resultType, rewriter, op, failureReason);
        if (mlir::failed(dest))
          return mlir::failure();
        auto kind = InstrReduceKindAttr::get(rewriter.getContext(),
                                             InstrReduceKind::Sum);
        rewriter.create<InstrReduceOp>(op.getLoc(), kind, op.getInput(), *dest,
                                       getI64Attr(rewriter, *targetDim),
                                       getDefaultNCCWorkerAttr(rewriter));
        rewriter.replaceOp(op, *dest);
        return mlir::success();
      }
      return failPattern(
          rewriter, op, failureReason,
          "static_terminal_budget_exceeded: ordered tile.reduce minimum "
          "terminal operation count exceeds 4096");
    }

    auto tensorType = mlir::MemRefType::get(
        resultType.getShape(), elementType, mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(rewriter.getContext(), MemorySpace::SPM,
                        MemLayout::Tensor));

    struct SlicePlan {
      llvm::SmallVector<LogicalMovementSegment> segments;
    };
    llvm::SmallVector<SlicePlan, 8> slicePlans;
    slicePlans.reserve(static_cast<size_t>(*reductionTupleCount));
    uint64_t terminalOperationCount = 1; // Initial fill.

    for (int64_t linearTuple = 0; linearTuple < *reductionTupleCount;
         ++linearTuple) {
      mlir::FailureOr<llvm::SmallVector<int64_t>> tuple =
          delinearizeIndex(rewriter, op, reductionShape, linearTuple,
                           failureReason, "tile.reduce ordered tuple");
      if (mlir::failed(tuple))
        return mlir::failure();

      auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices,
                               llvm::SmallVectorImpl<int64_t> &sourceIndices) {
        sourceIndices.resize(inputType.getRank(), 0);
        size_t reducedIndex = 0;
        size_t resultIndex = 0;
        for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
          if (llvm::is_contained(reducedDims, inputDim))
            sourceIndices[inputDim] = (*tuple)[reducedIndex++];
          else
            sourceIndices[inputDim] = resultIndices[resultIndex++];
        }
        return mlir::success();
      };
      auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices,
                            llvm::SmallVectorImpl<int64_t> &destIndices) {
        destIndices.assign(resultIndices.begin(), resultIndices.end());
        return mlir::success();
      };

      mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
          getStaticMappedMovementSegments(rewriter, op, inputType, tensorType,
                                          resultType.getShape(), sourceIndexFn,
                                          destIndexFn, failureReason,
                                          "tile.reduce ordered slice movement");
      if (mlir::failed(segments))
        return mlir::failure();
      mlir::FailureOr<uint64_t> commandCount = preflightPackedMovementCommands(
          rewriter, op, *segments, failureReason,
          "tile.reduce ordered slice movement");
      if (mlir::failed(commandCount))
        return mlir::failure();
      if (*commandCount > budget - terminalOperationCount ||
          1 > budget - terminalOperationCount - *commandCount)
        return failPattern(
            rewriter, op, failureReason,
            "static_terminal_budget_exceeded: ordered tile.reduce terminal "
            "operation count exceeds 4096");
      terminalOperationCount += *commandCount + 1;
      slicePlans.push_back({std::move(*segments)});
    }

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> finalSegments =
        getStaticLogicalMovementSegments(rewriter, op, tensorType, resultType,
                                         failureReason,
                                         "tile.reduce final logical movement");
    if (mlir::failed(finalSegments))
      return mlir::failure();
    mlir::FailureOr<uint64_t> finalCommandCount =
        preflightPackedMovementCommands(rewriter, op, *finalSegments,
                                        failureReason,
                                        "tile.reduce final logical movement");
    if (mlir::failed(finalCommandCount))
      return mlir::failure();
    if (*finalCommandCount > budget - terminalOperationCount)
      return failPattern(
          rewriter, op, failureReason,
          "static_terminal_budget_exceeded: ordered tile.reduce terminal "
          "operation count exceeds 4096");

    // All legality, geometry, packing and budget checks above are deliberately
    // completed before creating any effectful instruction.
    mlir::Value init = op.getInit();
    if (!init)
      init = rewriter
                 .create<mlir::arith::ConstantOp>(
                     op.getLoc(), mlir::cast<mlir::TypedAttr>(initValue))
                 .getResult();
    mlir::Value accumulatorA =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), tensorType);
    mlir::Value accumulatorB =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), tensorType);
    mlir::Value slice =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), tensorType);
    mlir::FailureOr<mlir::Value> dest =
        createDestAlloc(op.getLoc(), resultType, rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    rewriter.create<InstrFillOp>(op.getLoc(), accumulatorA, init,
                                 /*fill_domain=*/FillDomainAttr{},
                                 getDefaultNCCWorkerAttr(rewriter));
    mlir::Value currentAccumulator = accumulatorA;
    mlir::Value nextAccumulator = accumulatorB;
    for (const SlicePlan &plan : slicePlans) {
      createGatherScatterSegments(rewriter, op.getLoc(), op.getInput(), slice,
                                  plan.segments);
      llvm::SmallVector<mlir::Value, 2> inputs{currentAccumulator, slice};
      rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                          inputs, nextAccumulator,
                                          getDefaultNCCWorkerAttr(rewriter));
      std::swap(currentAccumulator, nextAccumulator);
    }
    createGatherScatterSegments(rewriter, op.getLoc(), currentAccumulator,
                                *dest, *finalSegments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class GemmLowering : public mlir::OpRewritePattern<ComputeGemmOp> {
public:
  GemmLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ComputeGemmOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeGemmOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<int64_t, 3>> mkn =
        inferGemmMKN(op, rewriter, failureReason);
    if (mlir::failed(mkn))
      return mlir::failure();

    auto instr = rewriter.create<InstrGemmOp>(
        op.getLoc(), op.getLhs(), op.getRhs(), *dest,
        getI64Attr(rewriter, (*mkn)[0]), getI64Attr(rewriter, (*mkn)[1]),
        getI64Attr(rewriter, (*mkn)[2]), op.getLhsOrientationAttr(),
        op.getRhsOrientationAttr(),
        /*batch_count=*/mlir::IntegerAttr{},
        /*lhs_batch_dims=*/mlir::DenseI64ArrayAttr{},
        /*lhs_m_dim=*/mlir::IntegerAttr{},
        /*lhs_contracting_dim=*/mlir::IntegerAttr{},
        /*rhs_batch_dims=*/mlir::DenseI64ArrayAttr{},
        /*rhs_contracting_dim=*/mlir::IntegerAttr{},
        /*rhs_n_dim=*/mlir::IntegerAttr{},
        /*result_batch_dims=*/mlir::DenseI64ArrayAttr{},
        /*result_m_dim=*/mlir::IntegerAttr{},
        /*result_n_dim=*/mlir::IntegerAttr{},
        getDefaultNCCWorkerAttr(rewriter));
    copyOptionalAttr(op, instr, "batch_count");
    copyOptionalAttr(op, instr, "lhs_batch_dims");
    copyOptionalAttr(op, instr, "lhs_m_dim");
    copyOptionalAttr(op, instr, "lhs_contracting_dim");
    copyOptionalAttr(op, instr, "rhs_batch_dims");
    copyOptionalAttr(op, instr, "rhs_contracting_dim");
    copyOptionalAttr(op, instr, "rhs_n_dim");
    copyOptionalAttr(op, instr, "result_batch_dims");
    copyOptionalAttr(op, instr, "result_m_dim");
    copyOptionalAttr(op, instr, "result_n_dim");
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

static mlir::FailureOr<InstrElementwiseKindAttr> getInstrElementwiseKindAttr(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeElementwiseKindAttr computeKind, std::string *failureReason) {
  InstrElementwiseKind instrKind;
  switch (computeKind.getValue()) {
  case ComputeElementwiseKind::Add:
    instrKind = InstrElementwiseKind::Add;
    break;
  case ComputeElementwiseKind::Sub:
    instrKind = InstrElementwiseKind::Sub;
    break;
  case ComputeElementwiseKind::Mul:
    instrKind = InstrElementwiseKind::Mul;
    break;
  case ComputeElementwiseKind::Div:
    instrKind = InstrElementwiseKind::Div;
    break;
  case ComputeElementwiseKind::Max:
    instrKind = InstrElementwiseKind::Max;
    break;
  case ComputeElementwiseKind::Min:
    instrKind = InstrElementwiseKind::Min;
    break;
  case ComputeElementwiseKind::Neg:
    instrKind = InstrElementwiseKind::Neg;
    break;
  case ComputeElementwiseKind::Recip:
    instrKind = InstrElementwiseKind::Recip;
    break;
  case ComputeElementwiseKind::Sqrt:
    instrKind = InstrElementwiseKind::Sqrt;
    break;
  case ComputeElementwiseKind::Rsqrt:
    instrKind = InstrElementwiseKind::Rsqrt;
    break;
  case ComputeElementwiseKind::Exp:
    instrKind = InstrElementwiseKind::Exp;
    break;
  case ComputeElementwiseKind::Tanh:
    instrKind = InstrElementwiseKind::Tanh;
    break;
  case ComputeElementwiseKind::Eq:
    instrKind = InstrElementwiseKind::Eq;
    break;
  case ComputeElementwiseKind::Ne:
    instrKind = InstrElementwiseKind::Ne;
    break;
  case ComputeElementwiseKind::Lt:
    instrKind = InstrElementwiseKind::Lt;
    break;
  case ComputeElementwiseKind::Le:
    instrKind = InstrElementwiseKind::Le;
    break;
  case ComputeElementwiseKind::Gt:
    instrKind = InstrElementwiseKind::Gt;
    break;
  case ComputeElementwiseKind::Ge:
    instrKind = InstrElementwiseKind::Ge;
    break;
  case ComputeElementwiseKind::Select:
    return failFailureOr<InstrElementwiseKindAttr>(
        rewriter, op, failureReason,
        "tile.elementwise select must lower to target movement sequence before "
        "instruction elementwise");
  }
  return InstrElementwiseKindAttr::get(rewriter.getContext(), instrKind);
}

} // namespace

mlir::FailureOr<InstrElementwiseKindAttr>
wafer::tile_region_to_instr::getAccumulationElementwiseKind(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeReduceKindAttr reduceKind, std::string *failureReason,
    llvm::StringRef opLabel) {
  InstrElementwiseKind elementwiseKind;
  switch (reduceKind.getValue()) {
  case ComputeReduceKind::Sum:
    elementwiseKind = InstrElementwiseKind::Add;
    break;
  case ComputeReduceKind::Max:
    elementwiseKind = InstrElementwiseKind::Max;
    break;
  case ComputeReduceKind::Min:
    elementwiseKind = InstrElementwiseKind::Min;
    break;
  case ComputeReduceKind::Avg:
    return failFailureOr<InstrElementwiseKindAttr>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" lowering does not support avg accumulation")
            .str());
  }

  return InstrElementwiseKindAttr::get(rewriter.getContext(), elementwiseKind);
}

void wafer::tile_region_to_instr::populateComputeLoweringPatterns(
    mlir::RewritePatternSet &patterns, std::string *failureReason) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns
      .add<ConvertLowering, ElementwiseLowering, ReduceLowering, GemmLowering>(
          context, failureReason);
}

void wafer::tile_region_to_instr::populateFillLoweringPattern(
    mlir::RewritePatternSet &patterns) {
  patterns.add<FillLowering>(patterns.getContext());
}

void wafer::tile_region_to_instr::
    populateConstantPredicateSelectCanonicalizationPattern(
        mlir::RewritePatternSet &patterns) {
  patterns.add<ConstantPredicateSelectToCopy>(patterns.getContext());
}
