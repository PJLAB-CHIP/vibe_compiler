//===- ComputeLowering.cpp - Tile-region compute lowering --------------===//

#include "Internal.h"
#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"
#include "Wafer/Target/Core/TargetCall.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

static mlir::FailureOr<int64_t>
getStaticDim(mlir::PatternRewriter &rewriter, mlir::Operation *op,
             mlir::RankedTensorType type, int64_t dim, llvm::StringRef role) {
  int64_t value = type.getDimSize(dim);
  if (value == mlir::ShapedType::kDynamic)
    return failFailureOr<int64_t>(
        rewriter, op,
        llvm::Twine(role).concat(" requires static GEMM dimensions").str());
  return value;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 3>>
inferGemmMKN(ComputeGemmOp op, mlir::PatternRewriter &rewriter) {
  std::optional<mlir::RankedTensorType> lhs =
      getLogicalTensorTypeFromMemRef(op.getLhs().getType());
  std::optional<mlir::RankedTensorType> rhs =
      getLogicalTensorTypeFromMemRef(op.getRhs().getType());
  std::optional<mlir::RankedTensorType> result =
      getLogicalTensorTypeFromMemRef(op.getResult().getType());
  if (!lhs || !rhs || !result)
    return failFailureOr<llvm::SmallVector<int64_t, 3>>(
        rewriter, op, "tile.gemm lowering requires Wafer memref operands");

  if (lhs->getRank() != 2 || rhs->getRank() != 2 || result->getRank() != 2) {
    mlir::IntegerAttr lhsMDimAttr = op.getLhsMDimAttr();
    mlir::IntegerAttr lhsKDimAttr = op.getLhsContractingDimAttr();
    mlir::IntegerAttr rhsNDimAttr = op.getRhsNDimAttr();
    if (!lhsMDimAttr || !lhsKDimAttr || !rhsNDimAttr)
      return failFailureOr<llvm::SmallVector<int64_t, 3>>(
          rewriter, op,
          "batched tile.gemm requires typed dimension attributes");
    int64_t lhsMDim = lhsMDimAttr.getInt();
    int64_t lhsKDim = lhsKDimAttr.getInt();
    int64_t rhsNDim = rhsNDimAttr.getInt();
    mlir::FailureOr<int64_t> m =
        getStaticDim(rewriter, op, *lhs, lhsMDim, "batched tile.gemm");
    mlir::FailureOr<int64_t> k =
        getStaticDim(rewriter, op, *lhs, lhsKDim, "batched tile.gemm");
    mlir::FailureOr<int64_t> n =
        getStaticDim(rewriter, op, *rhs, rhsNDim, "batched tile.gemm");
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
  mlir::FailureOr<int64_t> m =
      getStaticDim(rewriter, op, *lhs, lhsMDim, "rank-2 tile.gemm");
  mlir::FailureOr<int64_t> k =
      getStaticDim(rewriter, op, *lhs, lhsKDim, "rank-2 tile.gemm");
  mlir::FailureOr<int64_t> n =
      getStaticDim(rewriter, op, *rhs, rhsNDim, "rank-2 tile.gemm");
  if (mlir::failed(m) || mlir::failed(k) || mlir::failed(n))
    return mlir::failure();
  return llvm::SmallVector<int64_t, 3>{*m, *k, *n};
}

static mlir::FailureOr<InstrElementwiseKindAttr>
getInstrElementwiseKindAttr(mlir::PatternRewriter &rewriter,
                            mlir::Operation *op,
                            ComputeElementwiseKindAttr computeKind);

static mlir::LogicalResult
proveIdentityPhysicalTraversal(mlir::PatternRewriter &rewriter,
                               mlir::Operation *op, mlir::MemRefType sourceType,
                               mlir::MemRefType destType,
                               llvm::StringRef subject) {
  if (sourceType.getShape() != destType.getShape())
    return failPattern(rewriter, op,
                       (subject + " requires equal logical shapes").str());
  analysis::IndexRelationResult identity =
      analysis::IndexRelation::identity(destType.getShape());
  if (!identity.isExact() ||
      mlir::failed(analysis::TransferRealizability::provePhysicalTraversal(
          sourceType, destType, destType.getShape(), *identity.get(),
          *identity.get())))
    return failPattern(
        rewriter, op,
        (subject + " has incompatible physical element traversal").str());
  return mlir::success();
}

#include "WaferTileRegionToInstr/Patterns.inc"

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
  ConvertLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ComputeConvertOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeConvertOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op,
                         "tile.compute.convert requires memref storage");
    if (mlir::failed(proveIdentityPhysicalTraversal(
            rewriter, op, sourceType, resultType, "tile.compute.convert")))
      return mlir::failure();
    std::optional<InstrConvertKind> kind = resolveInstrConvertKind(
        sourceType.getElementType(), resultType.getElementType());
    if (!kind)
      return failPattern(
          rewriter, op, "tile.compute.convert has no target instruction route");

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
      return failPattern(rewriter, op,
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
};

struct ConstantPredicateSelectPlan {
  mlir::memref::AllocOp predicateAllocation;
  ComputeFillOp sourcePredicateFill;
  InstrFillOp loweredPredicateFill;
  mlir::arith::ConstantOp predicateConstant;
  mlir::Value selectedInput;
};

// Return a direct-copy plan when a select is controlled by a private constant
// predicate. Dialect conversion may visit the producer fill before the select
// root, so accept both the source ComputeFillOp and its legal InstrFillOp
// replacement. The caller emits the complete replacement from the select's
// own conversion pattern, making the result independent of worklist order.
static std::optional<ConstantPredicateSelectPlan>
getConstantPredicateSelectPlan(ComputeElementwiseOp op) {
  if (op.getKind() != ComputeElementwiseKind::Select ||
      op.getInputs().size() != 3)
    return std::nullopt;

  mlir::Value predicate = op.getInputs().front();
  auto predicateAlloc = predicate.getDefiningOp<mlir::memref::AllocOp>();
  if (!predicateAlloc)
    return std::nullopt;

  mlir::OpOperand *selectPredicateUse = &op->getOpOperand(0);
  ComputeFillOp sourcePredicateFill;
  InstrFillOp loweredPredicateFill;
  mlir::Value predicateValue;
  unsigned selectUseCount = 0;
  for (mlir::OpOperand &use : predicate.getUses()) {
    if (&use == selectPredicateUse) {
      ++selectUseCount;
      continue;
    }
    mlir::Value fillValue;
    if (auto fill = mlir::dyn_cast<ComputeFillOp>(use.getOwner())) {
      if (&use != &fill.getDestMutable() || sourcePredicateFill)
        return std::nullopt;
      sourcePredicateFill = fill;
      fillValue = fill.getValue();
    } else if (auto fill = mlir::dyn_cast<InstrFillOp>(use.getOwner())) {
      if (&use != &fill.getDestMutable() || loweredPredicateFill)
        return std::nullopt;
      loweredPredicateFill = fill;
      fillValue = fill.getValue();
    } else {
      return std::nullopt;
    }
    if (predicateValue && predicateValue != fillValue)
      return std::nullopt;
    predicateValue = fillValue;
  }
  if (selectUseCount != 1 || (!sourcePredicateFill && !loweredPredicateFill))
    return std::nullopt;

  for (mlir::Operation *predicateFill : {sourcePredicateFill.getOperation(),
                                         loweredPredicateFill.getOperation()}) {
    if (!predicateFill)
      continue;
    if (predicateFill->getBlock() != op->getBlock() ||
        !predicateFill->isBeforeInBlock(op))
      return std::nullopt;
  }

  auto constant = predicateValue.getDefiningOp<mlir::arith::ConstantOp>();
  auto valueAttr = constant
                       ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                       : mlir::IntegerAttr{};
  if (!constant || !constant.getType().isInteger(1) || !valueAttr ||
      !valueAttr.getType().isInteger(1))
    return std::nullopt;

  unsigned selectedInputIndex = valueAttr.getValue().isZero() ? 2 : 1;
  mlir::Value selected = op.getInputs()[selectedInputIndex];
  auto resultType = mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
  auto selectedType = mlir::dyn_cast<mlir::MemRefType>(selected.getType());
  if (!resultType || !selectedType || selectedType != resultType)
    return std::nullopt;

  if (mlir::ArrayAttr maps = op.getIndexingMapsAttr()) {
    if (maps.size() != op.getInputs().size() + 1)
      return std::nullopt;
    auto hasIdentityMap = [&](unsigned mapIndex) {
      auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(maps[mapIndex]);
      if (!mapAttr)
        return false;
      mlir::AffineMap map = mapAttr.getValue();
      return map.getNumDims() == resultType.getRank() &&
             map.getNumSymbols() == 0 &&
             map.getNumResults() == resultType.getRank() && map.isIdentity();
    };
    if (!hasIdentityMap(selectedInputIndex) || !hasIdentityMap(maps.size() - 1))
      return std::nullopt;
  }
  return ConstantPredicateSelectPlan{predicateAlloc, sourcePredicateFill,
                                     loweredPredicateFill, constant, selected};
}

class ElementwiseLowering
    : public mlir::OpRewritePattern<ComputeElementwiseOp> {
public:
  struct InputMovementPlan {
    mlir::Value source;
    mlir::MemRefType materializedType;
    llvm::SmallVector<MovementDescriptorPair> descriptors;
  };

  ElementwiseLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ComputeElementwiseOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeElementwiseOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    if (op.getKind() == ComputeElementwiseKind::Select) {
      if (op.getInputs().size() != 3)
        return failPattern(
            rewriter, op,
            "target select lowering requires predicate, true and false "
            "operands");
      auto destType =
          mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
      if (!destType || !mlir::isa<mlir::FloatType>(destType.getElementType()))
        return failPattern(
            rewriter, op,
            "target select lowering currently requires floating-point values");
    }

    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType)
      return failPattern(rewriter, op,
                         "tile.elementwise lowering requires a memref result");

    if (std::optional<ConstantPredicateSelectPlan> constantSelect =
            getConstantPredicateSelectPlan(op)) {
      analysis::IndexRelationResult relation =
          analysis::IndexRelation::identity(resultType.getShape());
      if (!relation.isExact())
        return failPattern(rewriter, op,
                           "constant select copy relation is not exact");
      auto selectedType =
          mlir::cast<mlir::MemRefType>(constantSelect->selectedInput.getType());
      mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
          getRelationMovementDescriptors(
              rewriter, op, selectedType, resultType, resultType.getShape(),
              *relation.get(), *relation.get(), MovementEngine::GatherScatter,
              "constant select copy");
      if (mlir::failed(descriptors))
        return mlir::failure();

      mlir::Value dest =
          rewriter.create<mlir::memref::AllocOp>(op.getLoc(), resultType)
              .getResult();
      createGatherScatterDescriptors(rewriter, op.getLoc(),
                                     constantSelect->selectedInput, dest,
                                     *descriptors);
      rewriter.replaceOp(op, dest);
      // In rollback-enabled dialect conversion, a source fill and its legal
      // replacement can temporarily coexist. Erase the replacement and let
      // the conversion driver retire its already-scheduled source root. If the
      // fill has not been legalized yet, erase the source root directly.
      if (constantSelect->loweredPredicateFill)
        rewriter.eraseOp(constantSelect->loweredPredicateFill);
      else
        rewriter.eraseOp(constantSelect->sourcePredicateFill);
      rewriter.eraseOp(constantSelect->predicateAllocation);
      if (constantSelect->predicateConstant->use_empty())
        rewriter.eraseOp(constantSelect->predicateConstant);
      return mlir::success();
    }

    // Validate every map before creating an allocation or an instruction.
    // A failed conversion therefore cannot leave a partially materialized
    // operand sequence in the pattern rewriter.
    llvm::SmallVector<InputMovementPlan, 3> movementPlans;
    movementPlans.reserve(op.getInputs().size());
    mlir::ArrayAttr indexingMaps = op.getIndexingMapsAttr();
    if (indexingMaps) {
      if (indexingMaps.size() != op.getInputs().size() + 1)
        return failPattern(
            rewriter, op,
            "tile.elementwise indexing map count must match inputs plus "
            "result");

      auto resultMapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(
          indexingMaps[indexingMaps.size() - 1]);
      if (!resultMapAttr ||
          resultMapAttr.getValue().getNumDims() != resultType.getRank() ||
          resultMapAttr.getValue().getNumSymbols() != 0 ||
          !resultMapAttr.getValue().isIdentity())
        return failPattern(
            rewriter, op,
            "tile.elementwise result indexing map must be identity");
    }

    for (auto [index, input] : llvm::enumerate(op.getInputs())) {
      InputMovementPlan plan;
      plan.source = input;
      auto sourceType = mlir::dyn_cast<mlir::MemRefType>(input.getType());
      if (!sourceType)
        return failPattern(rewriter, op,
                           "tile.elementwise requires memref inputs");
      if (!indexingMaps) {
        if (mlir::failed(proveIdentityPhysicalTraversal(
                rewriter, op, sourceType, resultType,
                "map-free tile.elementwise")))
          return mlir::failure();
        movementPlans.push_back(std::move(plan));
        continue;
      }

      auto inputMapAttr =
          mlir::dyn_cast<mlir::AffineMapAttr>(indexingMaps[index]);
      if (!inputMapAttr)
        return failPattern(
            rewriter, op,
            "tile.elementwise indexing map materialization requires memref "
            "inputs and affine maps");

      mlir::AffineMap inputMap = inputMapAttr.getValue();
      if (inputMap.getNumDims() != resultType.getRank() ||
          inputMap.getNumSymbols() != 0 ||
          inputMap.getNumResults() != sourceType.getRank() ||
          !inputMap.isProjectedPermutation())
        return failPattern(
            rewriter, op,
            "tile.elementwise input indexing map must be a projected "
            "permutation of result dimensions");

      if (inputMap.isIdentity() &&
          sourceType.getShape() == resultType.getShape()) {
        analysis::IndexRelationResult identity =
            analysis::IndexRelation::identity(resultType.getShape());
        if (identity.isExact() &&
            mlir::succeeded(
                analysis::TransferRealizability::provePhysicalTraversal(
                    sourceType, resultType, resultType.getShape(),
                    *identity.get(), *identity.get()))) {
          movementPlans.push_back(std::move(plan));
          continue;
        }
      }

      plan.materializedType = mlir::MemRefType::get(
          resultType.getShape(), sourceType.getElementType(),
          resultType.getLayout(), resultType.getMemorySpace());
      analysis::IndexRelationResult sourceRelation =
          analysis::IndexRelation::fromAffineMap(
              inputMap, resultType.getShape(), sourceType.getShape());
      analysis::IndexRelationResult destRelation =
          analysis::IndexRelation::identity(resultType.getShape());
      if (!sourceRelation.isExact() || !destRelation.isExact())
        return failPattern(rewriter, op,
                           "tile.elementwise indexing relation is not exact");
      mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
          getRelationMovementDescriptors(
              rewriter, op, sourceType, plan.materializedType,
              resultType.getShape(), *sourceRelation.get(), *destRelation.get(),
              MovementEngine::GatherScatter,
              "tile.elementwise indexing map materialization");
      if (mlir::failed(descriptors))
        return mlir::failure();
      plan.descriptors = std::move(*descriptors);
      if (mlir::failed(proveIdentityPhysicalTraversal(
              rewriter, op, plan.materializedType, resultType,
              "materialized tile.elementwise operand")))
        return mlir::failure();
      movementPlans.push_back(std::move(plan));
    }

    InstrElementwiseKindAttr instrKind;
    if (op.getKind() != ComputeElementwiseKind::Select) {
      mlir::FailureOr<InstrElementwiseKindAttr> resolvedInstrKind =
          getInstrElementwiseKindAttr(rewriter, op, op.getKindAttr());
      if (mlir::failed(resolvedInstrKind))
        return mlir::failure();
      instrKind = *resolvedInstrKind;
    }

    // Select has a movement-based target sequence. Validate its copy
    // descriptors as well, still before emitting any effect.
    llvm::SmallVector<MovementDescriptorPair> selectCopyDescriptors;
    if (op.getKind() == ComputeElementwiseKind::Select) {
      mlir::Type falseType = movementPlans[2].source.getType();
      if (movementPlans[2].materializedType)
        falseType = movementPlans[2].materializedType;
      auto falseMemRef = mlir::dyn_cast<mlir::MemRefType>(falseType);
      analysis::IndexRelationResult relation =
          analysis::IndexRelation::identity(resultType.getShape());
      if (!falseMemRef || !relation.isExact())
        return failPattern(rewriter, op,
                           "target select copy relation is not exact");
      mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
          getRelationMovementDescriptors(
              rewriter, op, falseMemRef, resultType, resultType.getShape(),
              *relation.get(), *relation.get(), MovementEngine::GatherScatter,
              "target select false-value copy");
      if (mlir::failed(descriptors))
        return mlir::failure();
      selectCopyDescriptors = std::move(*descriptors);
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
      createGatherScatterDescriptors(rewriter, op.getLoc(), plan.source,
                                     materialized, plan.descriptors);
      inputs.push_back(materialized);
      materializedMappedInput = true;
    }
    mlir::Value dest =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), resultType)
            .getResult();

    if (op.getKind() == ComputeElementwiseKind::Select) {
      createGatherScatterDescriptors(rewriter, op.getLoc(), inputs[2], dest,
                                     selectCopyDescriptors);
      mlir::Value mask =
          rewriter.create<mlir::memref::AllocOp>(op.getLoc(), resultType)
              .getResult();
      rewriter.create<InstrBit2FpOp>(op.getLoc(), inputs[0], mask);
      rewriter.create<InstrMaskMoveOp>(op.getLoc(), inputs[1], mask, dest);
      rewriter.replaceOp(op, dest);
      return mlir::success();
    }

    rewriter.create<InstrElementwiseOp>(op.getLoc(), instrKind, inputs, dest,
                                        getDefaultNCCWorkerAttr(rewriter));
    rewriter.replaceOp(op, dest);
    return mlir::success();
  }
};

class ElementwiseIntoLowering
    : public mlir::OpRewritePattern<ComputeElementwiseIntoOp> {
public:
  ElementwiseIntoLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ComputeElementwiseIntoOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeElementwiseIntoOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    mlir::FailureOr<InstrElementwiseKindAttr> instrKind =
        getInstrElementwiseKindAttr(rewriter, op, op.getKindAttr());
    if (mlir::failed(instrKind))
      return mlir::failure();
    rewriter.create<InstrElementwiseOp>(
        op.getLoc(), *instrKind, op.getInputs(), op.getDest(),
        getDefaultNCCWorkerAttr(rewriter));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ReduceLowering : public mlir::OpRewritePattern<ComputeReduceOp> {
public:
  ReduceLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ComputeReduceOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeReduceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto inputType = mlir::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failPattern(rewriter, op,
                         "tile.reduce lowering requires memref operands");
    if (inputType.getElementType() != resultType.getElementType())
      return failPattern(
          rewriter, op, "tile.reduce lowering requires matching element types");

    std::optional<int64_t> inputElementCount =
        getStaticPositiveElementCount(inputType.getShape());
    std::optional<int64_t> resultElementCount =
        getStaticPositiveElementCount(resultType.getShape());
    if (!inputElementCount || !resultElementCount)
      return failPattern(
          rewriter, op,
          "tile.reduce lowering requires static positive input/result shapes");
    if (static_cast<uint64_t>(*resultElementCount) >
        std::numeric_limits<uint32_t>::max())
      return failPattern(
          rewriter, op,
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
          rewriter, op,
          "tile.reduce lowering requires target-encodable integer or float "
          "elements");
    if (scalarWidth > 32)
      return failPattern(
          rewriter, op,
          "tile.reduce lowering element width exceeds uint32 scalar ABI");
    std::optional<WaferPhysicalTensorInfo> inputPhysical =
        wafer::computeWaferPhysicalTensorInfo(inputType);
    std::optional<WaferPhysicalTensorInfo> resultPhysical =
        wafer::computeWaferPhysicalTensorInfo(resultType);
    if (!inputPhysical || !resultPhysical || inputPhysical->bitPackedElement ||
        resultPhysical->bitPackedElement || inputPhysical->elementBytes <= 0 ||
        resultPhysical->elementBytes <= 0)
      return failPattern(
          rewriter, op,
          "tile.reduce lowering requires byte-addressable elements");

    bool targetEncodableElement =
        (mlir::isa<mlir::IntegerType>(elementType) &&
         (scalarWidth == 8 || scalarWidth == 16 || scalarWidth == 32)) ||
        mlir::isa<mlir::Float16Type, mlir::BFloat16Type, mlir::Float32Type>(
            elementType);
    if (!targetEncodableElement)
      return failPattern(
          rewriter, op,
          "tile.reduce element type is not encodable by the target "
          "data-format ABI");

    mlir::FailureOr<InstrElementwiseKindAttr> accumulationKind =
        getAccumulationElementwiseKind(rewriter, op, op.getKindAttr(),
                                       "tile.reduce");
    if (mlir::failed(accumulationKind))
      return mlir::failure();

    mlir::TypedAttr initValue = op.getInitValueAttr();
    bool hasInitOperand = static_cast<bool>(op.getInit());
    if (hasInitOperand == static_cast<bool>(initValue))
      return failPattern(
          rewriter, op,
          "tile.reduce lowering requires exactly one constant init source");
    mlir::TypedAttr typedInit;
    if (initValue) {
      typedInit = initValue;
      if (typedInit.getType() != elementType)
        return failPattern(
            rewriter, op,
            "tile.reduce init_value type must match input element type");
    } else {
      auto constant = op.getInit().getDefiningOp<mlir::arith::ConstantOp>();
      typedInit = constant
                      ? mlir::dyn_cast<mlir::TypedAttr>(constant.getValue())
                      : mlir::TypedAttr{};
      if (!constant || !typedInit || typedInit.getType() != elementType)
        return failPattern(
            rewriter, op,
            "tile.reduce init operand must be a matching arith.constant");
    }

    mlir::DenseI64ArrayAttr dimensionsAttr = op.getDimensionsAttr();

    llvm::SmallVector<int64_t, 4> reducedDims(
        dimensionsAttr.asArrayRef().begin(), dimensionsAttr.asArrayRef().end());
    if (reducedDims.empty())
      return failPattern(rewriter, op,
                         "tile.reduce lowering requires non-empty dimensions");
    llvm::sort(reducedDims);
    if (std::adjacent_find(reducedDims.begin(), reducedDims.end()) !=
        reducedDims.end())
      return failPattern(rewriter, op,
                         "tile.reduce lowering dimensions must be unique");
    for (int64_t dim : reducedDims)
      if (dim < 0 || dim >= inputType.getRank())
        return failPattern(
            rewriter, op,
            "tile.reduce lowering dimension is outside input rank");

    llvm::SmallVector<int64_t, 4> nonReducedDims;
    llvm::SmallVector<int64_t, 4> reductionShape;
    for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
      if (llvm::is_contained(reducedDims, inputDim))
        reductionShape.push_back(inputType.getDimSize(inputDim));
      else
        nonReducedDims.push_back(inputDim);
    }
    // The complete-reduction boundary keeps the source rank with extent one
    // on every reduced dimension (constant-position output map); the
    // canonical form drops the reduced dimensions entirely.
    const bool extentOneBoundary =
        resultType.getRank() == inputType.getRank() &&
        llvm::all_of(reducedDims, [&](int64_t dim) {
          return resultType.getDimSize(dim) == 1;
        });
    if (!extentOneBoundary &&
        static_cast<int64_t>(nonReducedDims.size()) != resultType.getRank())
      return failPattern(
          rewriter, op,
          "tile.reduce result rank does not match non-reduced dimensions");
    for (auto [resultDim, inputDim] : llvm::enumerate(nonReducedDims))
      if (resultType.getDimSize(extentOneBoundary ? inputDim : resultDim) !=
          inputType.getDimSize(inputDim))
        return failPattern(
            rewriter, op,
            "tile.reduce result shape does not match non-reduced dimensions");

    std::optional<int64_t> reductionTupleCount =
        getStaticPositiveElementCount(reductionShape);
    if (!reductionTupleCount)
      return failPattern(
          rewriter, op,
          "tile.reduce reduction tuple count overflows or is not positive");
    // This is a profitability/materialization preference, not a legality
    // limit. The exact final instruction count is fed to schedule cost; a
    // larger ordered program remains representable when native reduction
    // cannot preserve the source identity or target contract.
    constexpr uint64_t preferredMaximumOrderedReductionOperations = 4096;
    uint64_t tupleCount = static_cast<uint64_t>(*reductionTupleCount);
    bool preferNativeReduction =
        resultType.getRank() != 0 &&
        tupleCount > (preferredMaximumOrderedReductionOperations - 4) / 4;
    if (preferNativeReduction) {
      // The native CT reduction encodes a complete logical reduction rather
      // than one scalar tuple at a time. Select it only when the source init
      // is the exact identity, the logical dimensions have one typed target
      // selector, and both layouts already satisfy that instruction's rank
      // contract.
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
      std::optional<InstrReduceKind> nativeKind;
      bool hasExactNativeIdentity = false;
      if (auto floatInit = mlir::dyn_cast<mlir::FloatAttr>(typedInit)) {
        const llvm::APFloat &value = floatInit.getValue();
        switch (op.getKind()) {
        case ComputeReduceKind::Sum:
          nativeKind = InstrReduceKind::Sum;
          hasExactNativeIdentity = value.isZero() && !value.isNegative();
          break;
        case ComputeReduceKind::Max:
          nativeKind = InstrReduceKind::Max;
          hasExactNativeIdentity = value.isInfinity() && value.isNegative();
          break;
        case ComputeReduceKind::Min:
          nativeKind = InstrReduceKind::Min;
          hasExactNativeIdentity = value.isInfinity() && !value.isNegative();
          break;
        case ComputeReduceKind::Avg:
          break;
        }
      } else if (auto integerInit =
                     mlir::dyn_cast<mlir::IntegerAttr>(typedInit)) {
        const llvm::APInt &value = integerInit.getValue();
        switch (op.getKind()) {
        case ComputeReduceKind::Sum:
          nativeKind = InstrReduceKind::Sum;
          hasExactNativeIdentity = value.isZero();
          break;
        case ComputeReduceKind::Max:
          nativeKind = InstrReduceKind::Max;
          hasExactNativeIdentity =
              value == llvm::APInt::getSignedMinValue(value.getBitWidth());
          break;
        case ComputeReduceKind::Min:
          nativeKind = InstrReduceKind::Min;
          hasExactNativeIdentity =
              value == llvm::APInt::getSignedMaxValue(value.getBitWidth());
          break;
        case ComputeReduceKind::Avg:
          break;
        }
      }

      MemLayout expectedInputLayout =
          inputType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
      MemLayout expectedResultLayout =
          resultType.getRank() == 0
              ? MemLayout::Tensor
              : (resultType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx);
      MemoryAttr inputMemory = wafer::getWaferMemoryAttr(inputType);
      MemoryAttr resultMemory = wafer::getWaferMemoryAttr(resultType);
      std::optional<LogicalFormat> logicalFormat;
      if (mlir::isa<mlir::Float16Type>(elementType))
        logicalFormat = LogicalFormat::F16;
      else if (mlir::isa<mlir::BFloat16Type>(elementType))
        logicalFormat = LogicalFormat::BF16;
      else if (mlir::isa<mlir::Float32Type>(elementType))
        logicalFormat = LogicalFormat::F32;
      const TargetFormatEncodingRecord *reduceFormat =
          logicalFormat
              ? findTargetFormatEncoding(TargetFormatEngine::CT, *logicalFormat)
              : nullptr;
      const bool targetAllowsNativeReduce = reduceFormat != nullptr;
      if (nativeKind && hasExactNativeIdentity && targetDim &&
          targetAllowsNativeReduce && inputMemory &&
          inputMemory.getLayout() == expectedInputLayout && resultMemory &&
          resultMemory.getLayout() == expectedResultLayout) {
        mlir::FailureOr<mlir::Value> dest =
            createDestAlloc(op.getLoc(), resultType, rewriter, op);
        if (mlir::failed(dest))
          return mlir::failure();
        auto kind =
            InstrReduceKindAttr::get(rewriter.getContext(), *nativeKind);
        rewriter.create<InstrReduceOp>(op.getLoc(), kind, op.getInput(), *dest,
                                       getI64Attr(rewriter, *targetDim),
                                       getDefaultNCCWorkerAttr(rewriter));
        rewriter.replaceOp(op, *dest);
        return mlir::success();
      }
    }

    auto tensorType = mlir::MemRefType::get(
        resultType.getShape(), elementType, mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(rewriter.getContext(), MemorySpace::SPM,
                        MemLayout::Tensor));

    struct SlicePlan {
      llvm::SmallVector<MovementDescriptorPair> descriptors;
    };
    llvm::SmallVector<SlicePlan, 8> slicePlans;
    slicePlans.reserve(static_cast<size_t>(*reductionTupleCount));
    for (int64_t linearTuple = 0; linearTuple < *reductionTupleCount;
         ++linearTuple) {
      mlir::FailureOr<llvm::SmallVector<int64_t>> tuple =
          delinearizeIndex(rewriter, op, reductionShape, linearTuple,
                           "tile.reduce ordered tuple");
      if (mlir::failed(tuple))
        return mlir::failure();
      SlicePlan plan;
      llvm::SmallVector<mlir::AffineExpr, 4> sourceResults;
      sourceResults.reserve(inputType.getRank());
      size_t reducedIndex = 0;
      unsigned resultIndex = 0;
      for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
        if (llvm::is_contained(reducedDims, inputDim)) {
          sourceResults.push_back(mlir::getAffineConstantExpr(
              (*tuple)[reducedIndex++], rewriter.getContext()));
        } else {
          sourceResults.push_back(
              mlir::getAffineDimExpr(resultIndex++, rewriter.getContext()));
        }
      }
      analysis::IndexRelationResult sourceRelation =
          analysis::IndexRelation::fromAffineMap(
              mlir::AffineMap::get(resultType.getRank(), 0, sourceResults,
                                   rewriter.getContext()),
              resultType.getShape(), inputType.getShape());
      analysis::IndexRelationResult destRelation =
          analysis::IndexRelation::identity(resultType.getShape());
      if (!sourceRelation.isExact() || !destRelation.isExact())
        return failPattern(rewriter, op,
                           "tile.reduce slice relation is not exact");
      mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
          relationDescriptors = getRelationMovementDescriptors(
              rewriter, op, inputType, tensorType, resultType.getShape(),
              *sourceRelation.get(), *destRelation.get(),
              MovementEngine::GatherScatter,
              "tile.reduce ordered slice movement");
      if (mlir::failed(relationDescriptors))
        return mlir::failure();
      plan.descriptors = std::move(*relationDescriptors);
      slicePlans.push_back(std::move(plan));
    }

    analysis::IndexRelationResult finalRelation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!finalRelation.isExact())
      return failPattern(rewriter, op,
                         "tile.reduce final relation is not exact");
    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
        finalDescriptors = getRelationMovementDescriptors(
            rewriter, op, tensorType, resultType, resultType.getShape(),
            *finalRelation.get(), *finalRelation.get(),
            MovementEngine::GatherScatter,
            "tile.reduce final logical movement");
    if (mlir::failed(finalDescriptors))
      return mlir::failure();
    // All legality, geometry and packing checks above are deliberately
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
        createDestAlloc(op.getLoc(), resultType, rewriter, op);
    if (mlir::failed(dest))
      return mlir::failure();

    rewriter.create<InstrFillOp>(op.getLoc(), accumulatorA, init,
                                 /*fill_domain=*/FillDomainAttr{},
                                 getDefaultNCCWorkerAttr(rewriter));
    mlir::Value currentAccumulator = accumulatorA;
    mlir::Value nextAccumulator = accumulatorB;
    for (const SlicePlan &plan : slicePlans) {
      createGatherScatterDescriptors(rewriter, op.getLoc(), op.getInput(),
                                     slice, plan.descriptors);
      llvm::SmallVector<mlir::Value, 2> inputs{currentAccumulator, slice};
      rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                          inputs, nextAccumulator,
                                          getDefaultNCCWorkerAttr(rewriter));
      std::swap(currentAccumulator, nextAccumulator);
    }
    createGatherScatterDescriptors(rewriter, op.getLoc(), currentAccumulator,
                                   *dest, *finalDescriptors);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

class GemmLowering : public mlir::OpRewritePattern<ComputeGemmOp> {
public:
  GemmLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ComputeGemmOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeGemmOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    mlir::FailureOr<llvm::SmallVector<int64_t, 3>> mkn =
        inferGemmMKN(op, rewriter);
    if (mlir::failed(mkn))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> dest =
        createDestAlloc(op.getLoc(), op.getResult().getType(), rewriter, op);
    if (mlir::failed(dest))
      return mlir::failure();

    auto instr = rewriter.create<InstrGemmOp>(
        op.getLoc(), op.getLhs(), op.getRhs(), *dest,
        getI64Attr(rewriter, (*mkn)[0]), getI64Attr(rewriter, (*mkn)[1]),
        getI64Attr(rewriter, (*mkn)[2]), op.getLhsOrientationAttr(),
        op.getRhsOrientationAttr(), op.getBatchCountAttr(),
        op.getLhsBatchDimsAttr(), op.getLhsMDimAttr(),
        op.getLhsContractingDimAttr(), op.getRhsBatchDimsAttr(),
        op.getRhsContractingDimAttr(), op.getRhsNDimAttr(),
        op.getResultBatchDimsAttr(), op.getResultMDimAttr(),
        op.getResultNDimAttr(), getDefaultNCCWorkerAttr(rewriter));
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

class ConvLowering : public mlir::OpRewritePattern<ComputeConvOp> {
public:
  ConvLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ComputeConvOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeConvOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    std::optional<mlir::RankedTensorType> input =
        getLogicalTensorTypeFromMemRef(op.getInput().getType());
    std::optional<mlir::RankedTensorType> weight =
        getLogicalTensorTypeFromMemRef(op.getWeight().getType());
    std::optional<mlir::RankedTensorType> output =
        getLogicalTensorTypeFromMemRef(op.getResult().getType());
    if (!input || !weight || !output || input->getRank() != 4 ||
        weight->getRank() != 4 || output->getRank() != 4 ||
        !input->hasStaticShape() || !weight->hasStaticShape() ||
        !output->hasStaticShape())
      return failPattern(
          rewriter, op,
          "tile.conv lowering requires static rank-4 Wafer memrefs");
    llvm::ArrayRef<int64_t> strides = op.getStrides();
    llvm::ArrayRef<int64_t> dilations = op.getDilations();
    if (strides.size() != 2 || dilations.size() != 2)
      return failPattern(rewriter, op,
                         "tile.conv lowering requires two spatial strides and "
                         "dilations");

    mlir::FailureOr<mlir::Value> dest =
        createDestAlloc(op.getLoc(), op.getResult().getType(), rewriter, op);
    if (mlir::failed(dest))
      return mlir::failure();
    auto kind =
        InstrConvKindAttr::get(rewriter.getContext(), InstrConvKind::Conv);
    auto inputShape = rewriter.getDenseI64ArrayAttr(input->getShape());
    auto weightShape = rewriter.getDenseI64ArrayAttr(weight->getShape());
    auto outputShape = rewriter.getDenseI64ArrayAttr(output->getShape());
    // Instruction fields use X/Y order while tile.conv keeps semantic H/W
    // order. Weight is already canonical XYOI at this boundary.
    auto kernelStrides = rewriter.getDenseI64ArrayAttr(
        {weight->getDimSize(0), weight->getDimSize(1), strides[1], strides[0]});
    auto instructionDilations =
        rewriter.getDenseI64ArrayAttr({dilations[1], dilations[0]});
    rewriter.create<InstrConvOp>(
        op.getLoc(), kind, op.getInput(), op.getWeight(), *dest, inputShape,
        weightShape, outputShape, op.getPadsAttr(), op.getUnpadsAttr(),
        kernelStrides, instructionDilations, getDefaultNCCWorkerAttr(rewriter));
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

static mlir::FailureOr<InstrElementwiseKindAttr>
getInstrElementwiseKindAttr(mlir::PatternRewriter &rewriter,
                            mlir::Operation *op,
                            ComputeElementwiseKindAttr computeKind) {
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
  case ComputeElementwiseKind::Ln:
    instrKind = InstrElementwiseKind::Ln;
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
        rewriter, op,
        "tile.elementwise select must lower to target movement sequence before "
        "instruction elementwise");
  }
  return InstrElementwiseKindAttr::get(rewriter.getContext(), instrKind);
}

} // namespace

mlir::FailureOr<InstrElementwiseKindAttr>
wafer::tile_region_to_instr::getAccumulationElementwiseKind(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeReduceKindAttr reduceKind, llvm::StringRef opLabel) {
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
        rewriter, op,
        llvm::Twine(opLabel)
            .concat(" lowering does not support avg accumulation")
            .str());
  }

  return InstrElementwiseKindAttr::get(rewriter.getContext(), elementwiseKind);
}

void wafer::tile_region_to_instr::populateComputeLoweringPatterns(
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns
      .add<ConvertLowering, ElementwiseLowering, ElementwiseIntoLowering,
           GemmLowering, ConvLowering>(context);
  patterns.add<ReduceLowering>(context);
}

void wafer::tile_region_to_instr::populateFillLoweringPattern(
    mlir::RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}
