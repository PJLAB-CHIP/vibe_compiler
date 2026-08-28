//===- ComputeLowering.cpp - Tile-region compute lowering --------------===//

#include "Internal.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/Target/Core/TargetCall.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

class ScratchRecorderHolder {
protected:
  explicit ScratchRecorderHolder(
      TileRegionToInstrBufferRecorder *bufferRecorder)
      : bufferRecorder(bufferRecorder) {}
  TileRegionToInstrBufferRecorder *bufferRecorder = nullptr;
};

static bool haveEqualMovementDescriptorStructure(const MovementDescriptor &lhs,
                                                 const MovementDescriptor &rhs,
                                                 bool compareByteOffset) {
  return lhs.byteCount == rhs.byteCount && lhs.innerBytes == rhs.innerBytes &&
         (!compareByteOffset || lhs.byteOffset == rhs.byteOffset) &&
         lhs.strides == rhs.strides && lhs.iterations == rhs.iterations;
}

static bool
haveEqualSliceDescriptorStructure(llvm::ArrayRef<MovementDescriptorPair> lhs,
                                  llvm::ArrayRef<MovementDescriptorPair> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip_equal(lhs, rhs), [](auto descriptors) {
           const auto &[lhsDescriptor, rhsDescriptor] = descriptors;
           return haveEqualMovementDescriptorStructure(
                      lhsDescriptor.source, rhsDescriptor.source,
                      /*compareByteOffset=*/false) &&
                  haveEqualMovementDescriptorStructure(
                      lhsDescriptor.dest, rhsDescriptor.dest,
                      /*compareByteOffset=*/true);
         });
}

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

static mlir::LogicalResult proveIdentityPhysicalTraversal(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    MovementDescriptorCache *descriptorCache, llvm::StringRef subject) {
  if (sourceType.getShape() != destType.getShape())
    return failPattern(rewriter, op,
                       (subject + " requires equal logical shapes").str());
  if (!descriptorCache || !descriptorCache->hasOrProveIdentityPhysicalTraversal(
                              sourceType, destType))
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

class ConvertLowering : public mlir::OpRewritePattern<ComputeConvertOp>,
                        private ScratchRecorderHolder {
public:
  ConvertLowering(mlir::MLIRContext *context,
                  TileRegionToInstrBufferRecorder *bufferRecorder,
                  MovementDescriptorCache *descriptorCache)
      : mlir::OpRewritePattern<ComputeConvertOp>(context),
        ScratchRecorderHolder(bufferRecorder),
        descriptorCache(descriptorCache) {}

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
    if (mlir::failed(proveIdentityPhysicalTraversal(rewriter, op, sourceType,
                                                    resultType, descriptorCache,
                                                    "tile.compute.convert")))
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

    mlir::FailureOr<mlir::Value> dest =
        createDestAlloc(op.getLoc(), resultType, rewriter, op, bufferRecorder);
    if (mlir::failed(dest))
      return mlir::failure();
    auto instr = rewriter.create<InstrConvertOp>(
        op.getLoc(), InstrConvertKindAttr::get(rewriter.getContext(), *kind),
        op.getSource(), *dest, zeroPoint, roundingMode,
        getDefaultNCCWorkerAttr(rewriter));
    if (bufferRecorder)
      bufferRecorder->recordLoweredOperation(op, instr);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  MovementDescriptorCache *descriptorCache = nullptr;
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

class ElementwiseLowering : public mlir::OpRewritePattern<ComputeElementwiseOp>,
                            private ScratchRecorderHolder {
public:
  struct MappedInputRewrite {
    mlir::Value source;
    mlir::MemRefType materializedType;
    llvm::SmallVector<MovementDescriptorPair> descriptors;
  };

  ElementwiseLowering(mlir::MLIRContext *context,
                      TileRegionToInstrBufferRecorder *bufferRecorder,
                      MovementDescriptorCache *descriptorCache)
      : mlir::OpRewritePattern<ComputeElementwiseOp>(context),
        ScratchRecorderHolder(bufferRecorder),
        descriptorCache(descriptorCache) {}

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
      mlir::FailureOr<SharedMovementDescriptorPlan> descriptors =
          descriptorCache->getOrCreate(
              rewriter, op, selectedType, resultType, resultType.getShape(),
              *relation.get(), *relation.get(), MovementEngine::GatherScatter,
              "constant select copy");
      if (mlir::failed(descriptors))
        return mlir::failure();

      mlir::FailureOr<mlir::Value> dest = createDestAlloc(
          op.getLoc(), resultType, rewriter, op, bufferRecorder);
      if (mlir::failed(dest))
        return mlir::failure();
      llvm::SmallVector<InstrGatherScatterOp, 4> lowered =
          createGatherScatterDescriptors(rewriter, op.getLoc(),
                                         constantSelect->selectedInput, *dest,
                                         **descriptors);
      if (bufferRecorder)
        for (InstrGatherScatterOp operation : lowered)
          bufferRecorder->recordLoweredOperation(op, operation);
      rewriter.replaceOp(op, *dest);
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
    llvm::SmallVector<MappedInputRewrite, 3> inputRewrites;
    inputRewrites.reserve(op.getInputs().size());
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
      MappedInputRewrite inputRewrite;
      inputRewrite.source = input;
      auto sourceType = mlir::dyn_cast<mlir::MemRefType>(input.getType());
      if (!sourceType)
        return failPattern(rewriter, op,
                           "tile.elementwise requires memref inputs");
      if (!indexingMaps) {
        if (mlir::failed(proveIdentityPhysicalTraversal(
                rewriter, op, sourceType, resultType, descriptorCache,
                "map-free tile.elementwise")))
          return mlir::failure();
        inputRewrites.push_back(std::move(inputRewrite));
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
            descriptorCache->hasOrProveIdentityPhysicalTraversal(sourceType,
                                                                 resultType)) {
          inputRewrites.push_back(std::move(inputRewrite));
          continue;
        }
      }

      inputRewrite.materializedType = mlir::MemRefType::get(
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
      mlir::FailureOr<SharedMovementDescriptorPlan> descriptors =
          descriptorCache->getOrCreate(
              rewriter, op, sourceType, inputRewrite.materializedType,
              resultType.getShape(), *sourceRelation.get(), *destRelation.get(),
              MovementEngine::GatherScatter,
              "tile.elementwise indexing map materialization");
      if (mlir::failed(descriptors))
        return mlir::failure();
      inputRewrite.descriptors.assign((*descriptors)->begin(),
                                      (*descriptors)->end());
      if (mlir::failed(proveIdentityPhysicalTraversal(
              rewriter, op, inputRewrite.materializedType, resultType,
              descriptorCache, "materialized tile.elementwise operand")))
        return mlir::failure();
      inputRewrites.push_back(std::move(inputRewrite));
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
      mlir::Type falseType = inputRewrites[2].source.getType();
      if (inputRewrites[2].materializedType)
        falseType = inputRewrites[2].materializedType;
      auto falseMemRef = mlir::dyn_cast<mlir::MemRefType>(falseType);
      analysis::IndexRelationResult relation =
          analysis::IndexRelation::identity(resultType.getShape());
      if (!falseMemRef || !relation.isExact())
        return failPattern(rewriter, op,
                           "target select copy relation is not exact");
      mlir::FailureOr<SharedMovementDescriptorPlan> descriptors =
          descriptorCache->getOrCreate(
              rewriter, op, falseMemRef, resultType, resultType.getShape(),
              *relation.get(), *relation.get(), MovementEngine::GatherScatter,
              "target select false-value copy");
      if (mlir::failed(descriptors))
        return mlir::failure();
      selectCopyDescriptors.assign((*descriptors)->begin(),
                                   (*descriptors)->end());
    }

    llvm::SmallVector<mlir::Value, 3> inputs;
    inputs.reserve(inputRewrites.size());
    for (const MappedInputRewrite &inputRewrite : inputRewrites) {
      if (!inputRewrite.materializedType) {
        inputs.push_back(inputRewrite.source);
        continue;
      }
      mlir::FailureOr<mlir::Value> materialized =
          createDestAlloc(op.getLoc(), inputRewrite.materializedType, rewriter,
                          op, bufferRecorder);
      if (mlir::failed(materialized))
        return mlir::failure();
      llvm::SmallVector<InstrGatherScatterOp, 4> lowered =
          createGatherScatterDescriptors(rewriter, op.getLoc(),
                                         inputRewrite.source, *materialized,
                                         inputRewrite.descriptors);
      if (bufferRecorder)
        for (InstrGatherScatterOp operation : lowered)
          bufferRecorder->recordLoweredOperation(op, operation);
      inputs.push_back(*materialized);
    }
    mlir::FailureOr<mlir::Value> dest =
        createDestAlloc(op.getLoc(), resultType, rewriter, op, bufferRecorder);
    if (mlir::failed(dest))
      return mlir::failure();

    if (op.getKind() == ComputeElementwiseKind::Select) {
      llvm::SmallVector<InstrGatherScatterOp, 4> lowered =
          createGatherScatterDescriptors(rewriter, op.getLoc(), inputs[2],
                                         *dest, selectCopyDescriptors);
      if (bufferRecorder)
        for (InstrGatherScatterOp operation : lowered)
          bufferRecorder->recordLoweredOperation(op, operation);
      mlir::FailureOr<mlir::Value> mask = createDestAlloc(
          op.getLoc(), resultType, rewriter, op, bufferRecorder);
      if (mlir::failed(mask))
        return mlir::failure();
      auto bit2fp =
          rewriter.create<InstrBit2FpOp>(op.getLoc(), inputs[0], *mask);
      auto maskMove = rewriter.create<InstrMaskMoveOp>(op.getLoc(), inputs[1],
                                                       *mask, *dest);
      if (bufferRecorder) {
        bufferRecorder->recordLoweredOperation(op, bit2fp);
        bufferRecorder->recordLoweredOperation(op, maskMove);
      }
      rewriter.replaceOp(op, *dest);
      return mlir::success();
    }

    auto instr = rewriter.create<InstrElementwiseOp>(
        op.getLoc(), instrKind, inputs, *dest,
        getDefaultNCCWorkerAttr(rewriter));
    if (bufferRecorder)
      bufferRecorder->recordLoweredOperation(op, instr);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  MovementDescriptorCache *descriptorCache = nullptr;
};

class ElementwiseIntoLowering
    : public mlir::OpRewritePattern<ComputeElementwiseIntoOp> {
public:
  ElementwiseIntoLowering(mlir::MLIRContext *context,
                          TileRegionToInstrBufferRecorder *bufferRecorder)
      : mlir::OpRewritePattern<ComputeElementwiseIntoOp>(context),
        bufferRecorder(bufferRecorder) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeElementwiseIntoOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    mlir::FailureOr<InstrElementwiseKindAttr> instrKind =
        getInstrElementwiseKindAttr(rewriter, op, op.getKindAttr());
    if (mlir::failed(instrKind))
      return mlir::failure();
    auto instr = rewriter.create<InstrElementwiseOp>(
        op.getLoc(), *instrKind, op.getInputs(), op.getDest(),
        getDefaultNCCWorkerAttr(rewriter));
    if (bufferRecorder)
      bufferRecorder->recordLoweredOperation(op, instr);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  TileRegionToInstrBufferRecorder *bufferRecorder = nullptr;
};

class ReduceLowering : public mlir::OpRewritePattern<ComputeReduceOp>,
                       private ScratchRecorderHolder {
public:
  ReduceLowering(mlir::MLIRContext *context,
                 TileRegionToInstrBufferRecorder *bufferRecorder,
                 MovementDescriptorCache *descriptorCache)
      : mlir::OpRewritePattern<ComputeReduceOp>(context),
        ScratchRecorderHolder(bufferRecorder),
        descriptorCache(descriptorCache) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeReduceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    if (!descriptorCache)
      return failPattern(
          rewriter, op,
          "tile.reduce lowering requires a request-local descriptor cache");
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
        mlir::FailureOr<mlir::Value> dest = createDestAlloc(
            op.getLoc(), resultType, rewriter, op, bufferRecorder);
        if (mlir::failed(dest))
          return mlir::failure();
        auto kind =
            InstrReduceKindAttr::get(rewriter.getContext(), *nativeKind);
        auto instr = rewriter.create<InstrReduceOp>(
            op.getLoc(), kind, op.getInput(), *dest,
            getI64Attr(rewriter, *targetDim),
            getDefaultNCCWorkerAttr(rewriter));
        if (bufferRecorder)
          bufferRecorder->recordLoweredOperation(op, instr);
        rewriter.replaceOp(op, *dest);
        return mlir::success();
      }
    }

    if (resultType.getRank() == 0 &&
        tupleCount > preferredMaximumOrderedReductionOperations)
      return failPattern(
          rewriter, op,
          "tile.reduce ordered scalar lowering exceeds the current 4096-tuple "
          "compiler work limit and has no legal native rank-zero route");

    auto tensorType = mlir::MemRefType::get(
        resultType.getShape(), elementType, mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(rewriter.getContext(), MemorySpace::SPM,
                        MemLayout::Tensor));

    analysis::IndexRelationResult sliceDestRelation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!sliceDestRelation.isExact())
      return failPattern(rewriter, op,
                         "tile.reduce destination relation is not exact");
    auto buildSlicePlan = [&](llvm::ArrayRef<int64_t> tuple)
        -> mlir::FailureOr<SharedMovementDescriptorPlan> {
      if (tuple.size() != reducedDims.size())
        return mlir::failure();
      llvm::SmallVector<mlir::AffineExpr, 4> sourceResults;
      sourceResults.reserve(inputType.getRank());
      size_t reducedIndex = 0;
      unsigned resultIndex = 0;
      for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
        if (llvm::is_contained(reducedDims, inputDim)) {
          sourceResults.push_back(mlir::getAffineConstantExpr(
              tuple[reducedIndex++], rewriter.getContext()));
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
      if (!sourceRelation.isExact())
        return mlir::failure();
      return descriptorCache->getOrCreate(
          rewriter, op, inputType, tensorType, resultType.getShape(),
          *sourceRelation.get(), *sliceDestRelation.get(),
          MovementEngine::GatherScatter, "tile.reduce ordered slice movement");
    };

    struct SliceRun {
      int64_t begin = 0;
      int64_t end = 0;
      SharedMovementDescriptorPlan descriptors;
      llvm::SmallVector<int64_t, 2> sourceOffsetSteps;

      int64_t size() const { return end - begin; }
    };
    llvm::SmallVector<SliceRun, 8> sliceRuns;

    auto initializeAffineRun =
        [&](int64_t begin, int64_t end, SharedMovementDescriptorPlan first,
            SharedMovementDescriptorPlan second) -> mlir::FailureOr<SliceRun> {
      if (begin < 0 || end <= begin || !first)
        return mlir::failure();
      SliceRun run{begin, end, std::move(first), {}};
      if (run.size() == 1)
        return run;
      if (!second ||
          !haveEqualSliceDescriptorStructure(*run.descriptors, *second))
        return mlir::failure();
      run.sourceOffsetSteps.reserve(run.descriptors->size());
      for (auto [base, next] : llvm::zip_equal(*run.descriptors, *second)) {
        int64_t step = 0;
        if (llvm::SubOverflow(next.source.byteOffset, base.source.byteOffset,
                              step))
          return mlir::failure();
        run.sourceOffsetSteps.push_back(step);
      }
      return run;
    };

    auto followsAffineRun = [&](const SliceRun &run,
                                const MovementDescriptorPlan &plan,
                                int64_t linearTuple) {
      if (!haveEqualSliceDescriptorStructure(*run.descriptors, plan) ||
          linearTuple < run.begin)
        return false;
      const int64_t relative = linearTuple - run.begin;
      for (auto [descriptorIndex, descriptor] : llvm::enumerate(plan)) {
        int64_t delta = 0;
        int64_t expected = 0;
        if (llvm::MulOverflow(run.sourceOffsetSteps[descriptorIndex], relative,
                              delta) ||
            llvm::AddOverflow(
                (*run.descriptors)[descriptorIndex].source.byteOffset, delta,
                expected) ||
            expected != descriptor.source.byteOffset)
          return false;
      }
      return true;
    };

    if (reducedDims.size() == 1) {
      MemoryAttr inputMemory = getWaferMemoryAttr(inputType);
      if (!inputMemory)
        return failPattern(
            rewriter, op, "tile.reduce source has no physical memory encoding");
      mlir::FailureOr<llvm::SmallVector<WaferPhysicalLayoutPiece, 2>>
          physicalPieces = inputMemory.getPhysicalLayoutPieces(inputType);
      if (mlir::failed(physicalPieces))
        return failPattern(
            rewriter, op,
            "tile.reduce cannot derive the source physical layout pieces");

      const int64_t extent = reductionShape.front();
      llvm::SmallVector<int64_t, 16> boundaries{0, extent};
      for (const WaferPhysicalLayoutPiece &piece : *physicalPieces) {
        if (piece.logicalLowerBounds.size() !=
                static_cast<size_t>(inputType.getRank()) ||
            piece.logicalUpperBounds.size() !=
                static_cast<size_t>(inputType.getRank()) ||
            piece.logicalTilePeriods.size() !=
                static_cast<size_t>(inputType.getRank()))
          return failPattern(
              rewriter, op,
              "tile.reduce physical layout piece rank is inconsistent");
        const int64_t reducedDim = reducedDims.front();
        const int64_t lower = piece.logicalLowerBounds[reducedDim];
        const int64_t upper = piece.logicalUpperBounds[reducedDim];
        const int64_t period = piece.logicalTilePeriods[reducedDim];
        if (lower > 0 && lower < extent)
          boundaries.push_back(lower);
        if (upper > 0 && upper < extent)
          boundaries.push_back(upper);
        if (period > 0) {
          for (int64_t boundary = period; boundary < extent;) {
            boundaries.push_back(boundary);
            int64_t next = 0;
            if (llvm::AddOverflow(boundary, period, next) || next <= boundary)
              return failPattern(
                  rewriter, op,
                  "tile.reduce physical layout period overflows int64");
            boundary = next;
          }
        }
      }
      llvm::sort(boundaries);
      boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                       boundaries.end());
      for (size_t index = 1; index < boundaries.size(); ++index) {
        const int64_t begin = boundaries[index - 1];
        const int64_t end = boundaries[index];
        if (begin >= end)
          continue;
        mlir::FailureOr<SharedMovementDescriptorPlan> first =
            buildSlicePlan(llvm::ArrayRef<int64_t>(begin));
        if (mlir::failed(first))
          return mlir::failure();
        mlir::FailureOr<SharedMovementDescriptorPlan> second = *first;
        if (end - begin > 1)
          second = buildSlicePlan(llvm::ArrayRef<int64_t>(begin + 1));
        if (mlir::failed(second))
          return mlir::failure();
        mlir::FailureOr<SliceRun> run =
            initializeAffineRun(begin, end, *first, *second);
        if (mlir::failed(run))
          return failPattern(
              rewriter, op,
              "tile.reduce physical layout period does not preserve the "
              "slice descriptor structure");
        if (run->size() > 2) {
          mlir::FailureOr<SharedMovementDescriptorPlan> last =
              buildSlicePlan(llvm::ArrayRef<int64_t>(end - 1));
          if (mlir::failed(last) || !followsAffineRun(*run, **last, end - 1))
            return failPattern(
                rewriter, op,
                "tile.reduce physical layout period does not produce an "
                "exact affine descriptor run");
        }
        sliceRuns.push_back(std::move(*run));
      }
    } else {
      llvm::SmallVector<SharedMovementDescriptorPlan, 8> slicePlans;
      slicePlans.reserve(static_cast<size_t>(*reductionTupleCount));
      for (int64_t linearTuple = 0; linearTuple < *reductionTupleCount;
           ++linearTuple) {
        mlir::FailureOr<llvm::SmallVector<int64_t>> tuple =
            delinearizeIndex(rewriter, op, reductionShape, linearTuple,
                             "tile.reduce ordered tuple");
        if (mlir::failed(tuple))
          return mlir::failure();
        mlir::FailureOr<SharedMovementDescriptorPlan> plan =
            buildSlicePlan(*tuple);
        if (mlir::failed(plan))
          return mlir::failure();
        slicePlans.push_back(*plan);
      }
      for (int64_t begin = 0; begin < *reductionTupleCount;) {
        int64_t end = begin + 1;
        mlir::FailureOr<SliceRun> run = initializeAffineRun(
            begin, end, slicePlans[begin], slicePlans[begin]);
        if (begin + 1 < *reductionTupleCount &&
            haveEqualSliceDescriptorStructure(*slicePlans[begin],
                                              *slicePlans[begin + 1])) {
          run = initializeAffineRun(begin, begin + 2, slicePlans[begin],
                                    slicePlans[begin + 1]);
          if (mlir::succeeded(run)) {
            end = begin + 2;
            while (end < *reductionTupleCount &&
                   followsAffineRun(*run, *slicePlans[end], end))
              ++end;
            run->end = end;
          }
        }
        if (mlir::failed(run))
          return mlir::failure();
        if (run->size() < 3) {
          run = initializeAffineRun(begin, begin + 1, slicePlans[begin],
                                    slicePlans[begin]);
          if (mlir::failed(run))
            return mlir::failure();
        }
        sliceRuns.push_back(std::move(*run));
        begin = sliceRuns.back().end;
      }
    }

    analysis::IndexRelationResult finalRelation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!finalRelation.isExact())
      return failPattern(rewriter, op,
                         "tile.reduce final relation is not exact");
    mlir::FailureOr<SharedMovementDescriptorPlan> finalDescriptors =
        descriptorCache->getOrCreate(rewriter, op, tensorType, resultType,
                                     resultType.getShape(),
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
    mlir::FailureOr<mlir::Value> accumulatorA =
        createDestAlloc(op.getLoc(), tensorType, rewriter, op, bufferRecorder);
    mlir::FailureOr<mlir::Value> accumulatorB =
        createDestAlloc(op.getLoc(), tensorType, rewriter, op, bufferRecorder);
    mlir::FailureOr<mlir::Value> slice =
        createDestAlloc(op.getLoc(), tensorType, rewriter, op, bufferRecorder);
    mlir::FailureOr<mlir::Value> dest =
        createDestAlloc(op.getLoc(), resultType, rewriter, op, bufferRecorder);
    if (mlir::failed(accumulatorA) || mlir::failed(accumulatorB) ||
        mlir::failed(slice) || mlir::failed(dest))
      return mlir::failure();

    auto initialize = rewriter.create<InstrFillOp>(
        op.getLoc(), *accumulatorA, init,
        /*fill_domain=*/FillDomainAttr{}, getDefaultNCCWorkerAttr(rewriter));
    if (bufferRecorder)
      bufferRecorder->recordLoweredOperation(op, initialize);
    mlir::Value currentAccumulator = *accumulatorA;
    mlir::Value nextAccumulator = *accumulatorB;
    for (const SliceRun &run : sliceRuns) {
      const MovementDescriptorPlan &basePlan = *run.descriptors;
      if (run.size() == 1) {
        llvm::SmallVector<InstrGatherScatterOp, 4> lowered =
            createGatherScatterDescriptors(rewriter, op.getLoc(), op.getInput(),
                                           *slice, basePlan);
        if (bufferRecorder)
          for (InstrGatherScatterOp operation : lowered)
            bufferRecorder->recordLoweredOperation(op, operation);
        llvm::SmallVector<mlir::Value, 2> inputs{currentAccumulator, *slice};
        auto accumulate = rewriter.create<InstrElementwiseOp>(
            op.getLoc(), *accumulationKind, inputs, nextAccumulator,
            getDefaultNCCWorkerAttr(rewriter));
        if (bufferRecorder)
          bufferRecorder->recordLoweredOperation(op, accumulate);
        std::swap(currentAccumulator, nextAccumulator);
        continue;
      }

      auto lower =
          rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
      auto upper = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(),
                                                                 run.size());
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          op.getLoc(), lower, upper, step,
          mlir::ValueRange{currentAccumulator, nextAccumulator});
      if (!loop.getBody()->empty() &&
          mlir::isa<mlir::scf::YieldOp>(loop.getBody()->back()))
        rewriter.eraseOp(&loop.getBody()->back());
      rewriter.setInsertionPointToStart(loop.getBody());

      for (auto [descriptorIndex, descriptor] : llvm::enumerate(basePlan)) {
        const int64_t offsetBase = descriptor.source.byteOffset;
        const int64_t offsetStep = run.sourceOffsetSteps[descriptorIndex];
        mlir::Value dynamicOffset;
        if (offsetStep == 0) {
          dynamicOffset = rewriter.create<mlir::arith::ConstantIndexOp>(
              op.getLoc(), offsetBase);
        } else {
          mlir::Value stepValue = rewriter.create<mlir::arith::ConstantIndexOp>(
              op.getLoc(), offsetStep);
          dynamicOffset = rewriter.create<mlir::arith::MulIOp>(
              op.getLoc(), loop.getInductionVar(), stepValue);
          if (offsetBase != 0) {
            mlir::Value baseValue =
                rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(),
                                                              offsetBase);
            dynamicOffset = rewriter.create<mlir::arith::AddIOp>(
                op.getLoc(), baseValue, dynamicOffset);
          }
        }
        MovementDescriptor normalizedSource = descriptor.source;
        normalizedSource.byteOffset = 0;
        InstrGatherScatterOp lowered = createGatherScatter(
            rewriter, op.getLoc(), op.getInput(), *slice, normalizedSource,
            descriptor.dest, dynamicOffset);
        if (bufferRecorder)
          bufferRecorder->recordLoweredOperation(op, lowered);
      }

      llvm::SmallVector<mlir::Value, 2> loopInputs{loop.getRegionIterArgs()[0],
                                                   *slice};
      auto accumulate = rewriter.create<InstrElementwiseOp>(
          op.getLoc(), *accumulationKind, loopInputs,
          loop.getRegionIterArgs()[1], getDefaultNCCWorkerAttr(rewriter));
      if (bufferRecorder)
        bufferRecorder->recordLoweredOperation(op, accumulate);
      rewriter.create<mlir::scf::YieldOp>(
          op.getLoc(), mlir::ValueRange{loop.getRegionIterArgs()[1],
                                        loop.getRegionIterArgs()[0]});
      rewriter.setInsertionPointAfter(loop);
      currentAccumulator = loop.getResult(0);
      nextAccumulator = loop.getResult(1);
    }
    llvm::SmallVector<InstrGatherScatterOp, 4> lowered =
        createGatherScatterDescriptors(rewriter, op.getLoc(),
                                       currentAccumulator, *dest,
                                       **finalDescriptors);
    if (bufferRecorder)
      for (InstrGatherScatterOp operation : lowered)
        bufferRecorder->recordLoweredOperation(op, operation);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  MovementDescriptorCache *descriptorCache = nullptr;
};

class GemmLowering : public mlir::OpRewritePattern<ComputeGemmOp>,
                     private ScratchRecorderHolder {
public:
  GemmLowering(mlir::MLIRContext *context,
               TileRegionToInstrBufferRecorder *bufferRecorder)
      : mlir::OpRewritePattern<ComputeGemmOp>(context),
        ScratchRecorderHolder(bufferRecorder) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeGemmOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    mlir::FailureOr<llvm::SmallVector<int64_t, 3>> mkn =
        inferGemmMKN(op, rewriter);
    if (mlir::failed(mkn))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, bufferRecorder);
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
    if (bufferRecorder)
      bufferRecorder->recordLoweredOperation(op, instr);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

class ConvLowering : public mlir::OpRewritePattern<ComputeConvOp>,
                     private ScratchRecorderHolder {
public:
  ConvLowering(mlir::MLIRContext *context,
               TileRegionToInstrBufferRecorder *bufferRecorder)
      : mlir::OpRewritePattern<ComputeConvOp>(context),
        ScratchRecorderHolder(bufferRecorder) {}

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

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, bufferRecorder);
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
    auto instr = rewriter.create<InstrConvOp>(
        op.getLoc(), kind, op.getInput(), op.getWeight(), *dest, inputShape,
        weightShape, outputShape, op.getPadsAttr(), op.getUnpadsAttr(),
        kernelStrides, instructionDilations, getDefaultNCCWorkerAttr(rewriter));
    if (bufferRecorder)
      bufferRecorder->recordLoweredOperation(op, instr);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

static mlir::FailureOr<InstrElementwiseKindAttr>
getInstrElementwiseKindAttr(mlir::PatternRewriter &rewriter,
                            mlir::Operation *op,
                            ComputeElementwiseKindAttr computeKind) {
  auto makeKind = [&](InstrElementwiseKind kind) {
    return InstrElementwiseKindAttr::get(rewriter.getContext(), kind);
  };
  switch (computeKind.getValue()) {
  case ComputeElementwiseKind::Add:
    return makeKind(InstrElementwiseKind::Add);
  case ComputeElementwiseKind::Sub:
    return makeKind(InstrElementwiseKind::Sub);
  case ComputeElementwiseKind::Mul:
    return makeKind(InstrElementwiseKind::Mul);
  case ComputeElementwiseKind::Div:
    return makeKind(InstrElementwiseKind::Div);
  case ComputeElementwiseKind::Max:
    return makeKind(InstrElementwiseKind::Max);
  case ComputeElementwiseKind::Min:
    return makeKind(InstrElementwiseKind::Min);
  case ComputeElementwiseKind::Neg:
    return makeKind(InstrElementwiseKind::Neg);
  case ComputeElementwiseKind::Recip:
    return makeKind(InstrElementwiseKind::Recip);
  case ComputeElementwiseKind::Sqrt:
    return makeKind(InstrElementwiseKind::Sqrt);
  case ComputeElementwiseKind::Rsqrt:
    return makeKind(InstrElementwiseKind::Rsqrt);
  case ComputeElementwiseKind::Exp:
    return makeKind(InstrElementwiseKind::Exp);
  case ComputeElementwiseKind::Ln:
    return makeKind(InstrElementwiseKind::Ln);
  case ComputeElementwiseKind::Tanh:
    return makeKind(InstrElementwiseKind::Tanh);
  case ComputeElementwiseKind::Eq:
    return makeKind(InstrElementwiseKind::Eq);
  case ComputeElementwiseKind::Ne:
    return makeKind(InstrElementwiseKind::Ne);
  case ComputeElementwiseKind::Lt:
    return makeKind(InstrElementwiseKind::Lt);
  case ComputeElementwiseKind::Le:
    return makeKind(InstrElementwiseKind::Le);
  case ComputeElementwiseKind::Gt:
    return makeKind(InstrElementwiseKind::Gt);
  case ComputeElementwiseKind::Ge:
    return makeKind(InstrElementwiseKind::Ge);
  case ComputeElementwiseKind::Select:
    return failFailureOr<InstrElementwiseKindAttr>(
        rewriter, op,
        "tile.elementwise select must lower to target movement sequence before "
        "instruction elementwise");
  }
  return failFailureOr<InstrElementwiseKindAttr>(
      rewriter, op, "tile.elementwise kind is outside the closed enum");
}

} // namespace

mlir::FailureOr<InstrElementwiseKindAttr>
wafer::tile_region_to_instr::getAccumulationElementwiseKind(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeReduceKindAttr reduceKind, llvm::StringRef opLabel) {
  switch (reduceKind.getValue()) {
  case ComputeReduceKind::Sum:
    return InstrElementwiseKindAttr::get(rewriter.getContext(),
                                         InstrElementwiseKind::Add);
  case ComputeReduceKind::Max:
    return InstrElementwiseKindAttr::get(rewriter.getContext(),
                                         InstrElementwiseKind::Max);
  case ComputeReduceKind::Min:
    return InstrElementwiseKindAttr::get(rewriter.getContext(),
                                         InstrElementwiseKind::Min);
  case ComputeReduceKind::Avg:
    return failFailureOr<InstrElementwiseKindAttr>(
        rewriter, op,
        llvm::Twine(opLabel)
            .concat(" lowering does not support avg accumulation")
            .str());
  }
  return failFailureOr<InstrElementwiseKindAttr>(
      rewriter, op,
      llvm::Twine(opLabel).concat(" kind is outside the closed enum").str());
}

void wafer::tile_region_to_instr::populateComputeLoweringPatterns(
    mlir::RewritePatternSet &patterns,
    TileRegionToInstrBufferRecorder *bufferRecorder,
    MovementDescriptorCache *descriptorCache) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns.add<ElementwiseIntoLowering>(context, bufferRecorder);
  patterns.add<GemmLowering, ConvLowering>(context, bufferRecorder);
  patterns.add<ConvertLowering, ElementwiseLowering>(context, bufferRecorder,
                                                     descriptorCache);
  patterns.add<ReduceLowering>(context, bufferRecorder, descriptorCache);
}

void wafer::tile_region_to_instr::populateFillLoweringPattern(
    mlir::RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}
