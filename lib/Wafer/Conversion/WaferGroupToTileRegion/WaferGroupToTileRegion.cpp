//===- WaferGroupToTileRegion.cpp - Group to tile-region conversion -------===//

#include "Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h"

#include "Wafer/Analysis/Group/LayoutPlanningAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace {

struct BufferVersions {
  mlir::Value tensor;
  mlir::Value nTensor;
  mlir::Value cx;
  mlir::Value nCx;
};

static void markFailure(TileRegionCandidate &candidate,
                        llvm::StringRef reason) {
  candidate.succeeded = false;
  candidate.failureReason = reason.str();
}

class TileRegionBodyEmitter {
public:
  explicit TileRegionBodyEmitter(TileRegionCandidate &candidate)
      : candidate(candidate) {}

  mlir::FailureOr<TileRegionOp>
  emit(GroupOp group, mlir::ValueRange convertedInputs,
       mlir::ValueRange convertedOuts,
       mlir::ConversionPatternRewriter &rewriter) {
    GroupLayoutPlan layoutPlan;
    if (mlir::failed(collectGroupLayoutPlan(group, layoutPlan)))
      return failAndReturn("tile-region candidate layout planning failed");
    if (!layoutPlan.succeeded)
      return failAndReturn(layoutPlan.failureReason);

    llvm::SmallVector<mlir::Value, 4> tileRegionInputs;
    tileRegionInputs.append(convertedInputs.begin(), convertedInputs.end());
    tileRegionInputs.append(convertedOuts.begin(), convertedOuts.end());

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto tileRegion = rewriter.create<TileRegionOp>(
        group.getLoc(), group.getResultTypes(), tileRegionInputs);
    mlir::Block *tileBlock = new mlir::Block();
    tileRegion.getBody().push_back(tileBlock);
    for (mlir::Value input : tileRegion.getInputs())
      tileBlock->addArgument(input.getType(), input.getLoc());

    rewriter.setInsertionPointToStart(tileBlock);
    if (mlir::failed(initializeBoundary(group, tileRegion, rewriter)))
      return mlir::failure();

    for (mlir::Operation &op : group.getBody().front().without_terminator()) {
      if (mlir::isa<WaferTensorCollectiveOpInterface>(&op))
        return failAndReturn(
            "collective lowering requires placement/local-rank facts");
    }

    for (OpLayoutPlan &opPlan : layoutPlan.ops) {
      if (mlir::failed(convertOp(opPlan, rewriter)))
        return mlir::failure();
    }

    if (mlir::failed(finishRegion(group, tileRegion, rewriter)))
      return mlir::failure();

    return tileRegion;
  }

private:
  TileRegionCandidate &candidate;
  llvm::DenseMap<mlir::Value, BufferVersions> buffers;
  llvm::DenseMap<mlir::Value, mlir::Value> scalarValues;
  llvm::DenseMap<mlir::Value, mlir::Attribute> scalarAttrs;
  llvm::DenseMap<mlir::Value, mlir::Value> tensorValues;
  llvm::DenseMap<mlir::Value, mlir::Value> fillInitScalars;
  llvm::DenseMap<mlir::Value, mlir::Attribute> fillInitAttrs;

  mlir::LogicalResult fail(llvm::StringRef reason) {
    markFailure(candidate, reason);
    return mlir::failure();
  }

  mlir::FailureOr<TileRegionOp> failAndReturn(llvm::StringRef reason) {
    markFailure(candidate, reason);
    return mlir::failure();
  }

  mlir::FailureOr<mlir::Value> failValue(llvm::StringRef reason) {
    markFailure(candidate, reason);
    return mlir::failure();
  }

  mlir::Attribute layoutAttr(mlir::MLIRContext *context, MemLayout layout) {
    return MemLayoutAttr::get(context, layout);
  }

  mlir::Attribute spmAttr(mlir::MLIRContext *context) {
    return MemorySpaceAttr::get(context, MemorySpace::SPM);
  }

  mlir::Type tileBufferType(mlir::Type tensorType, MemLayout layout) {
    auto *context = tensorType.getContext();
    return TileBufferType::get(context, tensorType, layoutAttr(context, layout),
                               spmAttr(context));
  }

  MemLayout alignedLayoutForTensor(mlir::RankedTensorType tensorType) const {
    return tensorType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  }

  void record(mlir::Value original, MemLayout layout, mlir::Value buffer) {
    BufferVersions &versions = buffers[original];
    switch (layout) {
    case MemLayout::Tensor:
      versions.tensor = buffer;
      break;
    case MemLayout::NTensor:
      versions.nTensor = buffer;
      break;
    case MemLayout::Cx:
      versions.cx = buffer;
      break;
    case MemLayout::NCx:
      versions.nCx = buffer;
      break;
    }
  }

  mlir::Value lookup(mlir::Value original, MemLayout layout) const {
    auto it = buffers.find(original);
    if (it == buffers.end())
      return {};
    const BufferVersions &versions = it->second;
    switch (layout) {
    case MemLayout::Tensor:
      return versions.tensor;
    case MemLayout::NTensor:
      return versions.nTensor;
    case MemLayout::Cx:
      return versions.cx;
    case MemLayout::NCx:
      return versions.nCx;
    }
    llvm_unreachable("unknown memory layout");
  }

  mlir::Value lookupAny(mlir::Value original, MemLayout &layout) const {
    auto it = buffers.find(original);
    if (it == buffers.end())
      return {};
    const BufferVersions &versions = it->second;
    if (versions.tensor) {
      layout = MemLayout::Tensor;
      return versions.tensor;
    }
    if (versions.cx) {
      layout = MemLayout::Cx;
      return versions.cx;
    }
    if (versions.nTensor) {
      layout = MemLayout::NTensor;
      return versions.nTensor;
    }
    if (versions.nCx) {
      layout = MemLayout::NCx;
      return versions.nCx;
    }
    return {};
  }

  mlir::FailureOr<mlir::Value> getOrMaterialize(mlir::Value original,
                                                MemLayout targetLayout,
                                                mlir::OpBuilder &builder) {
    if (mlir::Value existing = lookup(original, targetLayout))
      return existing;

    MemLayout sourceLayout = MemLayout::Tensor;
    mlir::Value source = lookupAny(original, sourceLayout);
    if (!source)
      return failValue("missing tile buffer for value");
    if (sourceLayout == targetLayout)
      return source;

    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType)
      return failValue("cannot materialize non-ranked-tensor value");

    mlir::Type resultType = tileBufferType(tensorType, targetLayout);
    auto materialize = builder.create<LayoutMaterializeOp>(original.getLoc(),
                                                           resultType, source);
    record(original, targetLayout, materialize.getResult());
    return materialize.getResult();
  }

  mlir::LogicalResult initializeBoundary(GroupOp group, TileRegionOp tileRegion,
                                         mlir::OpBuilder &builder) {
    mlir::Block &groupBlock = group.getBody().front();
    mlir::Block &tileBlock = tileRegion.getBody().front();
    if (groupBlock.getNumArguments() != tileBlock.getNumArguments())
      return fail("group boundary argument count mismatch");

    for (auto [groupArg, tileArg] :
         llvm::zip(groupBlock.getArguments(), tileBlock.getArguments())) {
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(groupArg.getType());
      if (!tensorType) {
        if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
                groupArg.getType())) {
          scalarValues[groupArg] = tileArg;
          continue;
        }
        return fail("group boundary is not a ranked tensor or scalar");
      }
      auto load = builder.create<LoadTileOp>(
          groupArg.getLoc(), tileBufferType(tensorType, MemLayout::Tensor),
          tileArg);
      record(groupArg, MemLayout::Tensor, load.getResult());
      tensorValues[groupArg] = tileArg;
    }
    return mlir::success();
  }

  mlir::LogicalResult convertOp(const OpLayoutPlan &opPlan,
                                mlir::OpBuilder &builder) {
    if (opPlan.kind == OpTilingDemandKind::Failure)
      return fail(opPlan.failureReason);

    mlir::Operation *op = opPlan.op;
    if (opPlan.kind == OpTilingDemandKind::Support)
      return convertSupportOp(op, builder);
    if (opPlan.kind == OpTilingDemandKind::TensorCollective)
      return fail("collective lowering requires placement/local-rank facts");

    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op))
      return convertFill(fill, builder);
    if (mlir::isa<mlir::linalg::MatmulOp>(op))
      return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder);
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
      return convertGeneric(generic, builder);

    return fail("unsupported linalg op " + op->getName().getStringRef().str());
  }

  mlir::LogicalResult convertSupportOp(mlir::Operation *op,
                                       mlir::OpBuilder &builder) {
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      if (constant->getNumResults() == 0)
        return mlir::success();

      mlir::Operation *cloned = builder.clone(*constant.getOperation());
      for (auto [originalResult, clonedResult] :
           llvm::zip(constant->getResults(), cloned->getResults())) {
        auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(originalResult.getType());
        if (!tensorType) {
          scalarValues[originalResult] = clonedResult;
          scalarAttrs[originalResult] = constant.getValue();
          continue;
        }

        auto load = builder.create<LoadTileOp>(
            constant.getLoc(), tileBufferType(tensorType, MemLayout::Tensor),
            clonedResult);
        record(originalResult, MemLayout::Tensor, load.getResult());
        tensorValues[originalResult] = clonedResult;
      }
      return mlir::success();
    }

    if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(op)) {
      auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(empty.getType());
      if (!tensorType)
        return fail("tensor.empty result is not a ranked tensor");
      auto alloc = builder.create<AllocTileOp>(
          empty.getLoc(), tileBufferType(tensorType, MemLayout::Tensor));
      record(empty.getResult(), MemLayout::Tensor, alloc.getResult());
      return mlir::success();
    }

    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(op))
      return convertTensorExtract(extract, builder);
    if (auto extractSlice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op))
      return convertTensorExtractSlice(extractSlice, builder);
    if (auto insertSlice = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op))
      return convertTensorInsertSlice(insertSlice, builder);
    if (auto expandShape = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op))
      return convertTensorReshape(expandShape.getOperation(),
                                  expandShape.getSrc(), expandShape.getResult(),
                                  builder);
    if (auto collapseShape = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op))
      return convertTensorReshape(collapseShape.getOperation(),
                                  collapseShape.getSrc(),
                                  collapseShape.getResult(), builder);

    return mlir::success();
  }

  mlir::LogicalResult convertTensorExtract(mlir::tensor::ExtractOp extract,
                                           mlir::OpBuilder &builder) {
    mlir::IRMapping mapping;
    for (mlir::Value operand : extract->getOperands()) {
      if (auto tensorIt = tensorValues.find(operand);
          tensorIt != tensorValues.end()) {
        mapping.map(operand, tensorIt->second);
        continue;
      }
      if (auto scalarIt = scalarValues.find(operand);
          scalarIt != scalarValues.end()) {
        mapping.map(operand, scalarIt->second);
        continue;
      }
      return fail("missing value for tensor.extract operand");
    }

    mlir::Operation *cloned = builder.clone(*extract.getOperation(), mapping);
    scalarValues[extract.getResult()] = cloned->getResult(0);
    return mlir::success();
  }

  bool allStatic(llvm::ArrayRef<int64_t> values) const {
    return llvm::all_of(values, [](int64_t value) {
      return value != mlir::ShapedType::kDynamic;
    });
  }

  mlir::LogicalResult
  convertTensorExtractSlice(mlir::tensor::ExtractSliceOp extractSlice,
                            mlir::OpBuilder &builder) {
    if (!allStatic(extractSlice.getStaticOffsets()) ||
        !allStatic(extractSlice.getStaticSizes()) ||
        !allStatic(extractSlice.getStaticStrides()))
      return fail("dynamic tensor.extract_slice is not representable");

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(extractSlice.getSource(), MemLayout::Tensor, builder);
    if (mlir::failed(source))
      return mlir::failure();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(extractSlice.getType());
    if (!resultTensorType)
      return fail("tensor.extract_slice result is not a ranked tensor");

    mlir::MLIRContext *context = extractSlice.getContext();
    auto offsets =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticOffsets());
    auto sizes =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticSizes());
    auto strides =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticStrides());
    auto move = builder.create<MoveExtractSliceOp>(
        extractSlice.getLoc(),
        tileBufferType(resultTensorType, MemLayout::Tensor), *source, offsets,
        sizes, strides);
    record(extractSlice.getResult(), MemLayout::Tensor, move.getResult());
    return mlir::success();
  }

  mlir::LogicalResult
  convertTensorInsertSlice(mlir::tensor::InsertSliceOp insertSlice,
                           mlir::OpBuilder &builder) {
    if (!allStatic(insertSlice.getStaticOffsets()) ||
        !allStatic(insertSlice.getStaticSizes()) ||
        !allStatic(insertSlice.getStaticStrides()))
      return fail("dynamic tensor.insert_slice is not representable");

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
    mlir::FailureOr<mlir::Value> dest =
        getOrMaterialize(insertSlice.getDest(), MemLayout::Tensor, builder);
    if (mlir::failed(source) || mlir::failed(dest))
      return mlir::failure();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getType());
    if (!resultTensorType)
      return fail("tensor.insert_slice result is not a ranked tensor");

    mlir::MLIRContext *context = insertSlice.getContext();
    auto offsets =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticOffsets());
    auto sizes =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticSizes());
    auto strides =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticStrides());
    auto move = builder.create<MoveInsertSliceOp>(
        insertSlice.getLoc(),
        tileBufferType(resultTensorType, MemLayout::Tensor), *source, *dest,
        offsets, sizes, strides);
    record(insertSlice.getResult(), MemLayout::Tensor, move.getResult());
    return mlir::success();
  }

  mlir::LogicalResult convertTensorReshape(mlir::Operation *op,
                                           mlir::Value sourceValue,
                                           mlir::Value resultValue,
                                           mlir::OpBuilder &builder) {
    MemLayout sourceLayout = MemLayout::Tensor;
    mlir::Value source = lookupAny(sourceValue, sourceLayout);
    if (!source)
      return fail("missing tile buffer for tensor reshape source");

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(resultValue.getType());
    if (!resultTensorType)
      return fail("tensor reshape result is not a ranked tensor");

    auto reshape = builder.create<ViewReshapeOp>(
        op->getLoc(), tileBufferType(resultTensorType, sourceLayout), source);
    record(resultValue, sourceLayout, reshape.getResult());
    return mlir::success();
  }

  mlir::FailureOr<mlir::Value> getScalarValue(mlir::Value original) {
    auto it = scalarValues.find(original);
    if (it == scalarValues.end())
      return failValue("missing scalar value for tile compute");
    return it->second;
  }

  mlir::LogicalResult convertFill(mlir::linalg::FillOp fill,
                                  mlir::OpBuilder &builder) {
    mlir::linalg::LinalgOp op = fill;
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
        fill->getNumResults() != 1)
      return fail("unsupported linalg.fill arity");

    mlir::FailureOr<mlir::Value> dest =
        getOrMaterialize(op.getDpsInits()[0], MemLayout::Tensor, builder);
    mlir::FailureOr<mlir::Value> value = getScalarValue(op.getDpsInputs()[0]);
    if (mlir::failed(dest) || mlir::failed(value))
      return mlir::failure();

    builder.create<ComputeFillOp>(fill.getLoc(), *dest, *value);
    record(fill.getResult(0), MemLayout::Tensor, *dest);
    fillInitScalars[fill.getResult(0)] = *value;
    if (auto attrIt = scalarAttrs.find(op.getDpsInputs()[0]);
        attrIt != scalarAttrs.end())
      fillInitAttrs[fill.getResult(0)] = attrIt->second;
    return mlir::success();
  }

  mlir::LogicalResult convertMatmul(mlir::linalg::LinalgOp op,
                                    mlir::OpBuilder &builder) {
    if (op.getNumDpsInputs() != 2 || op->getNumResults() != 1)
      return fail("unsupported matmul arity");

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
        op->getLoc(), tileBufferType(resultTensorType, MemLayout::Cx), *lhs,
        *rhs);
    record(op->getResult(0), MemLayout::Cx, gemm.getResult());
    return mlir::success();
  }

  std::optional<ComputeElementwiseKind>
  inferElementwiseKind(mlir::linalg::GenericOp generic) {
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
        generic.getBody()->getTerminator());
    if (!yield || yield.getValues().size() != 1) {
      (void)fail("unsupported linalg.generic yield");
      return std::nullopt;
    }

    mlir::Operation *def = yield.getValues()[0].getDefiningOp();
    if (!def) {
      (void)fail("unsupported linalg.generic passthrough body");
      return std::nullopt;
    }

    if (mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(def))
      return ComputeElementwiseKind::Add;
    if (mlir::isa<mlir::arith::SubFOp, mlir::arith::SubIOp>(def))
      return ComputeElementwiseKind::Sub;
    if (mlir::isa<mlir::arith::MulFOp, mlir::arith::MulIOp>(def))
      return ComputeElementwiseKind::Mul;
    if (mlir::isa<mlir::arith::DivFOp, mlir::arith::DivSIOp,
                  mlir::arith::DivUIOp>(def))
      return ComputeElementwiseKind::Div;
    if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxNumFOp,
                  mlir::arith::MaxSIOp, mlir::arith::MaxUIOp>(def))
      return ComputeElementwiseKind::Max;
    if (mlir::isa<mlir::arith::MinimumFOp, mlir::arith::MinNumFOp,
                  mlir::arith::MinSIOp, mlir::arith::MinUIOp>(def))
      return ComputeElementwiseKind::Min;
    if (mlir::isa<mlir::arith::NegFOp>(def))
      return ComputeElementwiseKind::Neg;
    if (mlir::isa<mlir::math::ExpOp>(def))
      return ComputeElementwiseKind::Exp;
    if (mlir::isa<mlir::math::SqrtOp>(def))
      return ComputeElementwiseKind::Sqrt;
    if (mlir::isa<mlir::math::RsqrtOp>(def))
      return ComputeElementwiseKind::Rsqrt;
    if (mlir::isa<mlir::math::TanhOp>(def))
      return ComputeElementwiseKind::Tanh;
    if (auto cmpf = mlir::dyn_cast<mlir::arith::CmpFOp>(def))
      return inferCompareKind(cmpf.getPredicate());
    if (auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(def))
      return inferCompareKind(cmpi.getPredicate());

    (void)fail("unsupported linalg.generic body op " +
               def->getName().getStringRef().str());
    return std::nullopt;
  }

  std::optional<ComputeElementwiseKind>
  inferCompareKind(mlir::arith::CmpFPredicate predicate) {
    switch (predicate) {
    case mlir::arith::CmpFPredicate::OEQ:
    case mlir::arith::CmpFPredicate::UEQ:
      return ComputeElementwiseKind::Eq;
    case mlir::arith::CmpFPredicate::ONE:
    case mlir::arith::CmpFPredicate::UNE:
      return ComputeElementwiseKind::Ne;
    case mlir::arith::CmpFPredicate::OLT:
    case mlir::arith::CmpFPredicate::ULT:
      return ComputeElementwiseKind::Lt;
    case mlir::arith::CmpFPredicate::OLE:
    case mlir::arith::CmpFPredicate::ULE:
      return ComputeElementwiseKind::Le;
    case mlir::arith::CmpFPredicate::OGT:
    case mlir::arith::CmpFPredicate::UGT:
      return ComputeElementwiseKind::Gt;
    case mlir::arith::CmpFPredicate::OGE:
    case mlir::arith::CmpFPredicate::UGE:
      return ComputeElementwiseKind::Ge;
    case mlir::arith::CmpFPredicate::AlwaysFalse:
    case mlir::arith::CmpFPredicate::ORD:
    case mlir::arith::CmpFPredicate::UNO:
    case mlir::arith::CmpFPredicate::AlwaysTrue:
      (void)fail("unsupported arith.cmpf predicate");
      return std::nullopt;
    }
    llvm_unreachable("unknown cmpf predicate");
  }

  std::optional<ComputeElementwiseKind>
  inferCompareKind(mlir::arith::CmpIPredicate predicate) {
    switch (predicate) {
    case mlir::arith::CmpIPredicate::eq:
      return ComputeElementwiseKind::Eq;
    case mlir::arith::CmpIPredicate::ne:
      return ComputeElementwiseKind::Ne;
    case mlir::arith::CmpIPredicate::slt:
    case mlir::arith::CmpIPredicate::ult:
      return ComputeElementwiseKind::Lt;
    case mlir::arith::CmpIPredicate::sle:
    case mlir::arith::CmpIPredicate::ule:
      return ComputeElementwiseKind::Le;
    case mlir::arith::CmpIPredicate::sgt:
    case mlir::arith::CmpIPredicate::ugt:
      return ComputeElementwiseKind::Gt;
    case mlir::arith::CmpIPredicate::sge:
    case mlir::arith::CmpIPredicate::uge:
      return ComputeElementwiseKind::Ge;
    }
    llvm_unreachable("unknown cmpi predicate");
  }

  mlir::Value unwrapScalarTensorValue(mlir::Value value) const {
    auto extract = value.getDefiningOp<mlir::tensor::ExtractOp>();
    if (!extract || !extract.getIndices().empty())
      return value;
    auto fromElements =
        extract.getTensor().getDefiningOp<mlir::tensor::FromElementsOp>();
    if (!fromElements || fromElements.getElements().size() != 1)
      return value;
    return fromElements.getElements().front();
  }

  std::optional<ComputeReduceKind>
  inferReduceKind(mlir::linalg::GenericOp generic) {
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
        generic.getBody()->getTerminator());
    if (!yield || yield.getValues().size() != 1) {
      (void)fail("unsupported linalg.generic reduce yield");
      return std::nullopt;
    }

    mlir::Operation *def =
        unwrapScalarTensorValue(yield.getValues()[0]).getDefiningOp();
    if (!def) {
      (void)fail("unsupported linalg.generic reduce passthrough body");
      return std::nullopt;
    }

    if (mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(def))
      return ComputeReduceKind::Sum;
    if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxNumFOp,
                  mlir::arith::MaxSIOp, mlir::arith::MaxUIOp>(def))
      return ComputeReduceKind::Max;
    if (mlir::isa<mlir::arith::MinimumFOp, mlir::arith::MinNumFOp,
                  mlir::arith::MinSIOp, mlir::arith::MinUIOp>(def))
      return ComputeReduceKind::Min;

    (void)fail("unsupported linalg.generic reduce body op " +
               def->getName().getStringRef().str());
    return std::nullopt;
  }

  bool hasReductionIterator(mlir::linalg::GenericOp generic) const {
    for (mlir::utils::IteratorType iteratorType :
         generic.getIteratorTypesArray()) {
      if (iteratorType == mlir::utils::IteratorType::reduction)
        return true;
    }
    return false;
  }

  mlir::LogicalResult
  getReductionInputDims(mlir::linalg::GenericOp generic,
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

  mlir::LogicalResult createReduceOp(
      mlir::Location loc, mlir::Type resultType, ComputeReduceKindAttr kindAttr,
      mlir::Value input, llvm::ArrayRef<int64_t> dims, mlir::Value init,
      mlir::Attribute initAttr, mlir::OpBuilder &builder, mlir::Value &result) {
    mlir::OperationState state(loc, ComputeReduceOp::getOperationName());
    state.addAttribute("kind", kindAttr);
    state.addAttribute(
        "dimensions", mlir::DenseI64ArrayAttr::get(builder.getContext(), dims));
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

  mlir::LogicalResult convertReduceGeneric(mlir::linalg::GenericOp generic,
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

    mlir::Value initTensor = generic.getDpsInits()[0];
    mlir::Value initScalar;
    mlir::Attribute initAttr;
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

    auto inputTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
        generic.getDpsInputs()[0].getType());
    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!inputTensorType || !resultTensorType)
      return fail("reduction generic operands/results must be ranked tensors");

    mlir::FailureOr<mlir::Value> input =
        getOrMaterialize(generic.getDpsInputs()[0],
                         alignedLayoutForTensor(inputTensorType), builder);
    if (mlir::failed(input))
      return mlir::failure();

    mlir::Type reduceResultType = tileBufferType(
        resultTensorType, alignedLayoutForTensor(resultTensorType));
    mlir::Value reduceResult;
    auto kindAttr = ComputeReduceKindAttr::get(generic.getContext(), *kind);
    if (mlir::failed(createReduceOp(generic.getLoc(), reduceResultType,
                                    kindAttr, *input, reduceDims, initScalar,
                                    initAttr, builder, reduceResult)))
      return mlir::failure();

    record(generic->getResult(0), alignedLayoutForTensor(resultTensorType),
           reduceResult);
    return mlir::success();
  }

  std::optional<unsigned>
  getPassthroughInputIndex(mlir::linalg::GenericOp generic) {
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
        generic.getBody()->getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return std::nullopt;

    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(yield.getValues()[0]);
    if (!blockArg)
      return std::nullopt;
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber >= generic.getNumDpsInputs())
      return std::nullopt;
    return argNumber;
  }

  bool isIdentityMap(mlir::AffineMap map, int64_t rank) const {
    return map.getNumDims() == static_cast<unsigned>(rank) &&
           map.getNumSymbols() == 0 && map.isIdentity();
  }

  mlir::LogicalResult convertPassthroughGeneric(mlir::linalg::GenericOp generic,
                                                mlir::OpBuilder &builder) {
    if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
      return fail("unsupported passthrough generic arity");
    std::optional<unsigned> inputIndex = getPassthroughInputIndex(generic);
    if (!inputIndex)
      return fail("unsupported linalg.generic passthrough body");

    llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
        generic.getIndexingMapsArray();
    if (indexingMaps.size() != generic.getNumDpsInputs() + 1)
      return fail("passthrough generic indexing map count mismatch");

    mlir::Value inputValue = generic.getDpsInputs()[*inputIndex];
    auto inputTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(inputValue.getType());
    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!inputTensorType || !resultTensorType)
      return fail(
          "passthrough generic operands/results must be ranked tensors");

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(inputValue, MemLayout::Tensor, builder);
    if (mlir::failed(source))
      return mlir::failure();

    mlir::AffineMap inputMap = indexingMaps[*inputIndex];
    mlir::AffineMap resultMap = indexingMaps.back();
    if (!isIdentityMap(resultMap, resultTensorType.getRank()))
      return fail("passthrough generic result map must be identity");

    mlir::Type resultType = tileBufferType(resultTensorType, MemLayout::Tensor);
    mlir::Value result;
    if (inputTensorType == resultTensorType &&
        isIdentityMap(inputMap, resultTensorType.getRank())) {
      result = builder.create<MoveCopyOp>(generic.getLoc(), resultType, *source)
                   .getResult();
    } else if (inputTensorType.getRank() == resultTensorType.getRank()) {
      llvm::SmallVector<int64_t, 4> permutation(resultTensorType.getRank(), -1);
      for (auto [sourceDim, expr] : llvm::enumerate(inputMap.getResults())) {
        auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
        if (!dimExpr)
          return fail("unsupported passthrough transpose map");
        unsigned resultDim = dimExpr.getPosition();
        if (resultDim >= permutation.size())
          return fail("passthrough transpose map dim out of range");
        permutation[resultDim] = static_cast<int64_t>(sourceDim);
      }
      if (llvm::any_of(permutation, [](int64_t dim) { return dim < 0; }))
        return fail("passthrough transpose map is incomplete");
      result =
          builder
              .create<MoveTransposeOp>(generic.getLoc(), resultType, *source,
                                       mlir::DenseI64ArrayAttr::get(
                                           generic.getContext(), permutation))
              .getResult();
    } else if (inputTensorType.getRank() < resultTensorType.getRank()) {
      llvm::SmallVector<int64_t, 4> dimensions;
      for (mlir::AffineExpr expr : inputMap.getResults()) {
        auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
        if (!dimExpr)
          return fail("unsupported passthrough broadcast map");
        dimensions.push_back(dimExpr.getPosition());
      }
      result =
          builder
              .create<MoveBroadcastOp>(generic.getLoc(), resultType, *source,
                                       mlir::DenseI64ArrayAttr::get(
                                           generic.getContext(), dimensions))
              .getResult();
    } else {
      return fail("unsupported passthrough movement rank relation");
    }

    record(generic->getResult(0), MemLayout::Tensor, result);
    return mlir::success();
  }

  mlir::LogicalResult convertGeneric(mlir::linalg::GenericOp generic,
                                     mlir::OpBuilder &builder) {
    if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
      return fail("unsupported linalg.generic arity");

    if (hasReductionIterator(generic))
      return convertReduceGeneric(generic, builder);
    if (getPassthroughInputIndex(generic))
      return convertPassthroughGeneric(generic, builder);

    std::optional<ComputeElementwiseKind> kind = inferElementwiseKind(generic);
    if (!kind)
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> inputs;
    for (mlir::Value input : generic.getDpsInputs()) {
      mlir::FailureOr<mlir::Value> buffer =
          getOrMaterialize(input, MemLayout::Tensor, builder);
      if (mlir::failed(buffer))
        return mlir::failure();
      inputs.push_back(*buffer);
    }

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!resultTensorType)
      return fail("generic result is not a ranked tensor");

    auto kindAttr =
        ComputeElementwiseKindAttr::get(generic.getContext(), *kind);
    auto elementwise = builder.create<ComputeElementwiseOp>(
        generic.getLoc(), tileBufferType(resultTensorType, MemLayout::Tensor),
        kindAttr, inputs);
    if (mlir::Attribute indexingMaps = generic->getAttr("indexing_maps"))
      elementwise->setAttr("indexing_maps", indexingMaps);

    record(generic->getResult(0), MemLayout::Tensor, elementwise.getResult());
    return mlir::success();
  }

  mlir::LogicalResult finishRegion(GroupOp group, TileRegionOp tileRegion,
                                   mlir::OpBuilder &builder) {
    auto yield =
        mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
    if (!yield)
      return fail("group terminator is not wafer.group_yield");

    unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
    llvm::SmallVector<mlir::Value, 2> yieldedTensors;
    mlir::Block &tileBlock = tileRegion.getBody().front();
    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
      mlir::FailureOr<mlir::Value> tensorBuffer =
          getOrMaterialize(value, MemLayout::Tensor, builder);
      if (mlir::failed(tensorBuffer))
        return mlir::failure();

      if (inputCount + index >= tileBlock.getNumArguments())
        return fail("group result has no output boundary");
      mlir::Value output = tileBlock.getArgument(inputCount + index);
      builder.create<StoreTileOp>(value.getLoc(), *tensorBuffer, output);
      yieldedTensors.push_back(output);
    }

    builder.create<TileYieldOp>(group.getLoc(), yieldedTensors);
    return mlir::success();
  }
};

struct GroupToTileRegionCandidatePattern
    : public mlir::OpConversionPattern<GroupOp> {
  GroupToTileRegionCandidatePattern(mlir::MLIRContext *context,
                                    TileRegionCandidate &candidate)
      : mlir::OpConversionPattern<GroupOp>(context), candidate(candidate) {}

  mlir::LogicalResult
  matchAndRewrite(GroupOp group, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    TileRegionBodyEmitter emitter(candidate);
    mlir::FailureOr<TileRegionOp> tileRegion =
        emitter.emit(group, adaptor.getInputs(), adaptor.getOuts(), rewriter);
    if (mlir::failed(tileRegion))
      return mlir::failure();

    rewriter.replaceOp(group, (*tileRegion).getResults());
    return mlir::success();
  }

  TileRegionCandidate &candidate;
};

static mlir::OwningOpRef<mlir::ModuleOp>
cloneGroupToScratchModule(GroupOp group) {
  mlir::Location loc = group.getLoc();
  mlir::OwningOpRef<mlir::ModuleOp> scratchModule = mlir::ModuleOp::create(loc);
  mlir::OpBuilder moduleBuilder(scratchModule->getBodyRegion());

  llvm::SmallVector<mlir::Type, 4> inputTypes;
  for (mlir::Value input : group.getInputs())
    inputTypes.push_back(input.getType());
  for (mlir::Value output : group.getOuts())
    inputTypes.push_back(output.getType());

  auto funcType =
      moduleBuilder.getFunctionType(inputTypes, group.getResultTypes());
  auto func = moduleBuilder.create<mlir::func::FuncOp>(
      loc, "tile_region_candidate", funcType);
  mlir::Block *entry = func.addEntryBlock();

  mlir::IRMapping mapping;
  unsigned argumentIndex = 0;
  for (mlir::Value input : group.getInputs())
    mapping.map(input, entry->getArgument(argumentIndex++));
  for (mlir::Value output : group.getOuts())
    mapping.map(output, entry->getArgument(argumentIndex++));

  mlir::OpBuilder builder(entry, entry->end());
  auto clonedGroup =
      mlir::cast<GroupOp>(builder.clone(*group.getOperation(), mapping));
  builder.create<mlir::func::ReturnOp>(loc, clonedGroup.getResults());
  return scratchModule;
}

} // namespace

mlir::LogicalResult
wafer::buildTileRegionCandidate(GroupOp group, TileRegionCandidate &candidate) {
  candidate = {};
  candidate.group = group;

  mlir::OwningOpRef<mlir::ModuleOp> scratchModule =
      cloneGroupToScratchModule(group);
  mlir::MLIRContext *context = group.getContext();

  mlir::ConversionTarget target(*context);
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         wafer::WaferDialect>();
  target.addLegalOp<mlir::tensor::ExtractOp>();
  target.addIllegalOp<GroupOp, GroupYieldOp>();

  mlir::RewritePatternSet patterns(context);
  patterns.add<GroupToTileRegionCandidatePattern>(context, candidate);

  bool conversionSucceeded = false;
  {
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionSucceeded = mlir::succeeded(
        mlir::applyFullConversion(*scratchModule, target, std::move(patterns)));
  }

  if (!conversionSucceeded) {
    if (candidate.succeeded)
      markFailure(candidate, "tile-region candidate conversion failed");
    return mlir::success();
  }

  if (mlir::failed(mlir::verify(*scratchModule))) {
    markFailure(candidate, "candidate verifier failed");
    return mlir::success();
  }

  candidate.module = std::move(scratchModule);
  return mlir::success();
}

void wafer::dumpTileRegionCandidate(const TileRegionCandidate &candidate,
                                    llvm::StringRef groupLabel,
                                    llvm::raw_ostream &os) {
  os << "wafer.tile_region.candidate group " << groupLabel << "\n";
  if (!candidate.succeeded) {
    os << "  failure " << candidate.failureReason << "\n";
    return;
  }
  candidate.module.get().print(os);
  os << "\n";
}
