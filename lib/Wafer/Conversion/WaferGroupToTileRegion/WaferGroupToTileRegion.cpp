//===- WaferGroupToTileRegion.cpp - Group to tile-region conversion -------===//

#include "Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h"

#include "Wafer/Analysis/Group/LayoutPlanningAnalysis.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace wafer {
#define GEN_PASS_DEF_CONVERTGROUPTOTILEREGIONPASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

namespace {

struct BufferVersions {
  mlir::Value tensor;
  mlir::Value nTensor;
  mlir::Value cx;
  mlir::Value nCx;
};

static void setFailureReason(std::string *failureReason,
                             llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

class TileRegionBodyEmitter {
public:
  explicit TileRegionBodyEmitter(std::string *failureReason)
      : failureReason(failureReason) {}

  mlir::FailureOr<TileRegionOp>
  emit(GroupOp group, mlir::ValueRange convertedInputs,
       mlir::ValueRange convertedOuts,
       mlir::ConversionPatternRewriter &rewriter) {
    GroupLayoutPlan layoutPlan;
    if (mlir::failed(collectGroupLayoutPlan(group, layoutPlan)))
      return failAndReturn("group-to-tile-region layout planning failed");
    if (!layoutPlan.succeeded)
      return failAndReturn(layoutPlan.failureReason);

    llvm::SmallVector<mlir::Value, 4> tileRegionInputs;
    for (auto [original, converted] :
         llvm::zip(group.getInputs(), convertedInputs)) {
      mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
          original, converted, /*readOnly=*/true, rewriter);
      if (mlir::failed(boundary))
        return mlir::failure();
      tileRegionInputs.push_back(*boundary);
    }
    for (auto [original, converted] :
         llvm::zip(group.getOuts(), convertedOuts)) {
      mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
          original, converted, /*readOnly=*/false, rewriter);
      if (mlir::failed(boundary))
        return mlir::failure();
      tileRegionInputs.push_back(*boundary);
    }

    llvm::SmallVector<mlir::Type, 2> tileRegionResultTypes;
    for (mlir::Type resultType : group.getResultTypes()) {
      auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
      if (!tensorType)
        return failAndReturn("group result is not a ranked tensor");
      tileRegionResultTypes.push_back(makeDDRMemRefType(tensorType));
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto tileRegion = rewriter.create<TileRegionOp>(
        group.getLoc(), tileRegionResultTypes, tileRegionInputs);
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
  std::string *failureReason;
  llvm::DenseMap<mlir::Value, BufferVersions> buffers;
  llvm::DenseMap<mlir::Value, mlir::Value> scalarValues;
  llvm::DenseMap<mlir::Value, mlir::Attribute> scalarAttrs;
  llvm::DenseMap<mlir::Value, mlir::Value> externalBuffers;
  llvm::DenseSet<mlir::Value> writableExternalBuffers;
  llvm::DenseMap<mlir::Value, unsigned> externalOutputIndices;
  llvm::DenseMap<mlir::Value, mlir::Value> directYieldBuffers;
  llvm::DenseMap<mlir::Value, mlir::Value> fillInitScalars;
  llvm::DenseMap<mlir::Value, mlir::Attribute> fillInitAttrs;

  struct StateSnapshot {
    llvm::DenseMap<mlir::Value, BufferVersions> buffers;
    llvm::DenseMap<mlir::Value, mlir::Value> scalarValues;
    llvm::DenseMap<mlir::Value, mlir::Attribute> scalarAttrs;
    llvm::DenseMap<mlir::Value, mlir::Value> externalBuffers;
    llvm::DenseSet<mlir::Value> writableExternalBuffers;
    llvm::DenseMap<mlir::Value, unsigned> externalOutputIndices;
    llvm::DenseMap<mlir::Value, mlir::Value> directYieldBuffers;
    llvm::DenseMap<mlir::Value, mlir::Value> fillInitScalars;
    llvm::DenseMap<mlir::Value, mlir::Attribute> fillInitAttrs;
  };

  mlir::LogicalResult fail(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<TileRegionOp> failAndReturn(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<mlir::Value> failValue(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<mlir::Type> failType(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::MemRefType makeWaferMemRefType(mlir::RankedTensorType tensorType,
                                       MemorySpace space, MemLayout layout) {
    auto *context = tensorType.getContext();
    return mlir::MemRefType::get(tensorType.getShape(),
                                 tensorType.getElementType(),
                                 mlir::MemRefLayoutAttrInterface{},
                                 MemoryAttr::get(context, space, layout));
  }

  mlir::MemRefType makeSPMMemRefType(mlir::RankedTensorType tensorType,
                                     MemLayout layout) {
    return makeWaferMemRefType(tensorType, MemorySpace::SPM, layout);
  }

  mlir::MemRefType makeDDRMemRefType(mlir::RankedTensorType tensorType) {
    return makeWaferMemRefType(tensorType, MemorySpace::DDR, MemLayout::Tensor);
  }

  bool isScalarType(mlir::Type type) const {
    return mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(type);
  }

  StateSnapshot snapshotState() const {
    return {buffers,
            scalarValues,
            scalarAttrs,
            externalBuffers,
            writableExternalBuffers,
            externalOutputIndices,
            directYieldBuffers,
            fillInitScalars,
            fillInitAttrs};
  }

  void restoreState(const StateSnapshot &snapshot) {
    buffers = snapshot.buffers;
    scalarValues = snapshot.scalarValues;
    scalarAttrs = snapshot.scalarAttrs;
    externalBuffers = snapshot.externalBuffers;
    writableExternalBuffers = snapshot.writableExternalBuffers;
    externalOutputIndices = snapshot.externalOutputIndices;
    directYieldBuffers = snapshot.directYieldBuffers;
    fillInitScalars = snapshot.fillInitScalars;
    fillInitAttrs = snapshot.fillInitAttrs;
  }

  mlir::FailureOr<mlir::Type> convertControlFlowType(mlir::Type type) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type))
      return makeSPMMemRefType(tensorType, MemLayout::Tensor);
    if (isScalarType(type))
      return type;
    return failType("control-flow value is not a ranked tensor or scalar");
  }

  mlir::LogicalResult recordControlFlowValue(mlir::Value original,
                                             mlir::Value converted) {
    if (mlir::isa<mlir::RankedTensorType>(original.getType())) {
      record(original, MemLayout::Tensor, converted);
      return mlir::success();
    }
    if (isScalarType(original.getType())) {
      scalarValues[original] = converted;
      return mlir::success();
    }
    return fail("control-flow value is not a ranked tensor or scalar");
  }

  mlir::FailureOr<mlir::Value>
  materializeControlFlowValue(mlir::Value original, mlir::OpBuilder &builder) {
    if (mlir::isa<mlir::RankedTensorType>(original.getType()))
      return getOrMaterialize(original, MemLayout::Tensor, builder);
    if (isScalarType(original.getType()))
      return getScalarValue(original);
    return failValue("control-flow yield is not a ranked tensor or scalar");
  }

  mlir::FailureOr<mlir::Value>
  materializeDdrBoundary(mlir::Value original, mlir::Value converted,
                         bool readOnly,
                         mlir::ConversionPatternRewriter &rewriter) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType) {
      if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
              original.getType()))
        return converted;
      return failValue("group boundary is not a ranked tensor or scalar");
    }
    if (!readOnly && original.getDefiningOp<mlir::tensor::EmptyOp>()) {
      auto alloc = rewriter.create<mlir::memref::AllocOp>(
          original.getLoc(), makeDDRMemRefType(tensorType));
      return alloc.getResult();
    }
    auto toMemref = rewriter.create<mlir::bufferization::ToMemrefOp>(
        original.getLoc(), makeDDRMemRefType(tensorType), converted, readOnly);
    return toMemref.getMemref();
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
    if (!source) {
      auto externalIt = externalBuffers.find(original);
      if (externalIt == externalBuffers.end())
        return failValue("missing buffer for value");

      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
      if (!tensorType)
        return failValue("cannot materialize non-ranked-tensor value");

      auto load = builder.create<StorageLoadOp>(
          original.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor),
          externalIt->second);
      record(original, MemLayout::Tensor, load.getResult());
      source = load.getResult();
      sourceLayout = MemLayout::Tensor;
    }
    if (sourceLayout == targetLayout)
      return source;

    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType)
      return failValue("cannot materialize non-ranked-tensor value");

    mlir::Type resultType = makeSPMMemRefType(tensorType, targetLayout);
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
      externalBuffers[groupArg] = tileArg;
      unsigned argIndex = groupArg.getArgNumber();
      unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
      if (argIndex >= inputCount) {
        writableExternalBuffers.insert(groupArg);
        externalOutputIndices[groupArg] = argIndex - inputCount;
      }
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

        auto ddr = builder.create<mlir::bufferization::ToMemrefOp>(
            constant.getLoc(), makeDDRMemRefType(tensorType), clonedResult,
            /*read_only=*/true);
        auto load = builder.create<StorageLoadOp>(
            constant.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor),
            ddr.getMemref());
        record(originalResult, MemLayout::Tensor, load.getResult());
        externalBuffers[originalResult] = ddr.getMemref();
      }
      return mlir::success();
    }

    if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(op)) {
      auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(empty.getType());
      if (!tensorType)
        return fail("tensor.empty result is not a ranked tensor");
      auto alloc = builder.create<mlir::memref::AllocOp>(
          empty.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor));
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
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return convertScfIf(ifOp, builder);
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return convertScfFor(forOp, builder);

    return mlir::success();
  }

  mlir::LogicalResult convertNestedOp(mlir::Operation *op,
                                      mlir::OpBuilder &builder) {
    if (mlir::isa<WaferTensorCollectiveOpInterface>(op))
      return fail("collective lowering requires placement/local-rank facts");
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op))
      return convertFill(fill, builder);
    if (mlir::isa<mlir::linalg::MatmulOp>(op))
      return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder);
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
      return convertGeneric(generic, builder);
    if (mlir::isa<mlir::arith::ConstantOp, mlir::tensor::EmptyOp,
                  mlir::tensor::ExtractOp, mlir::tensor::ExtractSliceOp,
                  mlir::tensor::InsertSliceOp, mlir::tensor::ExpandShapeOp,
                  mlir::tensor::CollapseShapeOp, mlir::scf::IfOp,
                  mlir::scf::ForOp>(op))
      return convertSupportOp(op, builder);
    return fail("unsupported op inside structured control-flow " +
                op->getName().getStringRef().str());
  }

  mlir::LogicalResult convertScfYield(mlir::scf::YieldOp yield,
                                      mlir::OpBuilder &builder) {
    llvm::SmallVector<mlir::Value, 4> yielded;
    for (mlir::Value value : yield.getResults()) {
      mlir::FailureOr<mlir::Value> converted =
          materializeControlFlowValue(value, builder);
      if (mlir::failed(converted))
        return mlir::failure();
      yielded.push_back(*converted);
    }
    builder.create<mlir::scf::YieldOp>(yield.getLoc(), yielded);
    return mlir::success();
  }

  mlir::LogicalResult convertScfBlock(mlir::Block &source,
                                      mlir::OpBuilder &builder) {
    for (mlir::Operation &op : source.without_terminator()) {
      if (mlir::failed(convertNestedOp(&op, builder)))
        return mlir::failure();
    }
    auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(source.getTerminator());
    if (!yield)
      return fail("structured control-flow body must terminate with scf.yield");
    return convertScfYield(yield, builder);
  }

  void eraseImplicitYield(mlir::Block *block) {
    if (!block || block->empty())
      return;
    if (mlir::isa<mlir::scf::YieldOp>(block->back()))
      block->back().erase();
  }

  mlir::LogicalResult convertScfIf(mlir::scf::IfOp ifOp,
                                   mlir::OpBuilder &builder) {
    mlir::FailureOr<mlir::Value> condition =
        getScalarValue(ifOp.getCondition());
    if (mlir::failed(condition))
      return mlir::failure();

    llvm::SmallVector<mlir::Type, 4> resultTypes;
    for (mlir::Type type : ifOp->getResultTypes()) {
      mlir::FailureOr<mlir::Type> converted = convertControlFlowType(type);
      if (mlir::failed(converted))
        return mlir::failure();
      resultTypes.push_back(*converted);
    }

    bool hasElse = !ifOp.getElseRegion().empty();
    auto convertedIf = builder.create<mlir::scf::IfOp>(
        ifOp.getLoc(), resultTypes, *condition, hasElse);

    {
      StateSnapshot outer = snapshotState();
      mlir::Block *thenBlock = convertedIf.thenBlock();
      eraseImplicitYield(thenBlock);
      mlir::OpBuilder thenBuilder(thenBlock, thenBlock->end());
      if (mlir::failed(convertScfBlock(*ifOp.thenBlock(), thenBuilder)))
        return mlir::failure();
      restoreState(outer);
    }

    if (hasElse) {
      StateSnapshot outer = snapshotState();
      mlir::Block *elseBlock = convertedIf.elseBlock();
      eraseImplicitYield(elseBlock);
      mlir::OpBuilder elseBuilder(elseBlock, elseBlock->end());
      if (mlir::failed(convertScfBlock(*ifOp.elseBlock(), elseBuilder)))
        return mlir::failure();
      restoreState(outer);
    }

    for (auto [original, converted] :
         llvm::zip(ifOp->getResults(), convertedIf->getResults())) {
      if (mlir::failed(recordControlFlowValue(original, converted)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult convertScfFor(mlir::scf::ForOp forOp,
                                    mlir::OpBuilder &builder) {
    mlir::FailureOr<mlir::Value> lowerBound =
        getScalarValue(forOp.getLowerBound());
    mlir::FailureOr<mlir::Value> upperBound =
        getScalarValue(forOp.getUpperBound());
    mlir::FailureOr<mlir::Value> step = getScalarValue(forOp.getStep());
    if (mlir::failed(lowerBound) || mlir::failed(upperBound) ||
        mlir::failed(step))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> initArgs;
    for (mlir::Value init : forOp.getInitArgs()) {
      mlir::FailureOr<mlir::Value> converted =
          materializeControlFlowValue(init, builder);
      if (mlir::failed(converted))
        return mlir::failure();
      initArgs.push_back(*converted);
    }

    auto convertedFor = builder.create<mlir::scf::ForOp>(
        forOp.getLoc(), *lowerBound, *upperBound, *step, initArgs);
    eraseImplicitYield(convertedFor.getBody());

    StateSnapshot outer = snapshotState();
    scalarValues[forOp.getInductionVar()] = convertedFor.getInductionVar();
    for (auto [original, converted] : llvm::zip(
             forOp.getRegionIterArgs(), convertedFor.getRegionIterArgs())) {
      if (mlir::failed(recordControlFlowValue(original, converted)))
        return mlir::failure();
    }

    mlir::OpBuilder bodyBuilder(convertedFor.getBody(),
                                convertedFor.getBody()->end());
    if (mlir::failed(convertScfBlock(*forOp.getBody(), bodyBuilder)))
      return mlir::failure();
    restoreState(outer);

    for (auto [original, converted] :
         llvm::zip(forOp->getResults(), convertedFor->getResults())) {
      if (mlir::failed(recordControlFlowValue(original, converted)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult convertTensorExtract(mlir::tensor::ExtractOp extract,
                                           mlir::OpBuilder &builder) {
    auto bufferIt = externalBuffers.find(extract.getTensor());
    if (bufferIt == externalBuffers.end())
      return fail("tensor.extract from tile-local tensor is not representable");

    llvm::SmallVector<mlir::Value, 4> indices;
    for (mlir::Value index : extract.getIndices()) {
      auto scalarIt = scalarValues.find(index);
      if (scalarIt == scalarValues.end())
        return fail("missing index value for tensor.extract");
      indices.push_back(scalarIt->second);
    }

    auto load = builder.create<mlir::memref::LoadOp>(extract.getLoc(),
                                                     bufferIt->second, indices);
    scalarValues[extract.getResult()] = load.getResult();
    return mlir::success();
  }

  bool allStatic(llvm::ArrayRef<int64_t> values) const {
    return llvm::all_of(values, [](int64_t value) {
      return value != mlir::ShapedType::kDynamic;
    });
  }

  mlir::FailureOr<mlir::Value> materializeDdrSubview(
      mlir::Location loc, mlir::Value sourceDdr,
      mlir::RankedTensorType tileTensorType, llvm::ArrayRef<int64_t> offsets,
      llvm::ArrayRef<int64_t> sizes, llvm::ArrayRef<int64_t> strides,
      mlir::OpBuilder &builder) {
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(sourceDdr.getType());
    if (!sourceType)
      return failValue("external tile view source is not a memref");
    if (sourceType.getElementType() != tileTensorType.getElementType())
      return failValue("external tile view element type mismatch");
    if (sourceType.getRank() != static_cast<int64_t>(offsets.size()) ||
        sourceType.getRank() != static_cast<int64_t>(sizes.size()) ||
        sourceType.getRank() != static_cast<int64_t>(strides.size()))
      return failValue("external tile view rank mismatch");

    auto subviewType = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            tileTensorType.getShape(), sourceType, offsets, sizes, strides));
    auto subview = builder.create<mlir::memref::SubViewOp>(
        loc, subviewType, sourceDdr, offsets, sizes, strides);
    return subview.getResult();
  }

  std::optional<unsigned>
  getSingleGroupYieldOperandIndex(mlir::Value value) const {
    mlir::OpOperand *singleUse = nullptr;
    for (mlir::OpOperand &use : value.getUses()) {
      if (singleUse)
        return std::nullopt;
      singleUse = &use;
    }
    if (!singleUse)
      return std::nullopt;

    if (!mlir::isa<GroupYieldOp>(singleUse->getOwner()))
      return std::nullopt;
    return singleUse->getOperandNumber();
  }

  mlir::LogicalResult
  convertTensorExtractSlice(mlir::tensor::ExtractSliceOp extractSlice,
                            mlir::OpBuilder &builder) {
    if (!allStatic(extractSlice.getStaticOffsets()) ||
        !allStatic(extractSlice.getStaticSizes()) ||
        !allStatic(extractSlice.getStaticStrides()))
      return fail("dynamic tensor.extract_slice is not representable");

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(extractSlice.getType());
    if (!resultTensorType)
      return fail("tensor.extract_slice result is not a ranked tensor");

    if (auto externalIt = externalBuffers.find(extractSlice.getSource());
        externalIt != externalBuffers.end()) {
      mlir::FailureOr<mlir::Value> tileView = materializeDdrSubview(
          extractSlice.getLoc(), externalIt->second, resultTensorType,
          extractSlice.getStaticOffsets(), extractSlice.getStaticSizes(),
          extractSlice.getStaticStrides(), builder);
      if (mlir::failed(tileView))
        return mlir::failure();

      auto load = builder.create<StorageLoadOp>(
          extractSlice.getLoc(),
          makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *tileView);
      record(extractSlice.getResult(), MemLayout::Tensor, load.getResult());
      return mlir::success();
    }

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(extractSlice.getSource(), MemLayout::Tensor, builder);
    if (mlir::failed(source))
      return mlir::failure();

    mlir::MLIRContext *context = extractSlice.getContext();
    auto offsets =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticOffsets());
    auto sizes =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticSizes());
    auto strides =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticStrides());
    auto move = builder.create<MoveExtractSliceOp>(
        extractSlice.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *source,
        offsets, sizes, strides);
    record(extractSlice.getResult(), MemLayout::Tensor, move.getResult());
    if (auto attrIt = fillInitAttrs.find(extractSlice.getSource());
        attrIt != fillInitAttrs.end())
      fillInitAttrs[extractSlice.getResult()] = attrIt->second;
    if (auto scalarIt = fillInitScalars.find(extractSlice.getSource());
        scalarIt != fillInitScalars.end())
      fillInitScalars[extractSlice.getResult()] = scalarIt->second;
    return mlir::success();
  }

  mlir::LogicalResult
  convertTensorInsertSlice(mlir::tensor::InsertSliceOp insertSlice,
                           mlir::OpBuilder &builder) {
    if (!allStatic(insertSlice.getStaticOffsets()) ||
        !allStatic(insertSlice.getStaticSizes()) ||
        !allStatic(insertSlice.getStaticStrides()))
      return fail("dynamic tensor.insert_slice is not representable");

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getType());
    if (!resultTensorType)
      return fail("tensor.insert_slice result is not a ranked tensor");
    auto sourceTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getSourceType());
    if (!sourceTensorType)
      return fail("tensor.insert_slice source is not a ranked tensor");

    auto externalIt = externalBuffers.find(insertSlice.getDest());
    auto outputIndexIt = externalOutputIndices.find(insertSlice.getDest());
    std::optional<unsigned> yieldIndex =
        getSingleGroupYieldOperandIndex(insertSlice.getResult());
    if (externalIt != externalBuffers.end() &&
        writableExternalBuffers.contains(insertSlice.getDest()) &&
        outputIndexIt != externalOutputIndices.end() && yieldIndex &&
        outputIndexIt->second == *yieldIndex) {
      mlir::FailureOr<mlir::Value> source =
          getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
      if (mlir::failed(source))
        return mlir::failure();

      mlir::FailureOr<mlir::Value> tileView = materializeDdrSubview(
          insertSlice.getLoc(), externalIt->second, sourceTensorType,
          insertSlice.getStaticOffsets(), insertSlice.getStaticSizes(),
          insertSlice.getStaticStrides(), builder);
      if (mlir::failed(tileView))
        return mlir::failure();

      builder.create<StorageStoreOp>(insertSlice.getLoc(), *source, *tileView);
      directYieldBuffers[insertSlice.getResult()] = externalIt->second;
      return mlir::success();
    }

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
    mlir::FailureOr<mlir::Value> dest =
        getOrMaterialize(insertSlice.getDest(), MemLayout::Tensor, builder);
    if (mlir::failed(source) || mlir::failed(dest))
      return mlir::failure();

    mlir::MLIRContext *context = insertSlice.getContext();
    auto offsets =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticOffsets());
    auto sizes =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticSizes());
    auto strides =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticStrides());
    auto move = builder.create<MoveInsertSliceOp>(
        insertSlice.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *source, *dest,
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
    if (!source) {
      mlir::FailureOr<mlir::Value> loaded =
          getOrMaterialize(sourceValue, MemLayout::Tensor, builder);
      if (mlir::failed(loaded))
        return mlir::failure();
      source = *loaded;
      sourceLayout = MemLayout::Tensor;
    }

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(resultValue.getType());
    if (!resultTensorType)
      return fail("tensor reshape result is not a ranked tensor");

    auto reshape = builder.create<ViewReshapeOp>(
        op->getLoc(), makeSPMMemRefType(resultTensorType, sourceLayout),
        source);
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
        op->getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Cx), *lhs,
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

    mlir::Type reduceResultType = makeSPMMemRefType(
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

    mlir::Type resultType =
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
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
        generic.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor), kindAttr,
        inputs);
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
      return fail("group terminator is not wafer.group.yield");

    unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
    llvm::SmallVector<mlir::Value, 2> yieldedValues;
    mlir::Block &tileBlock = tileRegion.getBody().front();
    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
      if (inputCount + index >= tileBlock.getNumArguments())
        return fail("group result has no output boundary");
      mlir::Value output = tileBlock.getArgument(inputCount + index);

      if (auto directIt = directYieldBuffers.find(value);
          directIt != directYieldBuffers.end()) {
        if (directIt->second != output)
          return fail("direct boundary storeback target mismatch");
        yieldedValues.push_back(output);
        continue;
      }

      mlir::FailureOr<mlir::Value> tensorBuffer =
          getOrMaterialize(value, MemLayout::Tensor, builder);
      if (mlir::failed(tensorBuffer))
        return mlir::failure();

      builder.create<StorageStoreOp>(value.getLoc(), *tensorBuffer, output);
      yieldedValues.push_back(output);
    }

    builder.create<TileYieldOp>(group.getLoc(), yieldedValues);
    return mlir::success();
  }
};

struct GroupToTileRegionLoweringPattern
    : public mlir::OpConversionPattern<GroupOp> {
  GroupToTileRegionLoweringPattern(mlir::MLIRContext *context,
                                   std::string *failureReason)
      : mlir::OpConversionPattern<GroupOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(GroupOp group, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    TileRegionBodyEmitter emitter(failureReason);
    mlir::FailureOr<TileRegionOp> tileRegion =
        emitter.emit(group, adaptor.getInputs(), adaptor.getOuts(), rewriter);
    if (mlir::failed(tileRegion))
      return mlir::failure();

    rewriter.setInsertionPointAfter((*tileRegion).getOperation());
    llvm::SmallVector<mlir::Value, 2> replacements;
    for (mlir::Value result : (*tileRegion).getResults()) {
      auto tensor = rewriter.create<mlir::bufferization::ToTensorOp>(
          group.getLoc(), result, /*restrict=*/true, /*writeable=*/true);
      replacements.push_back(tensor.getResult());
    }

    rewriter.replaceOp(group, replacements);
    return mlir::success();
  }

  std::string *failureReason;
};

static mlir::OwningOpRef<mlir::ModuleOp>
cloneGroupToStandaloneModule(GroupOp group) {
  mlir::Location loc = group.getLoc();
  mlir::OwningOpRef<mlir::ModuleOp> standaloneModule =
      mlir::ModuleOp::create(loc);
  mlir::OpBuilder moduleBuilder(standaloneModule->getBodyRegion());

  llvm::SmallVector<mlir::Type, 4> inputTypes;
  for (mlir::Value input : group.getInputs())
    inputTypes.push_back(input.getType());
  for (mlir::Value output : group.getOuts())
    inputTypes.push_back(output.getType());

  auto funcType =
      moduleBuilder.getFunctionType(inputTypes, group.getResultTypes());
  auto func = moduleBuilder.create<mlir::func::FuncOp>(
      loc, "group_to_tile_region", funcType);
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
  return standaloneModule;
}

static GroupOp findSingleStandaloneGroup(mlir::ModuleOp module) {
  GroupOp found;
  module.walk([&](GroupOp group) {
    if (!found)
      found = group;
  });
  return found;
}

static bool isGroupOutputBoundary(GroupOp group, mlir::Value value,
                                  unsigned outputIndex) {
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &group.getBody().front())
    return false;
  unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
  return blockArg.getArgNumber() == inputCount + outputIndex;
}

static mlir::LogicalResult validateCandidateTile(
    mlir::RankedTensorType resultType, llvm::ArrayRef<int64_t> offsets,
    llvm::ArrayRef<int64_t> sizes, std::string *failureReason) {
  if (offsets.size() != static_cast<size_t>(resultType.getRank()) ||
      sizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "candidate tile rank does not match group result rank");
    return mlir::failure();
  }

  for (auto [dim, values] : llvm::enumerate(llvm::zip(offsets, sizes))) {
    int64_t offset = std::get<0>(values);
    int64_t size = std::get<1>(values);
    int64_t bound = resultType.getDimSize(dim);
    if (mlir::ShapedType::isDynamic(bound)) {
      setFailureReason(failureReason,
                       "candidate tile requires static result shape");
      return mlir::failure();
    }
    if (offset < 0 || size <= 0 || offset + size > bound) {
      setFailureReason(failureReason,
                       "candidate tile is outside group result bounds");
      return mlir::failure();
    }
  }
  return mlir::success();
}

struct CandidateLoopTile {
  llvm::SmallVector<mlir::OpFoldResult, 4> loopOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> ivs;
  llvm::SmallVector<mlir::OpFoldResult, 4> tileSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizeBounds;
};

static llvm::SmallVector<unsigned, 2>
getReductionLoopDims(mlir::linalg::LinalgOp op) {
  llvm::SmallVector<unsigned, 2> dims;
  for (auto [index, iteratorType] :
       llvm::enumerate(op.getIteratorTypesArray())) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      dims.push_back(static_cast<unsigned>(index));
  }
  return dims;
}

static mlir::LogicalResult
buildCandidateLoopTile(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::linalg::LinalgOp op, mlir::AffineMap outputMap,
                       llvm::ArrayRef<int64_t> candidateOffsets,
                       llvm::ArrayRef<int64_t> candidateSizes,
                       llvm::ArrayRef<int64_t> candidateReductionOffsets,
                       llvm::ArrayRef<int64_t> candidateReductionSizes,
                       CandidateLoopTile &tile, std::string *failureReason) {
  llvm::SmallVector<int64_t, 4> loopRanges = op.getStaticLoopRanges();
  if (llvm::any_of(loopRanges, [](int64_t value) {
        return mlir::ShapedType::isDynamic(value);
      })) {
    setFailureReason(failureReason,
                     "candidate tile requires static linalg loop ranges");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(op);
  bool hasReductionSplit =
      !candidateReductionOffsets.empty() || !candidateReductionSizes.empty();
  if (candidateReductionOffsets.size() != candidateReductionSizes.size() ||
      (hasReductionSplit &&
       candidateReductionOffsets.size() != reductionLoopDims.size())) {
    setFailureReason(failureReason, "candidate reduction split rank mismatch");
    return mlir::failure();
  }

  llvm::DenseMap<unsigned, unsigned> resultDimForLoopDim;
  for (auto [resultDim, expr] : llvm::enumerate(outputMap.getResults())) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr) {
      setFailureReason(failureReason,
                       "candidate tile requires permutation-only output map");
      return mlir::failure();
    }
    resultDimForLoopDim[dimExpr.getPosition()] =
        static_cast<unsigned>(resultDim);
  }

  llvm::DenseMap<unsigned, unsigned> reductionOrdinalForLoopDim;
  for (auto [ordinal, loopDim] : llvm::enumerate(reductionLoopDims))
    reductionOrdinalForLoopDim[loopDim] = static_cast<unsigned>(ordinal);

  mlir::OpFoldResult zero = builder.getIndexAttr(0);
  for (auto [loopDim, loopRange] : llvm::enumerate(loopRanges)) {
    tile.sizeBounds.push_back(builder.getIndexAttr(loopRange));
    tile.loopOffsets.push_back(zero);
    tile.tileSizes.push_back(zero);

    auto resultDimIt = resultDimForLoopDim.find(static_cast<unsigned>(loopDim));
    if (resultDimIt != resultDimForLoopDim.end()) {
      unsigned resultDim = resultDimIt->second;
      tile.loopOffsets.back() =
          builder.getIndexAttr(candidateOffsets[resultDim]);
      tile.tileSizes.back() = builder.getIndexAttr(candidateSizes[resultDim]);
      tile.ivs.push_back(tile.loopOffsets.back());
      continue;
    }

    auto reductionDimIt =
        reductionOrdinalForLoopDim.find(static_cast<unsigned>(loopDim));
    if (reductionDimIt == reductionOrdinalForLoopDim.end())
      continue;
    if (!hasReductionSplit)
      continue;

    unsigned reductionOrdinal = reductionDimIt->second;
    int64_t reductionOffset = candidateReductionOffsets[reductionOrdinal];
    int64_t reductionSize = candidateReductionSizes[reductionOrdinal];
    if (reductionOffset < 0 || reductionSize <= 0 ||
        reductionOffset + reductionSize > loopRange) {
      setFailureReason(failureReason,
                       "candidate reduction split is outside loop bounds");
      return mlir::failure();
    }
    tile.loopOffsets.back() = builder.getIndexAttr(reductionOffset);
    tile.tileSizes.back() = builder.getIndexAttr(reductionSize);
    tile.ivs.push_back(tile.loopOffsets.back());
  }

  return mlir::success();
}

struct ReductionChunk {
  llvm::SmallVector<int64_t, 2> offsets;
  llvm::SmallVector<int64_t, 2> sizes;
};

static void
buildReductionChunkProducts(llvm::ArrayRef<int64_t> ranges,
                            llvm::ArrayRef<int64_t> splitSizes, unsigned dim,
                            llvm::SmallVectorImpl<int64_t> &currentOffsets,
                            llvm::SmallVectorImpl<int64_t> &currentSizes,
                            llvm::SmallVectorImpl<ReductionChunk> &chunks) {
  if (dim == ranges.size()) {
    chunks.push_back(
        ReductionChunk{llvm::SmallVector<int64_t, 2>(currentOffsets.begin(),
                                                     currentOffsets.end()),
                       llvm::SmallVector<int64_t, 2>(currentSizes.begin(),
                                                     currentSizes.end())});
    return;
  }

  for (int64_t offset = 0; offset < ranges[dim]; offset += splitSizes[dim]) {
    currentOffsets.push_back(offset);
    currentSizes.push_back(std::min(splitSizes[dim], ranges[dim] - offset));
    buildReductionChunkProducts(ranges, splitSizes, dim + 1, currentOffsets,
                                currentSizes, chunks);
    currentOffsets.pop_back();
    currentSizes.pop_back();
  }
}

static mlir::FailureOr<llvm::SmallVector<ReductionChunk, 8>>
buildReductionChunks(mlir::linalg::LinalgOp root,
                     llvm::ArrayRef<int64_t> candidateReductionTileSizes,
                     std::string *failureReason) {
  if (candidateReductionTileSizes.empty())
    return llvm::SmallVector<ReductionChunk, 8>{ReductionChunk{}};

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(root);
  if (candidateReductionTileSizes.size() != reductionLoopDims.size()) {
    setFailureReason(failureReason, "candidate reduction split rank mismatch");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 4> loopRanges = root.getStaticLoopRanges();
  llvm::SmallVector<int64_t, 2> reductionRanges;
  for (unsigned loopDim : reductionLoopDims)
    reductionRanges.push_back(loopRanges[loopDim]);

  for (auto [range, splitSize] :
       llvm::zip(reductionRanges, candidateReductionTileSizes)) {
    if (splitSize <= 0 || splitSize > range) {
      setFailureReason(failureReason,
                       "candidate reduction split is outside loop bounds");
      return mlir::failure();
    }
  }

  llvm::SmallVector<ReductionChunk, 8> chunks;
  llvm::SmallVector<int64_t, 2> currentOffsets;
  llvm::SmallVector<int64_t, 2> currentSizes;
  buildReductionChunkProducts(reductionRanges, candidateReductionTileSizes,
                              /*dim=*/0, currentOffsets, currentSizes, chunks);
  return chunks;
}

static mlir::Value unwrapScalarTensorValue(mlir::Value value) {
  auto extract = value.getDefiningOp<mlir::tensor::ExtractOp>();
  if (!extract || !extract.getIndices().empty())
    return value;
  auto fromElements =
      extract.getTensor().getDefiningOp<mlir::tensor::FromElementsOp>();
  if (!fromElements || fromElements.getElements().size() != 1)
    return value;
  return fromElements.getElements().front();
}

static std::optional<ComputeReduceKind>
inferCandidateReduceKind(mlir::linalg::GenericOp generic,
                         std::string *failureReason) {
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
      generic.getBody()->getTerminator());
  if (!yield || yield.getValues().size() != 1) {
    setFailureReason(failureReason,
                     "candidate reduction split requires one reduce yield");
    return std::nullopt;
  }

  mlir::Operation *def =
      unwrapScalarTensorValue(yield.getValues()[0]).getDefiningOp();
  if (!def) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires structured reduce body");
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

  setFailureReason(failureReason,
                   "candidate reduction split requires sum/max/min body");
  return std::nullopt;
}

static mlir::FailureOr<ComputeReduceKind>
getCandidateCombineKind(mlir::linalg::LinalgOp root,
                        std::string *failureReason) {
  if (mlir::isa<mlir::linalg::MatmulOp>(root.getOperation()))
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
      "candidate reduction split requires matmul or generic reduction root");
  return mlir::failure();
}

static mlir::FailureOr<mlir::TypedAttr>
getNeutralScalarAttr(mlir::Type elementType, ComputeReduceKind kind,
                     std::string *failureReason) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    switch (kind) {
    case ComputeReduceKind::Sum:
      return mlir::cast<mlir::TypedAttr>(
          mlir::FloatAttr::get(floatType, 0.0));
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
                        ComputeReduceKind kind,
                        std::string *failureReason) {
  mlir::FailureOr<mlir::TypedAttr> attr = getNeutralScalarAttr(
      resultType.getElementType(), kind, failureReason);
  if (mlir::failed(attr))
    return mlir::failure();

  auto constant = builder.create<mlir::arith::ConstantOp>(loc, *attr);
  auto empty = builder.create<mlir::tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());
  auto fill =
      builder.create<mlir::linalg::FillOp>(loc, constant.getResult(),
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

static mlir::FailureOr<mlir::Value> materializeCandidateRootTile(
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
  llvm::SmallVector<mlir::Value, 4> insertOperands;
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
    if (insertOperands.empty()) {
      insertOperands = tiledOperands;
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
    mlir::FailureOr<mlir::Value> combined = createPartialCombine(
        builder, root.getLoc(), *combineKind, accumulator, partial,
        outputInitTile, failureReason);
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

  if (hasDirectOutputInit) {
    llvm::SmallVector<mlir::Value, 2> inserted = mlir::linalg::insertSlicesBack(
        builder, root.getLoc(), root, insertOperands,
        mlir::ValueRange{accumulator});
    if (inserted.size() != 1) {
      setFailureReason(
          failureReason,
          "candidate tile materialization expected one inserted result");
      return mlir::failure();
    }
    return inserted.front();
  }

  unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
  mlir::Block &body = group.getBody().front();
  if (inputCount + outputIndex >= body.getNumArguments()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output boundary");
    return mlir::failure();
  }

  mlir::Value outputBoundary = body.getArgument(inputCount + outputIndex);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  offsets.reserve(candidateTileOffsets.size());
  sizes.reserve(candidateTileSizes.size());
  strides.reserve(candidateTileSizes.size());
  for (auto [offset, size] : llvm::zip(candidateTileOffsets,
                                       candidateTileSizes)) {
    offsets.push_back(builder.getIndexAttr(offset));
    sizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }

  auto inserted = builder.create<mlir::tensor::InsertSliceOp>(
      root.getLoc(), accumulator, outputBoundary, offsets, sizes, strides);
  return inserted.getResult();
}

static mlir::LogicalResult materializeCandidateTileSlices(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  auto yield =
      mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
  if (!yield || yield.getValues().empty()) {
    setFailureReason(
        failureReason,
        "candidate tile materialization requires group results");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::linalg::LinalgOp, 4> roots;
  for (mlir::Value value : yield.getValues()) {
    auto root =
        mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(value.getDefiningOp());
    if (!root) {
      setFailureReason(failureReason,
                       "candidate tile materialization requires linalg roots");
      return mlir::failure();
    }
    roots.push_back(root);
  }

  for (mlir::linalg::LinalgOp root : roots) {
    for (mlir::Value result : root->getResults()) {
      for (mlir::OpOperand &use : result.getUses()) {
        if (use.getOwner() == yield.getOperation())
          continue;
        setFailureReason(
            failureReason,
            "candidate multi-output coverage requires independent yielded roots");
        return mlir::failure();
      }
    }
  }

  llvm::SmallVector<mlir::Value, 4> insertedValues;
  for (auto [index, root] : llvm::enumerate(roots)) {
    mlir::FailureOr<mlir::Value> inserted = materializeCandidateRootTile(
        group, root, static_cast<unsigned>(index), candidateTileOffsets,
        candidateTileSizes, candidateReductionTileSizes, failureReason);
    if (mlir::failed(inserted))
      return mlir::failure();
    insertedValues.push_back(*inserted);
  }

  for (auto [index, inserted] : llvm::enumerate(insertedValues))
    yield->setOperand(index, inserted);
  for (mlir::linalg::LinalgOp root : roots)
    root->erase();
  return mlir::success();
}

static void configureGroupToTileRegionTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::arith::ArithDialect,
                         mlir::bufferization::BufferizationDialect,
                         mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect, wafer::WaferDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalOp<GroupOp, GroupYieldOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

static mlir::LogicalResult
convertGroupToTileRegionModuleInPlace(mlir::ModuleOp module,
                                      mlir::MLIRContext *context,
                                      std::string *failureReason) {
  mlir::ConversionTarget target(*context);
  configureGroupToTileRegionTarget(target);

  mlir::RewritePatternSet patterns(context);
  patterns.add<GroupToTileRegionLoweringPattern>(context, failureReason);

  bool conversionSucceeded = false;
  {
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionSucceeded = mlir::succeeded(
        mlir::applyFullConversion(module, target, std::move(patterns)));
  }

  if (!conversionSucceeded) {
    if (!failureReason || failureReason->empty())
      setFailureReason(failureReason, "group-to-tile-region lowering failed");
    return mlir::failure();
  }

  if (mlir::failed(mlir::verify(module))) {
    setFailureReason(failureReason,
                     "lowered tile-region module failed verifier");
    return mlir::failure();
  }

  return mlir::success();
}

struct ConvertGroupToTileRegionPass
    : public wafer::impl::ConvertGroupToTileRegionPassBase<
          ConvertGroupToTileRegionPass> {
  using wafer::impl::ConvertGroupToTileRegionPassBase<
      ConvertGroupToTileRegionPass>::ConvertGroupToTileRegionPassBase;

  void runOnOperation() final {
    mlir::MLIRContext *context = &getContext();
    mlir::ConversionTarget target(*context);
    configureGroupToTileRegionTarget(target);

    std::string failureReason;
    mlir::RewritePatternSet patterns(context);
    patterns.add<GroupToTileRegionLoweringPattern>(context, &failureReason);

    if (mlir::succeeded(mlir::applyFullConversion(getOperation(), target,
                                                  std::move(patterns))))
      return;

    if (!failureReason.empty())
      getOperation().emitError(failureReason);
    else
      getOperation().emitError("group to tile-region conversion failed");
    signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult
wafer::lowerGroupToTileRegionModule(GroupOp group,
                                    mlir::OwningOpRef<mlir::ModuleOp> &module,
                                    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();

  module = cloneGroupToStandaloneModule(group);
  return convertGroupToTileRegionModuleInPlace(*module, group.getContext(),
                                               failureReason);
}

mlir::LogicalResult wafer::lowerCandidateGroupToTileRegionModule(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason) {
  if (failureReason)
    failureReason->clear();

  module = cloneGroupToStandaloneModule(group);
  GroupOp clonedGroup = findSingleStandaloneGroup(*module);
  if (!clonedGroup) {
    setFailureReason(failureReason, "standalone module has no wafer.group");
    return mlir::failure();
  }

  if (mlir::failed(materializeCandidateTileSlices(
          clonedGroup, candidateTileOffsets, candidateTileSizes,
          candidateReductionTileSizes, failureReason)))
    return mlir::failure();

  return convertGroupToTileRegionModuleInPlace(*module, group.getContext(),
                                               failureReason);
}

void wafer::dumpGroupToTileRegionModule(mlir::ModuleOp module,
                                        llvm::StringRef groupLabel,
                                        llvm::raw_ostream &os) {
  os << "wafer.group_to_tile_region group " << groupLabel << "\n";
  module.print(os);
  os << "\n";
}
