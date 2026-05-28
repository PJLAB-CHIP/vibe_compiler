//===- MaterializeSingleTile.cpp - Materialize single-tile groups as tile_region ----------===//

#include "Wafer/Transforms/Passes.h"

#include "Support/AttentionGemmUtils.h"
#include "Support/ElementwiseUtils.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace wafer {
namespace {

static wafer::TileBufferType getSPMTileBuffer(mlir::MLIRContext *context,
                                              mlir::RankedTensorType tensorType,
                                              wafer::MemLayout layout) {
  return wafer::TileBufferType::get(
      context, tensorType, wafer::MemLayoutAttr::get(context, layout),
      wafer::MemorySpaceAttr::get(context, wafer::MemorySpace::SPM));
}

static bool hasStaticMismatch(int64_t lhs, int64_t rhs) {
  return lhs != mlir::ShapedType::kDynamic &&
         rhs != mlir::ShapedType::kDynamic && lhs != rhs;
}

static std::optional<mlir::Attribute>
getScalarConstantAttr(mlir::Attribute attr) {
  if (mlir::isa<mlir::FloatAttr, mlir::IntegerAttr>(attr))
    return attr;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return std::nullopt;
  return dense.getSplatValue<mlir::Attribute>();
}

static std::optional<mlir::Attribute> getScalarConstantAttr(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    return getScalarConstantAttr(constant.getValue());
  if (auto extract = value.getDefiningOp<mlir::tensor::ExtractOp>()) {
    if (extract.getIndices().empty())
      return getScalarConstantAttr(extract.getTensor());
  }
  return std::nullopt;
}

static std::optional<mlir::Attribute>
getReduceInitValueAttr(mlir::Value output) {
  auto fill = output.getDefiningOp<mlir::linalg::FillOp>();
  if (!fill || fill.getInputs().size() != 1)
    return std::nullopt;
  return getScalarConstantAttr(fill.getInputs()[0]);
}

static mlir::linalg::MatmulOp getSingleMatmulBody(wafer::GroupOp group) {
  mlir::linalg::MatmulOp matmul;
  mlir::Block &block = group.getBody().front();
  for (mlir::Operation &op : block) {
    if (&op == block.getTerminator())
      continue;
    auto candidate = mlir::dyn_cast<mlir::linalg::MatmulOp>(op);
    if (!candidate || matmul)
      return {};
    matmul = candidate;
  }
  return matmul;
}

static mlir::linalg::GenericOp
getSingleAttentionGemmBody(wafer::GroupOp group) {
  mlir::linalg::GenericOp generic;
  mlir::Block &block = group.getBody().front();
  for (mlir::Operation &op : block) {
    if (&op == block.getTerminator())
      continue;
    auto candidate = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
    if (!candidate || generic)
      return {};
    if (!matchAttentionGemm(candidate))
      return {};
    generic = candidate;
  }
  return generic;
}

static mlir::linalg::GenericOp
getSingleElementwiseBody(wafer::GroupOp group) {
  mlir::linalg::GenericOp elementwise;
  mlir::Block &block = group.getBody().front();
  for (mlir::Operation &op : block) {
    if (&op == block.getTerminator())
      continue;
    auto candidate = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
    if (!candidate || elementwise)
      return {};
    if (!isLimitedBroadcastElementwiseGeneric(candidate))
      return {};
    elementwise = candidate;
  }
  return elementwise;
}

static mlir::linalg::ReduceOp getSingleReduceBody(wafer::GroupOp group) {
  mlir::linalg::ReduceOp reduce;
  mlir::Block &block = group.getBody().front();
  for (mlir::Operation &op : block) {
    if (&op == block.getTerminator())
      continue;
    auto candidate = mlir::dyn_cast<mlir::linalg::ReduceOp>(op);
    if (!candidate || reduce)
      return {};
    reduce = candidate;
  }
  return reduce;
}

static bool areBlockArguments(mlir::Value lhs, mlir::Value rhs,
                              mlir::BlockArgument arg0,
                              mlir::BlockArgument arg1) {
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

static std::optional<wafer::ComputeReduceKind>
mapReduceKind(mlir::linalg::ReduceOp reduce) {
  if (reduce->getNumResults() != 1 || reduce.getInputs().size() != 1 ||
      reduce.getInits().size() != 1 || reduce.getRegion().empty())
    return std::nullopt;

  mlir::Block &body = reduce.getRegion().front();
  if (body.getNumArguments() != 2 || !body.getTerminator() ||
      body.getTerminator()->getNumOperands() != 1)
    return std::nullopt;

  mlir::Value yielded = body.getTerminator()->getOperand(0);
  if (auto add = yielded.getDefiningOp<mlir::arith::AddFOp>())
    if (areBlockArguments(add->getOperand(0), add->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Sum;
  if (auto add = yielded.getDefiningOp<mlir::arith::AddIOp>())
    if (areBlockArguments(add->getOperand(0), add->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Sum;
  if (auto max = yielded.getDefiningOp<mlir::arith::MaximumFOp>())
    if (areBlockArguments(max->getOperand(0), max->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Max;
  if (auto max = yielded.getDefiningOp<mlir::arith::MaxSIOp>())
    if (areBlockArguments(max->getOperand(0), max->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Max;
  if (auto min = yielded.getDefiningOp<mlir::arith::MinimumFOp>())
    if (areBlockArguments(min->getOperand(0), min->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Min;
  if (auto min = yielded.getDefiningOp<mlir::arith::MinSIOp>())
    if (areBlockArguments(min->getOperand(0), min->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Min;

  return std::nullopt;
}

static mlir::LogicalResult
materializeAttentionGemmGroup(wafer::GroupOp group,
                              mlir::linalg::GenericOp generic) {
  if (group.getNumResults() != 1 || group.getOuts().size() != 1)
    return group.emitOpError(
        "cannot materialize single tile: expected one group result and out");

  std::optional<AttentionGemmDims> dims = matchAttentionGemm(generic);
  if (!dims)
    return generic.emitOpError(
        "cannot materialize single tile: unsupported attention contraction");

  llvm::SmallVector<mlir::Value> genericInputs = generic.getDpsInputs();
  mlir::OperandRange genericOuts = generic.getDpsInits();
  if (genericInputs.size() != 2 || genericOuts.size() != 1 ||
      generic->getNumResults() != 1)
    return generic.emitOpError(
        "cannot materialize single tile: expected attention contraction with "
        "two inputs, one out, and one result");

  auto lhsType =
      mlir::dyn_cast<mlir::RankedTensorType>(genericInputs[0].getType());
  auto rhsType =
      mlir::dyn_cast<mlir::RankedTensorType>(genericInputs[1].getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!lhsType || !rhsType || !resultType)
    return generic.emitOpError(
        "cannot materialize single tile: attention tensors must be ranked");

  llvm::SmallVector<mlir::Value> tileRegionInputs(group.getInputs().begin(),
                                                  group.getInputs().end());
  tileRegionInputs.append(group.getOuts().begin(), group.getOuts().end());

  mlir::OpBuilder builder(group);
  auto tileRegion = builder.create<wafer::TileRegionOp>(
      group.getLoc(), group->getResultTypes(), tileRegionInputs);

  mlir::Block *body = new mlir::Block();
  tileRegion.getBody().push_back(body);

  mlir::IRMapping mapping;
  mlir::Block &groupBlock = group.getBody().front();
  for (auto [oldArg, input] :
       llvm::zip(groupBlock.getArguments(), tileRegionInputs)) {
    mlir::BlockArgument newArg =
        body->addArgument(input.getType(), input.getLoc());
    mapping.map(oldArg, newArg);
  }

  mlir::Value lhs = mapping.lookup(genericInputs[0]);
  mlir::Value rhs = mapping.lookup(genericInputs[1]);
  mlir::Value out = mapping.lookup(genericOuts[0]);
  mlir::MLIRContext *context = group.getContext();
  mlir::Location loc = group.getLoc();
  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(body);

  auto lhsTensorType =
      getSPMTileBuffer(context, lhsType, wafer::MemLayout::Tensor);
  auto rhsTensorType =
      getSPMTileBuffer(context, rhsType, wafer::MemLayout::Tensor);
  auto lhsCxType = getSPMTileBuffer(context, lhsType, wafer::MemLayout::Cx);
  auto rhsCxType = getSPMTileBuffer(context, rhsType, wafer::MemLayout::Cx);
  auto resultCxType =
      getSPMTileBuffer(context, resultType, wafer::MemLayout::Cx);
  auto resultTensorType =
      getSPMTileBuffer(context, resultType, wafer::MemLayout::Tensor);

  auto lhsTile = bodyBuilder.create<wafer::LoadTileOp>(loc, lhsTensorType, lhs);
  auto rhsTile = bodyBuilder.create<wafer::LoadTileOp>(loc, rhsTensorType, rhs);
  auto lhsCx = bodyBuilder.create<wafer::LayoutMaterializeOp>(
      loc, lhsCxType, lhsTile.getResult());
  auto rhsCx = bodyBuilder.create<wafer::LayoutMaterializeOp>(
      loc, rhsCxType, rhsTile.getResult());
  auto gemm = bodyBuilder.create<wafer::ComputeGemmOp>(
      loc, resultCxType, lhsCx.getResult(), rhsCx.getResult());
  setAttentionGemmAttrs(gemm.getOperation(), bodyBuilder, resultType, *dims);
  auto resultTensor = bodyBuilder.create<wafer::LayoutMaterializeOp>(
      loc, resultTensorType, gemm.getResult());
  bodyBuilder.create<wafer::StoreTileOp>(loc, resultTensor.getResult(), out);
  bodyBuilder.create<wafer::TileYieldOp>(loc, out);

  group->replaceAllUsesWith(tileRegion->getResults());
  group.erase();
  return mlir::success();
}

static mlir::LogicalResult
materializeMatmulGroup(wafer::GroupOp group, mlir::linalg::MatmulOp matmul) {
  if (group.getNumResults() != 1 || group.getOuts().size() != 1)
    return group.emitOpError(
        "cannot materialize single tile: expected one group result and out");
  if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1 ||
      matmul->getNumResults() != 1)
    return matmul.emitOpError(
        "cannot materialize single tile: expected tensor matmul with two "
        "inputs, one out, and one result");

  auto lhsType =
      mlir::dyn_cast<mlir::RankedTensorType>(matmul.getInputs()[0].getType());
  auto rhsType =
      mlir::dyn_cast<mlir::RankedTensorType>(matmul.getInputs()[1].getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(matmul.getResult(0).getType());
  if (!lhsType || !rhsType || !resultType)
    return matmul.emitOpError(
        "cannot materialize single tile: matmul tensors must be ranked");

  llvm::SmallVector<mlir::Value> tileRegionInputs(group.getInputs().begin(),
                                                  group.getInputs().end());
  tileRegionInputs.append(group.getOuts().begin(), group.getOuts().end());

  mlir::OpBuilder builder(group);
  auto tileRegion = builder.create<wafer::TileRegionOp>(
      group.getLoc(), group->getResultTypes(), tileRegionInputs);

  mlir::Block *body = new mlir::Block();
  tileRegion.getBody().push_back(body);

  mlir::IRMapping mapping;
  mlir::Block &groupBlock = group.getBody().front();
  for (auto [oldArg, input] :
       llvm::zip(groupBlock.getArguments(), tileRegionInputs)) {
    mlir::BlockArgument newArg =
        body->addArgument(input.getType(), input.getLoc());
    mapping.map(oldArg, newArg);
  }

  mlir::Value lhs = mapping.lookup(matmul.getInputs()[0]);
  mlir::Value rhs = mapping.lookup(matmul.getInputs()[1]);
  mlir::Value out = mapping.lookup(matmul.getOutputs()[0]);
  mlir::MLIRContext *context = group.getContext();
  mlir::Location loc = group.getLoc();
  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(body);

  auto lhsTensorType =
      getSPMTileBuffer(context, lhsType, wafer::MemLayout::Tensor);
  auto rhsTensorType =
      getSPMTileBuffer(context, rhsType, wafer::MemLayout::Tensor);
  auto lhsCxType = getSPMTileBuffer(context, lhsType, wafer::MemLayout::Cx);
  auto rhsCxType = getSPMTileBuffer(context, rhsType, wafer::MemLayout::Cx);
  auto resultCxType =
      getSPMTileBuffer(context, resultType, wafer::MemLayout::Cx);
  auto resultTensorType =
      getSPMTileBuffer(context, resultType, wafer::MemLayout::Tensor);

  auto lhsTile = bodyBuilder.create<wafer::LoadTileOp>(loc, lhsTensorType, lhs);
  auto rhsTile = bodyBuilder.create<wafer::LoadTileOp>(loc, rhsTensorType, rhs);
  auto lhsCx = bodyBuilder.create<wafer::LayoutMaterializeOp>(
      loc, lhsCxType, lhsTile.getResult());
  auto rhsCx = bodyBuilder.create<wafer::LayoutMaterializeOp>(
      loc, rhsCxType, rhsTile.getResult());
  auto gemm = bodyBuilder.create<wafer::ComputeGemmOp>(
      loc, resultCxType, lhsCx.getResult(), rhsCx.getResult());
  auto resultTensor = bodyBuilder.create<wafer::LayoutMaterializeOp>(
      loc, resultTensorType, gemm.getResult());
  bodyBuilder.create<wafer::StoreTileOp>(loc, resultTensor.getResult(), out);
  bodyBuilder.create<wafer::TileYieldOp>(loc, out);

  group->replaceAllUsesWith(tileRegion->getResults());
  group.erase();
  return mlir::success();
}

static mlir::LogicalResult
materializeReduceGroup(wafer::GroupOp group, mlir::linalg::ReduceOp reduce) {
  if (group.getNumResults() != 1 || group.getOuts().size() != 1)
    return group.emitOpError(
        "cannot materialize single tile: expected one group result and out");
  if (reduce->getNumResults() != 1 || reduce.getInputs().size() != 1 ||
      reduce.getInits().size() != 1)
    return reduce.emitOpError(
        "cannot materialize single tile: expected tensor reduce with one "
        "input, one out, and one result");

  std::optional<wafer::ComputeReduceKind> kind = mapReduceKind(reduce);
  if (!kind)
    return reduce.emitOpError(
        "cannot materialize single tile: unsupported reduce kind");

  std::optional<mlir::Attribute> initValue =
      getReduceInitValueAttr(group.getOuts()[0]);
  if (!initValue)
    return reduce.emitOpError(
        "cannot materialize single tile: reduce out must be a scalar-constant "
        "linalg.fill");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(reduce.getInputs()[0].getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(reduce->getResult(0).getType());
  if (!inputType || !resultType)
    return reduce.emitOpError(
        "cannot materialize single tile: reduce tensors must be ranked");
  if (inputType.getElementType() != resultType.getElementType())
    return reduce.emitOpError(
        "cannot materialize single tile: reduce input element type must match "
        "result type");
  if (reduce.getInits()[0].getType() != resultType)
    return reduce.emitOpError(
        "cannot materialize single tile: reduce out tensor type must match "
        "result type");

  llvm::DenseSet<int64_t> reducedDims;
  for (int64_t dim : reduce.getDimensions()) {
    if (dim < 0 || dim >= inputType.getRank())
      return reduce.emitOpError(
          "cannot materialize single tile: reduce dimension is out of range");
    if (!reducedDims.insert(dim).second)
      return reduce.emitOpError(
          "cannot materialize single tile: reduce dimensions must be unique");
  }
  if (resultType.getRank() !=
      inputType.getRank() - static_cast<int64_t>(reducedDims.size()))
    return reduce.emitOpError(
        "cannot materialize single tile: reduce result rank must match input "
        "rank minus reduce dimensions");
  int64_t resultDim = 0;
  for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
    if (reducedDims.contains(inputDim))
      continue;
    if (hasStaticMismatch(inputType.getDimSize(inputDim),
                          resultType.getDimSize(resultDim)))
      return reduce.emitOpError(
          "cannot materialize single tile: reduce result shape must match "
          "non-reduced input dimensions");
    ++resultDim;
  }

  llvm::SmallVector<mlir::Value> tileRegionInputs(group.getInputs().begin(),
                                                  group.getInputs().end());
  tileRegionInputs.append(group.getOuts().begin(), group.getOuts().end());

  mlir::OpBuilder builder(group);
  auto tileRegion = builder.create<wafer::TileRegionOp>(
      group.getLoc(), group->getResultTypes(), tileRegionInputs);

  mlir::Block *body = new mlir::Block();
  tileRegion.getBody().push_back(body);

  mlir::IRMapping mapping;
  mlir::Block &groupBlock = group.getBody().front();
  for (auto [oldArg, input] :
       llvm::zip(groupBlock.getArguments(), tileRegionInputs)) {
    mlir::BlockArgument newArg =
        body->addArgument(input.getType(), input.getLoc());
    mapping.map(oldArg, newArg);
  }

  mlir::Value input = mapping.lookup(reduce.getInputs()[0]);
  mlir::Value out = mapping.lookup(reduce.getInits()[0]);
  mlir::MLIRContext *context = group.getContext();
  mlir::Location loc = group.getLoc();
  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(body);
  auto inputTileType =
      getSPMTileBuffer(context, inputType, wafer::MemLayout::Tensor);
  auto resultTileType =
      getSPMTileBuffer(context, resultType, wafer::MemLayout::Tensor);

  auto inputTile =
      bodyBuilder.create<wafer::LoadTileOp>(loc, inputTileType, input);
  auto compute = bodyBuilder.create<wafer::ComputeReduceOp>(
      loc, resultTileType, wafer::ComputeReduceKindAttr::get(context, *kind),
      inputTile.getResult());
  compute->setAttr("dimensions",
                   bodyBuilder.getDenseI64ArrayAttr(reduce.getDimensions()));
  compute->setAttr("init_value", *initValue);
  bodyBuilder.create<wafer::StoreTileOp>(loc, compute.getResult(), out);
  bodyBuilder.create<wafer::TileYieldOp>(loc, out);

  group->replaceAllUsesWith(tileRegion->getResults());
  group.erase();
  return mlir::success();
}

static mlir::LogicalResult
materializeElementwiseGroup(wafer::GroupOp group,
                            mlir::linalg::GenericOp elementwise) {
  if (group.getNumResults() != 1 || group.getOuts().size() != 1)
    return group.emitOpError(
        "cannot materialize single tile: expected one group result and out");
  if (elementwise->getNumResults() != 1 ||
      elementwise.getDpsInits().size() != 1)
    return elementwise.emitOpError(
        "cannot materialize single tile: expected tensor elementwise with one "
        "out and one result");

  std::optional<wafer::ComputeElementwiseKind> kind =
      matchElementwiseGeneric(elementwise);
  if (!kind)
    return elementwise.emitOpError(
        "cannot materialize single tile: unsupported elementwise kind");

  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      elementwise->getResult(0).getType());
  if (!resultType)
    return elementwise.emitOpError(
        "cannot materialize single tile: elementwise result must be ranked");

  llvm::SmallVector<mlir::AffineMap> maps = elementwise.getIndexingMapsArray();
  if (maps.size() !=
      elementwise.getDpsInputs().size() + elementwise.getDpsInits().size())
    return elementwise.emitOpError(
        "cannot materialize single tile: elementwise indexing map count must "
        "match inputs plus outputs");
  mlir::AffineMap resultMap = maps.back();
  if (resultMap.getNumDims() != resultType.getRank() ||
      resultMap.getNumSymbols() != 0 || !resultMap.isIdentity())
    return elementwise.emitOpError(
        "cannot materialize single tile: result indexing map must be identity");

  for (auto [index, input] : llvm::enumerate(elementwise.getDpsInputs())) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || inputType.getElementType() != resultType.getElementType())
      return elementwise.emitOpError(
          "cannot materialize single tile: elementwise input tensor element "
          "types must match result type");

    mlir::AffineMap inputMap = maps[index];
    if (inputMap.getNumDims() != resultType.getRank() ||
        inputMap.getNumSymbols() != 0 ||
        inputMap.getNumResults() != inputType.getRank() ||
        !inputMap.isProjectedPermutation())
      return elementwise.emitOpError(
          "cannot materialize single tile: input indexing maps must be "
          "projected permutations");

    for (auto [dim, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr || dimExpr.getPosition() >= resultType.getRank() ||
          hasStaticMismatch(inputType.getDimSize(dim),
                            resultType.getDimSize(dimExpr.getPosition())))
        return elementwise.emitOpError(
            "cannot materialize single tile: elementwise indexing map "
            "dimension must match tensor shape");
    }
  }
  if (elementwise.getDpsInits()[0].getType() != resultType)
    return elementwise.emitOpError(
        "cannot materialize single tile: elementwise out tensor type must "
        "match result type");

  llvm::SmallVector<mlir::Value> tileRegionInputs(group.getInputs().begin(),
                                                  group.getInputs().end());
  tileRegionInputs.append(group.getOuts().begin(), group.getOuts().end());

  mlir::OpBuilder builder(group);
  auto tileRegion = builder.create<wafer::TileRegionOp>(
      group.getLoc(), group->getResultTypes(), tileRegionInputs);

  mlir::Block *body = new mlir::Block();
  tileRegion.getBody().push_back(body);

  mlir::IRMapping mapping;
  mlir::Block &groupBlock = group.getBody().front();
  for (auto [oldArg, input] :
       llvm::zip(groupBlock.getArguments(), tileRegionInputs)) {
    mlir::BlockArgument newArg =
        body->addArgument(input.getType(), input.getLoc());
    mapping.map(oldArg, newArg);
  }

  mlir::Value out = mapping.lookup(elementwise.getDpsInits()[0]);
  mlir::MLIRContext *context = group.getContext();
  mlir::Location loc = group.getLoc();
  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(body);
  auto tileType =
      getSPMTileBuffer(context, resultType, wafer::MemLayout::Tensor);

  llvm::SmallVector<mlir::Value> inputTiles;
  for (mlir::Value input : elementwise.getDpsInputs()) {
    mlir::Value mappedInput = mapping.lookup(input);
    auto inputType = mlir::cast<mlir::RankedTensorType>(mappedInput.getType());
    auto inputTileType =
        getSPMTileBuffer(context, inputType, wafer::MemLayout::Tensor);
    auto tile =
        bodyBuilder.create<wafer::LoadTileOp>(loc, inputTileType, mappedInput);
    inputTiles.push_back(tile.getResult());
  }

  auto compute = bodyBuilder.create<wafer::ComputeElementwiseOp>(
      loc, tileType, wafer::ComputeElementwiseKindAttr::get(context, *kind),
      inputTiles);
  compute->setAttr("indexing_maps", elementwise.getIndexingMaps());
  bodyBuilder.create<wafer::StoreTileOp>(loc, compute.getResult(), out);
  bodyBuilder.create<wafer::TileYieldOp>(loc, out);

  group->replaceAllUsesWith(tileRegion->getResults());
  group.erase();
  return mlir::success();
}

static mlir::LogicalResult materializeGroup(wafer::GroupOp group) {
  if (mlir::linalg::MatmulOp matmul = getSingleMatmulBody(group))
    return materializeMatmulGroup(group, matmul);
  if (mlir::linalg::GenericOp generic = getSingleAttentionGemmBody(group))
    return materializeAttentionGemmGroup(group, generic);
  if (mlir::linalg::ReduceOp reduce = getSingleReduceBody(group))
    return materializeReduceGroup(group, reduce);
  if (mlir::linalg::GenericOp elementwise = getSingleElementwiseBody(group))
    return materializeElementwiseGroup(group, elementwise);
  return group.emitOpError(
      "cannot materialize single tile: expected one linalg.matmul, "
      "supported attention linalg.generic contraction, linalg.reduce, or "
      "elementwise linalg.generic body op");
}

struct MaterializeSingleTilePass
    : public mlir::PassWrapper<MaterializeSingleTilePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeSingleTilePass)

  llvm::StringRef getArgument() const final {
    return "wafer-materialize-single-tile";
  }

  llvm::StringRef getDescription() const final {
    return "materialize Wafer groups into single-tile tile_region ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, wafer::WaferDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<wafer::GroupOp> groups;
    getOperation().walk([&](wafer::GroupOp group) { groups.push_back(group); });

    for (wafer::GroupOp group : groups) {
      if (mlir::failed(materializeGroup(group))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createMaterializeSingleTilePass() {
  return std::make_unique<MaterializeSingleTilePass>();
}

} // namespace wafer
