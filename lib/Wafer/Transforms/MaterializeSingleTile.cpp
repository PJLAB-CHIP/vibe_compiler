//===- MaterializeSingleTile.cpp - Lower M0 group to tile_region ----------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

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

static mlir::linalg::ElementwiseOp
getSingleElementwiseBody(wafer::GroupOp group) {
  mlir::linalg::ElementwiseOp elementwise;
  mlir::Block &block = group.getBody().front();
  for (mlir::Operation &op : block) {
    if (&op == block.getTerminator())
      continue;
    auto candidate = mlir::dyn_cast<mlir::linalg::ElementwiseOp>(op);
    if (!candidate || elementwise)
      return {};
    elementwise = candidate;
  }
  return elementwise;
}

static std::optional<wafer::ComputeElementwiseKind>
mapElementwiseKind(mlir::linalg::ElementwiseKind kind) {
  switch (kind) {
  case mlir::linalg::ElementwiseKind::add:
    return wafer::ComputeElementwiseKind::Add;
  case mlir::linalg::ElementwiseKind::sub:
    return wafer::ComputeElementwiseKind::Sub;
  case mlir::linalg::ElementwiseKind::mul:
    return wafer::ComputeElementwiseKind::Mul;
  case mlir::linalg::ElementwiseKind::div:
    return wafer::ComputeElementwiseKind::Div;
  case mlir::linalg::ElementwiseKind::max_signed:
    return wafer::ComputeElementwiseKind::Max;
  case mlir::linalg::ElementwiseKind::min_signed:
    return wafer::ComputeElementwiseKind::Min;
  case mlir::linalg::ElementwiseKind::negf:
    return wafer::ComputeElementwiseKind::Neg;
  case mlir::linalg::ElementwiseKind::reciprocal:
    return wafer::ComputeElementwiseKind::Recip;
  case mlir::linalg::ElementwiseKind::sqrt:
    return wafer::ComputeElementwiseKind::Sqrt;
  case mlir::linalg::ElementwiseKind::rsqrt:
    return wafer::ComputeElementwiseKind::Rsqrt;
  case mlir::linalg::ElementwiseKind::exp:
    return wafer::ComputeElementwiseKind::Exp;
  case mlir::linalg::ElementwiseKind::tanh:
    return wafer::ComputeElementwiseKind::Tanh;
  default:
    return std::nullopt;
  }
}

static mlir::LogicalResult materializeMatmulGroup(wafer::GroupOp group,
                                                  mlir::linalg::MatmulOp matmul) {
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
materializeElementwiseGroup(wafer::GroupOp group,
                            mlir::linalg::ElementwiseOp elementwise) {
  if (group.getNumResults() != 1 || group.getOuts().size() != 1)
    return group.emitOpError(
        "cannot materialize single tile: expected one group result and out");
  if (elementwise->getNumResults() != 1 || elementwise.getOutputs().size() != 1)
    return elementwise.emitOpError(
        "cannot materialize single tile: expected tensor elementwise with one "
        "out and one result");
  for (mlir::AffineMap map : elementwise.getIndexingMapsArray()) {
    if (!map.isIdentity())
      return elementwise.emitOpError(
          "cannot materialize single tile: elementwise indexing maps require "
          "explicit broadcast/layout materialization");
  }
  if (elementwise.getIndexingMapsArray().size() !=
      elementwise.getInputs().size() + elementwise.getOutputs().size())
    return elementwise.emitOpError(
        "cannot materialize single tile: elementwise indexing map count must "
        "match inputs plus outputs");

  std::optional<wafer::ComputeElementwiseKind> kind =
      mapElementwiseKind(elementwise.getKind());
  if (!kind)
    return elementwise.emitOpError(
        "cannot materialize single tile: unsupported elementwise kind");

  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      elementwise->getResult(0).getType());
  if (!resultType)
    return elementwise.emitOpError(
        "cannot materialize single tile: elementwise result must be ranked");

  for (mlir::Value input : elementwise.getInputs()) {
    if (input.getType() != resultType)
      return elementwise.emitOpError(
          "cannot materialize single tile: elementwise input tensor types must "
          "match result type");
  }
  if (elementwise.getOutputs()[0].getType() != resultType)
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

  mlir::Value out = mapping.lookup(elementwise.getOutputs()[0]);
  mlir::MLIRContext *context = group.getContext();
  mlir::Location loc = group.getLoc();
  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(body);
  auto tileType = getSPMTileBuffer(context, resultType, wafer::MemLayout::Tensor);

  llvm::SmallVector<mlir::Value> inputTiles;
  for (mlir::Value input : elementwise.getInputs()) {
    mlir::Value mappedInput = mapping.lookup(input);
    auto tile = bodyBuilder.create<wafer::LoadTileOp>(loc, tileType, mappedInput);
    inputTiles.push_back(tile.getResult());
  }

  auto compute = bodyBuilder.create<wafer::ComputeElementwiseOp>(
      loc, tileType, wafer::ComputeElementwiseKindAttr::get(context, *kind),
      inputTiles);
  bodyBuilder.create<wafer::StoreTileOp>(loc, compute.getResult(), out);
  bodyBuilder.create<wafer::TileYieldOp>(loc, out);

  group->replaceAllUsesWith(tileRegion->getResults());
  group.erase();
  return mlir::success();
}

static mlir::LogicalResult materializeGroup(wafer::GroupOp group) {
  if (mlir::linalg::MatmulOp matmul = getSingleMatmulBody(group))
    return materializeMatmulGroup(group, matmul);
  if (mlir::linalg::ElementwiseOp elementwise = getSingleElementwiseBody(group))
    return materializeElementwiseGroup(group, elementwise);
  return group.emitOpError(
      "cannot materialize single tile: expected one linalg.matmul or "
      "linalg.elementwise body op");
}

struct MaterializeSingleTilePass
    : public mlir::PassWrapper<MaterializeSingleTilePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeSingleTilePass)

  llvm::StringRef getArgument() const final {
    return "wafer-materialize-single-tile";
  }

  llvm::StringRef getDescription() const final {
    return "materialize M0 Wafer groups into single tile_region skeletons";
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
