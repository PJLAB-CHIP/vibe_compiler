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

static mlir::LogicalResult materializeGroup(wafer::GroupOp group) {
  mlir::linalg::MatmulOp matmul = getSingleMatmulBody(group);
  if (!matmul)
    return group.emitOpError(
        "cannot materialize single tile: expected one linalg.matmul body op");
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
