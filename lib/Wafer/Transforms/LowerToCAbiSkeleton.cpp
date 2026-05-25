//===- LowerToCAbiSkeleton.cpp - Lower tile ops to C ABI skeleton --------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace wafer {
namespace {

static std::optional<mlir::Value>
resolveTileRegionBoundaryValue(wafer::TileRegionOp tileRegion,
                               mlir::Value bodyValue) {
  auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(bodyValue);
  if (!blockArgument)
    return std::nullopt;
  if (blockArgument.getOwner() != &tileRegion.getBody().front())
    return std::nullopt;
  if (blockArgument.getArgNumber() >= tileRegion.getInputs().size())
    return std::nullopt;
  return tileRegion.getInputs()[blockArgument.getArgNumber()];
}

static std::optional<int64_t>
getExternalBindingBytes(mlir::ModuleOp module, mlir::Value value,
                        wafer::DdrBindingKind kind) {
  std::optional<int64_t> bytes;
  module.walk([&](wafer::DdrExternalBindingOp binding) {
    if (binding.getValue() == value && binding.getKindAttr().getValue() == kind)
      bytes = binding.getBytesAttr().getInt();
  });
  return bytes;
}

static wafer::AbiWaitPolicyAttr getIssueOnlyPolicy(mlir::MLIRContext *context) {
  return wafer::AbiWaitPolicyAttr::get(context,
                                       wafer::AbiWaitPolicy::IssueOnly);
}

static mlir::LogicalResult lowerLoadTile(mlir::ModuleOp module,
                                         wafer::LoadTileOp load) {
  auto tileRegion = load->getParentOfType<wafer::TileRegionOp>();
  if (!tileRegion)
    return load.emitOpError(
        "cannot lower to C ABI skeleton outside tile_region");

  std::optional<mlir::Value> boundary =
      resolveTileRegionBoundaryValue(tileRegion, load.getSource());
  if (!boundary)
    return load.emitOpError(
        "cannot lower to C ABI skeleton without DDR external input binding");

  std::optional<int64_t> bytes =
      getExternalBindingBytes(module, *boundary, wafer::DdrBindingKind::Input);
  if (!bytes)
    return load.emitOpError("requires DDR external input binding demand");

  mlir::OpBuilder builder(load);
  auto abi = builder.create<wafer::AbiRdma1DOp>(
      load.getLoc(), load.getResult().getType(),
      getIssueOnlyPolicy(load.getContext()), load.getSource(),
      builder.getI64IntegerAttr(*bytes));
  load.getResult().replaceAllUsesWith(abi.getResult());
  load.erase();
  return mlir::success();
}

static mlir::LogicalResult lowerStoreTile(mlir::ModuleOp module,
                                          wafer::StoreTileOp store) {
  auto tileRegion = store->getParentOfType<wafer::TileRegionOp>();
  if (!tileRegion)
    return store.emitOpError(
        "cannot lower to C ABI skeleton outside tile_region");

  std::optional<mlir::Value> boundary =
      resolveTileRegionBoundaryValue(tileRegion, store.getDest());
  if (!boundary)
    return store.emitOpError(
        "cannot lower to C ABI skeleton without DDR external output binding");

  std::optional<int64_t> bytes =
      getExternalBindingBytes(module, *boundary, wafer::DdrBindingKind::Output);
  if (!bytes)
    return store.emitOpError("requires DDR external output binding demand");

  mlir::OpBuilder builder(store);
  builder.create<wafer::AbiWdma1DOp>(
      store.getLoc(), getIssueOnlyPolicy(store.getContext()), store.getSource(),
      store.getDest(), builder.getI64IntegerAttr(*bytes));
  store.erase();
  return mlir::success();
}

static mlir::LogicalResult lowerComputeGemm(wafer::ComputeGemmOp gemm) {
  auto lhsType = mlir::cast<wafer::TileBufferType>(gemm.getLhs().getType());
  auto rhsType = mlir::cast<wafer::TileBufferType>(gemm.getRhs().getType());
  auto lhsTensor = mlir::cast<mlir::RankedTensorType>(lhsType.getTensorType());
  auto rhsTensor = mlir::cast<mlir::RankedTensorType>(rhsType.getTensorType());

  mlir::OpBuilder builder(gemm);
  auto abi = builder.create<wafer::AbiGemmOp>(
      gemm.getLoc(), gemm.getResult().getType(),
      getIssueOnlyPolicy(gemm.getContext()), gemm.getLhs(), gemm.getRhs(),
      builder.getI64IntegerAttr(lhsTensor.getDimSize(0)),
      builder.getI64IntegerAttr(lhsTensor.getDimSize(1)),
      builder.getI64IntegerAttr(rhsTensor.getDimSize(1)));
  gemm.getResult().replaceAllUsesWith(abi.getResult());
  gemm.erase();
  return mlir::success();
}

static mlir::LogicalResult
lowerComputeElementwise(wafer::ComputeElementwiseOp elementwise) {
  mlir::OpBuilder builder(elementwise);
  auto abi = builder.create<wafer::AbiElementwiseOp>(
      elementwise.getLoc(), elementwise.getResult().getType(),
      getIssueOnlyPolicy(elementwise.getContext()), elementwise.getKindAttr(),
      elementwise.getInputs());
  if (mlir::Attribute indexingMaps = elementwise->getAttr("indexing_maps"))
    abi->setAttr("indexing_maps", indexingMaps);
  elementwise.getResult().replaceAllUsesWith(abi.getResult());
  elementwise.erase();
  return mlir::success();
}

static mlir::LogicalResult lowerComputeReduce(wafer::ComputeReduceOp reduce) {
  mlir::OpBuilder builder(reduce);
  auto abi = builder.create<wafer::AbiReduceOp>(
      reduce.getLoc(), reduce.getResult().getType(),
      getIssueOnlyPolicy(reduce.getContext()), reduce.getKindAttr(),
      reduce.getInput());
  abi->setAttr("dimensions", reduce->getAttr("dimensions"));
  abi->setAttr("init_value", reduce->getAttr("init_value"));
  reduce.getResult().replaceAllUsesWith(abi.getResult());
  reduce.erase();
  return mlir::success();
}

struct LowerToCAbiSkeletonPass
    : public mlir::PassWrapper<LowerToCAbiSkeletonPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToCAbiSkeletonPass)

  llvm::StringRef getArgument() const final {
    return "wafer-lower-to-c-abi-skeleton";
  }

  llvm::StringRef getDescription() const final {
    return "lower Wafer movement and compute ops to C ABI skeleton issue ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<wafer::WaferDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<mlir::Operation *> ops;
    module.walk([&](mlir::Operation *op) {
      if (mlir::isa<wafer::LoadTileOp, wafer::StoreTileOp, wafer::ComputeGemmOp,
                    wafer::ComputeElementwiseOp, wafer::ComputeReduceOp>(op))
        ops.push_back(op);
    });

    for (mlir::Operation *op : ops) {
      mlir::LogicalResult result = mlir::success();
      if (auto load = mlir::dyn_cast<wafer::LoadTileOp>(op))
        result = lowerLoadTile(module, load);
      else if (auto store = mlir::dyn_cast<wafer::StoreTileOp>(op))
        result = lowerStoreTile(module, store);
      else if (auto gemm = mlir::dyn_cast<wafer::ComputeGemmOp>(op))
        result = lowerComputeGemm(gemm);
      else if (auto elementwise =
                   mlir::dyn_cast<wafer::ComputeElementwiseOp>(op))
        result = lowerComputeElementwise(elementwise);
      else if (auto reduce = mlir::dyn_cast<wafer::ComputeReduceOp>(op))
        result = lowerComputeReduce(reduce);

      if (mlir::failed(result)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerToCAbiSkeletonPass() {
  return std::make_unique<LowerToCAbiSkeletonPass>();
}

} // namespace wafer
