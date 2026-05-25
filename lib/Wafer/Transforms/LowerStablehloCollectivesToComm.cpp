//===- LowerStablehloCollectivesToComm.cpp - StableHLO comm lowering -----===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <limits>
#include <optional>

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
namespace {

static wafer::TileBufferType getSPMTileBuffer(mlir::MLIRContext *context,
                                              mlir::RankedTensorType tensorType,
                                              wafer::MemLayout layout) {
  return wafer::TileBufferType::get(
      context, tensorType, wafer::MemLayoutAttr::get(context, layout),
      wafer::MemorySpaceAttr::get(context, wafer::MemorySpace::SPM));
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static std::optional<int64_t> getElementBitWidth(mlir::Type elementType) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
    return floatType.getWidth();
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
    return integerType.getWidth();
  if (mlir::isa<mlir::IndexType>(elementType))
    return 64;
  return std::nullopt;
}

static std::optional<int64_t>
getCompactTensorByteSize(mlir::RankedTensorType tensorType) {
  if (!tensorType.hasStaticShape())
    return std::nullopt;

  int64_t elements = 1;
  for (int64_t dim : tensorType.getShape()) {
    int64_t next = 0;
    if (!checkedMul(elements, dim, next))
      return std::nullopt;
    elements = next;
  }

  std::optional<int64_t> elementBits =
      getElementBitWidth(tensorType.getElementType());
  if (!elementBits || *elementBits <= 0)
    return std::nullopt;

  int64_t totalBits = 0;
  if (!checkedMul(elements, *elementBits, totalBits))
    return std::nullopt;
  return totalBits / 8 + (totalBits % 8 == 0 ? 0 : 1);
}

#ifdef WAFER_ENABLE_STABLEHLO
static mlir::FailureOr<int64_t>
getReplicaGroupSize(mlir::Operation *op,
                    mlir::DenseIntElementsAttr replicaGroups) {
  auto groupsType =
      mlir::dyn_cast<mlir::RankedTensorType>(replicaGroups.getType());
  if (!groupsType || groupsType.getRank() != 2)
    return op->emitOpError(
        "requires rank-2 StableHLO replica_groups for Wafer comm lowering");
  int64_t groupSize = groupsType.getDimSize(1);
  if (groupSize <= 1)
    return op->emitOpError(
        "requires StableHLO replica group size greater than one");
  return groupSize;
}

static mlir::LogicalResult
verifyLocalRank(mlir::Operation *op, int64_t localRank, int64_t groupSize) {
  if (localRank < 0 || localRank >= groupSize)
    return op->emitOpError("local-rank pass option must be within "
                           "StableHLO replica group size");
  return mlir::success();
}

static mlir::Value createTensorToTileCast(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value tensor,
                                          mlir::RankedTensorType tensorType) {
  auto tileType = getSPMTileBuffer(builder.getContext(), tensorType,
                                   wafer::MemLayout::Tensor);
  return builder.create<mlir::UnrealizedConversionCastOp>(loc, tileType, tensor)
      .getResult(0);
}

static mlir::Value createEmptyTileCast(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::RankedTensorType tensorType) {
  auto tileType = getSPMTileBuffer(builder.getContext(), tensorType,
                                   wafer::MemLayout::Tensor);
  return builder
      .create<mlir::UnrealizedConversionCastOp>(loc, tileType,
                                                mlir::ValueRange{})
      .getResult(0);
}

static mlir::Value createTileToTensorCast(mlir::OpBuilder &builder,
                                          mlir::Location loc, mlir::Value tile,
                                          mlir::Type tensorType) {
  return builder.create<mlir::UnrealizedConversionCastOp>(loc, tensorType, tile)
      .getResult(0);
}

static std::optional<wafer::ComputeReduceKind>
getStablehloReduceKind(mlir::Region &computation) {
  if (!computation.hasOneBlock())
    return std::nullopt;
  mlir::Block &block = computation.front();
  if (block.getOperations().size() != 2)
    return std::nullopt;

  mlir::Operation &combine = block.front();
  auto ret = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret || ret.getResults().size() != 1 || combine.getNumResults() != 1 ||
      ret.getResults().front() != combine.getResult(0))
    return std::nullopt;

  if (mlir::isa<mlir::stablehlo::AddOp>(combine))
    return wafer::ComputeReduceKind::Sum;
  if (mlir::isa<mlir::stablehlo::MaxOp>(combine))
    return wafer::ComputeReduceKind::Max;
  if (mlir::isa<mlir::stablehlo::MinOp>(combine))
    return wafer::ComputeReduceKind::Min;
  return std::nullopt;
}

static mlir::LogicalResult lowerAllGather(mlir::stablehlo::AllGatherOp op,
                                          int64_t localRank) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return op->emitOpError(
        "only single-result StableHLO all_gather is supported");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !resultType || !inputType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return op->emitOpError(
        "requires static ranked tensor types for Wafer comm lowering");

  mlir::FailureOr<int64_t> groupSize =
      getReplicaGroupSize(op.getOperation(), op.getReplicaGroups());
  if (mlir::failed(groupSize) ||
      mlir::failed(verifyLocalRank(op.getOperation(), localRank, *groupSize)))
    return mlir::failure();

  std::optional<int64_t> bytes = getCompactTensorByteSize(inputType);
  std::optional<int64_t> resultBytes = getCompactTensorByteSize(resultType);
  int64_t expectedResultBytes = 0;
  if (!bytes || !resultBytes ||
      !checkedMul(*bytes, *groupSize, expectedResultBytes) ||
      *resultBytes != expectedResultBytes)
    return op->emitOpError(
        "StableHLO all_gather result bytes must equal operand bytes times "
        "group size");

  mlir::OpBuilder builder(op);
  mlir::Value localTile = createTensorToTileCast(builder, op.getLoc(),
                                                 op->getOperand(0), inputType);
  mlir::Value gatherTile =
      createEmptyTileCast(builder, op.getLoc(), resultType);
  builder.create<wafer::CommAllGatherOp>(
      op.getLoc(), localTile, gatherTile, builder.getI64IntegerAttr(localRank),
      builder.getI64IntegerAttr(*groupSize), builder.getI64IntegerAttr(*bytes));
  mlir::Value result =
      createTileToTensorCast(builder, op.getLoc(), gatherTile, resultType);
  op->getResult(0).replaceAllUsesWith(result);
  op.erase();
  return mlir::success();
}

static mlir::LogicalResult lowerAllReduce(mlir::stablehlo::AllReduceOp op,
                                          int64_t localRank) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return op->emitOpError(
        "only single-result StableHLO all_reduce is supported");

  std::optional<wafer::ComputeReduceKind> kind =
      getStablehloReduceKind(op.getComputation());
  if (!kind)
    return op->emitOpError(
        "only sum/max/min StableHLO collective reductions are supported");

  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!tensorType || tensorType != op->getResult(0).getType() ||
      !tensorType.hasStaticShape())
    return op->emitOpError(
        "requires same static ranked operand/result tensor type");

  mlir::FailureOr<int64_t> groupSize =
      getReplicaGroupSize(op.getOperation(), op.getReplicaGroups());
  if (mlir::failed(groupSize) ||
      mlir::failed(verifyLocalRank(op.getOperation(), localRank, *groupSize)))
    return mlir::failure();

  std::optional<int64_t> bytes = getCompactTensorByteSize(tensorType);
  if (!bytes)
    return op->emitOpError(
        "StableHLO all_reduce byte size is not representable");

  mlir::OpBuilder builder(op);
  mlir::Value inputTile = createTensorToTileCast(builder, op.getLoc(),
                                                 op->getOperand(0), tensorType);
  mlir::Value recvTile = createEmptyTileCast(builder, op.getLoc(), tensorType);
  auto comm = builder.create<wafer::CommAllReduceOp>(
      op.getLoc(),
      getSPMTileBuffer(op.getContext(), tensorType, wafer::MemLayout::Tensor),
      wafer::ComputeReduceKindAttr::get(op.getContext(), *kind), inputTile,
      recvTile, builder.getI64IntegerAttr(localRank),
      builder.getI64IntegerAttr(*groupSize), builder.getI64IntegerAttr(*bytes));
  mlir::Value result = createTileToTensorCast(builder, op.getLoc(),
                                              comm.getResult(), tensorType);
  op->getResult(0).replaceAllUsesWith(result);
  op.erase();
  return mlir::success();
}

static mlir::LogicalResult
lowerReduceScatter(mlir::stablehlo::ReduceScatterOp op, int64_t localRank) {
  std::optional<wafer::ComputeReduceKind> kind =
      getStablehloReduceKind(op.getComputation());
  if (!kind)
    return op->emitOpError(
        "only sum/max/min StableHLO collective reductions are supported");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand().getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
  if (!inputType || !resultType || !inputType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return op->emitOpError(
        "requires static ranked tensor types for Wafer comm lowering");

  mlir::FailureOr<int64_t> groupSize =
      getReplicaGroupSize(op.getOperation(), op.getReplicaGroups());
  if (mlir::failed(groupSize) ||
      mlir::failed(verifyLocalRank(op.getOperation(), localRank, *groupSize)))
    return mlir::failure();

  std::optional<int64_t> inputBytes = getCompactTensorByteSize(inputType);
  std::optional<int64_t> resultBytes = getCompactTensorByteSize(resultType);
  int64_t expectedInputBytes = 0;
  if (!inputBytes || !resultBytes ||
      !checkedMul(*resultBytes, *groupSize, expectedInputBytes) ||
      *inputBytes != expectedInputBytes)
    return op->emitOpError(
        "StableHLO reduce_scatter input bytes must equal result bytes times "
        "group size");

  mlir::OpBuilder builder(op);
  mlir::Value inputSlot =
      createTensorToTileCast(builder, op.getLoc(), op.getOperand(), resultType);
  mlir::Value recvTile = createEmptyTileCast(builder, op.getLoc(), resultType);
  auto comm = builder.create<wafer::CommReduceScatterOp>(
      op.getLoc(),
      getSPMTileBuffer(op.getContext(), resultType, wafer::MemLayout::Tensor),
      wafer::ComputeReduceKindAttr::get(op.getContext(), *kind), inputSlot,
      recvTile, builder.getI64IntegerAttr(localRank),
      builder.getI64IntegerAttr(*groupSize),
      builder.getI64IntegerAttr(*resultBytes));
  mlir::Value result = createTileToTensorCast(builder, op.getLoc(),
                                              comm.getResult(), resultType);
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return mlir::success();
}
#endif

struct LowerStablehloCollectivesToCommPass
    : public mlir::PassWrapper<LowerStablehloCollectivesToCommPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  using Base = mlir::PassWrapper<LowerStablehloCollectivesToCommPass,
                                 mlir::OperationPass<mlir::ModuleOp>>;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerStablehloCollectivesToCommPass)

  LowerStablehloCollectivesToCommPass() = default;
  LowerStablehloCollectivesToCommPass(
      const LowerStablehloCollectivesToCommPass &pass)
      : Base(pass) {
    localRank = pass.localRank;
  }

  mlir::Pass::Option<int64_t> localRank{
      *this, "local-rank",
      llvm::cl::desc("local rank index within the StableHLO replica group"),
      llvm::cl::init(0)};

  llvm::StringRef getArgument() const final {
    return "wafer-lower-stablehlo-collectives-to-comm";
  }

  llvm::StringRef getDescription() const final {
    return "lower StableHLO logical collectives to wafer.comm collective ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<wafer::WaferDialect>();
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::Operation *, 4> ops;
    getOperation().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::stablehlo::AllGatherOp, mlir::stablehlo::AllReduceOp,
                    mlir::stablehlo::ReduceScatterOp>(op))
        ops.push_back(op);
    });

    for (mlir::Operation *op : ops) {
      mlir::LogicalResult result = mlir::success();
      if (auto allGather = mlir::dyn_cast<mlir::stablehlo::AllGatherOp>(op))
        result = lowerAllGather(allGather, localRank);
      else if (auto allReduce =
                   mlir::dyn_cast<mlir::stablehlo::AllReduceOp>(op))
        result = lowerAllReduce(allReduce, localRank);
      else if (auto reduceScatter =
                   mlir::dyn_cast<mlir::stablehlo::ReduceScatterOp>(op))
        result = lowerReduceScatter(reduceScatter, localRank);

      if (mlir::failed(result)) {
        signalPassFailure();
        return;
      }
    }
#endif
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerStablehloCollectivesToCommPass() {
  return std::make_unique<LowerStablehloCollectivesToCommPass>();
}

} // namespace wafer
