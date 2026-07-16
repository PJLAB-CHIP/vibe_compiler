//===- TilingDemandAnalysis.cpp - Structured scheduling tiling demand ----===//

#include "Wafer/Analysis/Scheduling/TilingDemandAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace {

static bool isRankedTensor(mlir::Type type) {
  return mlir::isa<mlir::RankedTensorType>(type);
}

static TilingDemandSlice fullRankSlice(mlir::Type type) {
  TilingDemandSlice slice;
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    for (int64_t dim = 0; dim < tensorType.getRank(); ++dim)
      slice.loopDims.push_back(static_cast<unsigned>(dim));
  }
  return slice;
}

static void addBoundaryValue(llvm::SmallVectorImpl<TilingDemandValue> &values,
                             TilingDemandValueRole role, unsigned index,
                             mlir::Value value) {
  values.push_back(
      {role, index, value, value.getType(), fullRankSlice(value.getType())});
}

static bool isSupportOp(mlir::Operation *op) {
  return mlir::isa<mlir::arith::ConstantOp, mlir::tensor::EmptyOp,
                   mlir::tensor::ExtractOp, mlir::tensor::ExtractSliceOp,
                   mlir::tensor::InsertSliceOp, mlir::tensor::ExpandShapeOp,
                   mlir::tensor::CollapseShapeOp, mlir::scf::IfOp,
                   mlir::scf::ForOp>(op);
}

static bool isReductionIterator(mlir::utils::IteratorType iteratorType) {
  return iteratorType == mlir::utils::IteratorType::reduction;
}

static mlir::LogicalResult
sliceFromIndexingMap(mlir::AffineMap map,
                     llvm::SmallVectorImpl<unsigned> &loopDims,
                     std::string &failureReason) {
  loopDims.clear();
  for (mlir::AffineExpr expr : map.getResults()) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr) {
      failureReason = "unsupported indexing map ";
      llvm::raw_string_ostream os(failureReason);
      map.print(os);
      return mlir::failure();
    }
    loopDims.push_back(dimExpr.getPosition());
  }
  return mlir::success();
}

static mlir::LogicalResult
addMappedValue(llvm::SmallVectorImpl<TilingDemandValue> &values,
               TilingDemandValueRole role, unsigned index, mlir::Value value,
               mlir::AffineMap map, std::string &failureReason) {
  TilingDemandValue demand{role, index, value, value.getType(), {}};
  if (mlir::failed(
          sliceFromIndexingMap(map, demand.slice.loopDims, failureReason)))
    return mlir::failure();
  values.push_back(demand);
  return mlir::success();
}

static mlir::LogicalResult collectLinalgDemand(mlir::linalg::LinalgOp op,
                                               unsigned opIndex,
                                               OpTilingDemand &demand) {
  demand = {};
  demand.kind = OpTilingDemandKind::Linalg;
  demand.op = op.getOperation();
  demand.opIndex = opIndex;
  demand.iteratorTypes = op.getIteratorTypesArray();

  if (!op.hasOnlyProjectedPermutations()) {
    demand.kind = OpTilingDemandKind::Failure;
    demand.failureReason = "unsupported linalg indexing maps";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      op.getIndexingMapsArray();
  unsigned expectedMapCount =
      static_cast<unsigned>(op.getNumDpsInputs() + op.getNumDpsInits());
  if (indexingMaps.size() != expectedMapCount) {
    demand.kind = OpTilingDemandKind::Failure;
    demand.failureReason =
        "linalg indexing map count does not match DPS operands";
    return mlir::failure();
  }

  for (auto [index, operand] : llvm::enumerate(op.getDpsInputs())) {
    if (!isRankedTensor(operand.getType()) &&
        !mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
            operand.getType())) {
      demand.kind = OpTilingDemandKind::Failure;
      demand.failureReason = "linalg input is not ranked tensor or scalar";
      return mlir::failure();
    }
    if (mlir::failed(addMappedValue(demand.values, TilingDemandValueRole::Input,
                                    static_cast<unsigned>(index), operand,
                                    indexingMaps[index],
                                    demand.failureReason))) {
      demand.kind = OpTilingDemandKind::Failure;
      return mlir::failure();
    }
  }

  unsigned outputMapBase = static_cast<unsigned>(op.getNumDpsInputs());
  for (auto [index, output] : llvm::enumerate(op.getDpsInits())) {
    if (!isRankedTensor(output.getType())) {
      demand.kind = OpTilingDemandKind::Failure;
      demand.failureReason = "linalg output is not ranked tensor";
      return mlir::failure();
    }
    if (mlir::failed(addMappedValue(
            demand.values, TilingDemandValueRole::Output,
            static_cast<unsigned>(index), output,
            indexingMaps[outputMapBase + index], demand.failureReason))) {
      demand.kind = OpTilingDemandKind::Failure;
      return mlir::failure();
    }
  }

  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    if (!isRankedTensor(result.getType())) {
      demand.kind = OpTilingDemandKind::Failure;
      demand.failureReason = "linalg result is not ranked tensor";
      return mlir::failure();
    }
    if (index >= op.getNumDpsInits()) {
      demand.kind = OpTilingDemandKind::Failure;
      demand.failureReason = "linalg result has no tied DPS output";
      return mlir::failure();
    }
    if (mlir::failed(addMappedValue(
            demand.values, TilingDemandValueRole::Result,
            static_cast<unsigned>(index), result,
            indexingMaps[outputMapBase + index], demand.failureReason))) {
      demand.kind = OpTilingDemandKind::Failure;
      return mlir::failure();
    }
  }

  llvm::SmallVector<unsigned, 4> reductionDims;
  for (auto [index, iteratorType] : llvm::enumerate(demand.iteratorTypes)) {
    if (isReductionIterator(iteratorType))
      reductionDims.push_back(static_cast<unsigned>(index));
  }
  if (!reductionDims.empty()) {
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      (void)result;
      demand.accumulators.push_back(
          {static_cast<unsigned>(index), reductionDims});
    }
  }

  return mlir::success();
}

static mlir::LogicalResult
collectCollectiveDemand(WaferLinalgExtCollectiveOpInterface op,
                        unsigned opIndex, OpTilingDemand &demand) {
  demand = {};
  demand.kind = OpTilingDemandKind::LinalgExtCollective;
  demand.op = op.getOperation();
  demand.opIndex = opIndex;
  op.collectWaferLinalgExtCollectiveInfo(demand.collectiveInfo);

  auto mlirTiling = mlir::dyn_cast<mlir::TilingInterface>(op.getOperation());
  if (!mlirTiling) {
    demand.kind = OpTilingDemandKind::Failure;
    demand.failureReason = "collective does not implement MLIR TilingInterface";
    return mlir::failure();
  }
  demand.iteratorTypes = mlirTiling.getLoopIteratorTypes();

  llvm::SmallVector<WaferTilingDemand, 4> tilingDemands;
  if (auto tiling = mlir::dyn_cast<WaferTilingInterface>(op.getOperation()))
    tiling.collectWaferTilingDemand(tilingDemands);
  if (tilingDemands.empty()) {
    demand.kind = OpTilingDemandKind::Failure;
    demand.failureReason = "collective did not expose tiling demand";
    return mlir::failure();
  }

  for (const WaferTilingDemand &tilingDemand : tilingDemands) {
    TilingDemandValueRole role;
    switch (tilingDemand.kind) {
    case WaferTilingDemandKind::Input:
      role = TilingDemandValueRole::Input;
      break;
    case WaferTilingDemandKind::Output:
      role = TilingDemandValueRole::Output;
      break;
    case WaferTilingDemandKind::Result:
      role = TilingDemandValueRole::Result;
      break;
    }

    mlir::Value value;
    if (role == TilingDemandValueRole::Input &&
        tilingDemand.index < op.getOperation()->getNumOperands())
      value = op.getOperation()->getOperand(tilingDemand.index);
    if (role == TilingDemandValueRole::Output) {
      auto dpsOp =
          mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op.getOperation());
      if (dpsOp && tilingDemand.index < dpsOp.getNumDpsInits())
        value = dpsOp.getDpsInits()[tilingDemand.index];
    }
    if (role == TilingDemandValueRole::Result &&
        tilingDemand.index < op.getOperation()->getNumResults())
      value = op.getOperation()->getResult(tilingDemand.index);

    if (!value) {
      demand.kind = OpTilingDemandKind::Failure;
      demand.failureReason =
          "collective tiling demand references missing value";
      return mlir::failure();
    }
    if (!isRankedTensor(value.getType())) {
      demand.kind = OpTilingDemandKind::Failure;
      demand.failureReason = "collective demand is not ranked tensor";
      return mlir::failure();
    }
    addBoundaryValue(demand.values, role, tilingDemand.index, value);
  }

  return mlir::success();
}

static void collectSupportDemand(mlir::Operation *op, unsigned opIndex,
                                 OpTilingDemand &demand) {
  demand = {};
  demand.kind = OpTilingDemandKind::Support;
  demand.op = op;
  demand.opIndex = opIndex;
}

static void collectUnsupportedDemand(mlir::Operation *op, unsigned opIndex,
                                     OpTilingDemand &demand) {
  demand = {};
  demand.kind = OpTilingDemandKind::Failure;
  demand.op = op;
  demand.opIndex = opIndex;
  demand.failureReason = "unsupported op ";
  demand.failureReason += op->getName().getStringRef().str();
}

} // namespace

mlir::LogicalResult wafer::collectStructuredSchedulingTilingDemand(
    mlir::func::FuncOp function, unsigned inputCount,
    StructuredSchedulingTilingDemand &demand) {
  demand = {};
  demand.function = function;
  demand.inputCount = inputCount;

  if (function.isExternal() || !llvm::hasSingleElement(function.getBody()) ||
      inputCount > function.getNumArguments())
    return mlir::failure();

  mlir::Block &block = function.getBody().front();
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
  if (!returnOp)
    return mlir::failure();

  for (auto [index, argument] : llvm::enumerate(block.getArguments())) {
    TilingDemandValueRole role = index < inputCount
                                     ? TilingDemandValueRole::Input
                                     : TilingDemandValueRole::Output;
    unsigned roleIndex = index < inputCount
                             ? static_cast<unsigned>(index)
                             : static_cast<unsigned>(index - inputCount);
    addBoundaryValue(demand.boundaryValues, role, roleIndex, argument);
  }
  for (auto [index, result] : llvm::enumerate(returnOp.getOperands())) {
    addBoundaryValue(demand.boundaryValues, TilingDemandValueRole::Result,
                     static_cast<unsigned>(index), result);
    addBoundaryValue(demand.resultTiles, TilingDemandValueRole::Result,
                     static_cast<unsigned>(index), result);
  }

  unsigned opIndex = 0;
  for (mlir::Operation &op : block.without_terminator()) {
    OpTilingDemand opDemand;
    mlir::LogicalResult result = mlir::success();
    if (auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(&op))
      result = collectLinalgDemand(linalgOp, opIndex, opDemand);
    else if (auto collective =
                 mlir::dyn_cast<WaferLinalgExtCollectiveOpInterface>(&op))
      result = collectCollectiveDemand(collective, opIndex, opDemand);
    else if (isSupportOp(&op))
      collectSupportDemand(&op, opIndex, opDemand);
    else {
      collectUnsupportedDemand(&op, opIndex, opDemand);
      result = mlir::failure();
    }

    if (mlir::failed(result)) {
      demand.succeeded = false;
      demand.failureReason = opDemand.failureReason;
      demand.ops.push_back(std::move(opDemand));
      return mlir::success();
    }

    demand.ops.push_back(std::move(opDemand));
    ++opIndex;
  }

  return mlir::success();
}
