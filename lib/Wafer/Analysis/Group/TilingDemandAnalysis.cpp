//===- TilingDemandAnalysis.cpp - Wafer group tiling demand analysis ------===//

#include "Wafer/Analysis/Group/TilingDemandAnalysis.h"

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

static llvm::StringRef iteratorName(mlir::utils::IteratorType iteratorType) {
  if (iteratorType == mlir::utils::IteratorType::parallel)
    return "parallel";
  if (iteratorType == mlir::utils::IteratorType::reduction)
    return "reduction";
  return "window";
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
collectCollectiveDemand(WaferLinalgExtCollectiveOpInterface op, unsigned opIndex,
                        OpTilingDemand &demand) {
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

static void printSlice(const TilingDemandSlice &slice, llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, dim] : llvm::enumerate(slice.loopDims)) {
    if (index != 0)
      os << ",";
    os << "d" << dim;
  }
  os << "]";
}

static void printRole(TilingDemandValueRole role, llvm::raw_ostream &os) {
  switch (role) {
  case TilingDemandValueRole::Input:
    os << "input";
    break;
  case TilingDemandValueRole::Output:
    os << "output";
    break;
  case TilingDemandValueRole::Result:
    os << "result";
    break;
  }
}

static void printDimList(llvm::ArrayRef<unsigned> dims, llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, dim] : llvm::enumerate(dims)) {
    if (index != 0)
      os << ",";
    os << "d" << dim;
  }
  os << "]";
}

static llvm::StringRef collectiveKindName(WaferLinalgExtCollectiveKind kind) {
  switch (kind) {
  case WaferLinalgExtCollectiveKind::AllGather:
    return "all_gather";
  case WaferLinalgExtCollectiveKind::ReduceScatter:
    return "reduce_scatter";
  case WaferLinalgExtCollectiveKind::AllReduce:
    return "all_reduce";
  case WaferLinalgExtCollectiveKind::AllToAll:
    return "all_to_all";
  case WaferLinalgExtCollectiveKind::CollectivePermute:
    return "collective_permute";
  }
  llvm_unreachable("unknown linalg-ext collective kind");
}

static void printI64List(llvm::ArrayRef<int64_t> values,
                         llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0)
      os << ",";
    os << value;
  }
  os << "]";
}

static void printLoopDimList(llvm::ArrayRef<unsigned> dims,
                             llvm::raw_ostream &os) {
  for (auto [index, dim] : llvm::enumerate(dims)) {
    if (index != 0)
      os << ",";
    os << "d" << dim;
  }
}

} // namespace

mlir::LogicalResult wafer::collectGroupTilingDemand(GroupOp group,
                                                    GroupTilingDemand &demand) {
  demand = {};
  demand.group = group;

  for (auto [index, input] : llvm::enumerate(group.getInputs()))
    addBoundaryValue(demand.boundaryValues, TilingDemandValueRole::Input,
                     static_cast<unsigned>(index), input);
  for (auto [index, output] : llvm::enumerate(group.getOuts()))
    addBoundaryValue(demand.boundaryValues, TilingDemandValueRole::Output,
                     static_cast<unsigned>(index), output);
  for (auto [index, result] : llvm::enumerate(group.getResults())) {
    addBoundaryValue(demand.boundaryValues, TilingDemandValueRole::Result,
                     static_cast<unsigned>(index), result);
    addBoundaryValue(demand.resultTiles, TilingDemandValueRole::Result,
                     static_cast<unsigned>(index), result);
  }

  mlir::Block &block = group.getBody().front();
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

void wafer::dumpGroupTilingDemand(const GroupTilingDemand &demand,
                                  llvm::StringRef groupLabel,
                                  llvm::raw_ostream &os) {
  os << "wafer.tiling_demand group " << groupLabel << "\n";
  for (const TilingDemandValue &tile : demand.resultTiles) {
    os << "  result_tile #" << tile.index << " type=" << tile.type << " slice=";
    printSlice(tile.slice, os);
    os << "\n";
  }

  for (const OpTilingDemand &opDemand : demand.ops) {
    os << "  op #" << opDemand.opIndex << " "
       << opDemand.op->getName().getStringRef() << "\n";

    if (opDemand.kind == OpTilingDemandKind::Failure) {
      os << "    failure " << opDemand.failureReason << "\n";
      continue;
    }

    if (!opDemand.iteratorTypes.empty()) {
      llvm::SmallVector<unsigned, 4> parallelDims;
      llvm::SmallVector<unsigned, 4> reductionDims;
      for (auto [index, iteratorType] :
           llvm::enumerate(opDemand.iteratorTypes)) {
        if (iteratorType == mlir::utils::IteratorType::parallel)
          parallelDims.push_back(static_cast<unsigned>(index));
        if (iteratorType == mlir::utils::IteratorType::reduction)
          reductionDims.push_back(static_cast<unsigned>(index));
      }
      os << "    iterators=";
      if (!parallelDims.empty()) {
        os << iteratorName(mlir::utils::IteratorType::parallel) << "(";
        printLoopDimList(parallelDims, os);
        os << ")";
      }
      if (!reductionDims.empty()) {
        if (!parallelDims.empty())
          os << " ";
        os << iteratorName(mlir::utils::IteratorType::reduction) << "(";
        printLoopDimList(reductionDims, os);
        os << ")";
      }
      os << "\n";
    }

    if (opDemand.kind == OpTilingDemandKind::LinalgExtCollective) {
      os << "    collective kind="
         << collectiveKindName(opDemand.collectiveInfo.kind);
      if (!opDemand.collectiveInfo.rankGroup.empty()) {
        os << " rank_group=";
        printI64List(opDemand.collectiveInfo.rankGroup, os);
      }
      if (opDemand.collectiveInfo.hasAxis)
        os << " axis=" << opDemand.collectiveInfo.axis;
      if (opDemand.collectiveInfo.hasSplitAxis)
        os << " split_axis=" << opDemand.collectiveInfo.splitAxis;
      if (opDemand.collectiveInfo.hasConcatAxis)
        os << " concat_axis=" << opDemand.collectiveInfo.concatAxis;
      os << "\n";
    }

    for (const TilingDemandValue &value : opDemand.values) {
      os << "    ";
      printRole(value.role, os);
      os << " #" << value.index << " slice=";
      printSlice(value.slice, os);
      os << "\n";
    }

    for (const TilingDemandAccumulator &accumulator : opDemand.accumulators) {
      os << "    accumulator result #" << accumulator.resultIndex
         << " reduction_dims=";
      printDimList(accumulator.reductionDims, os);
      os << "\n";
    }
  }
}
