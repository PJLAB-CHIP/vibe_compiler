//===- CardBaselineTemporalTiling.cpp --------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineTemporalTiling.h"

#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"

#include "Wafer/Support/TargetPolicy.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {

uint64_t ceilDivide(uint64_t numerator, uint64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>> getIteratorRanges(
    mlir::Operation *operation,
    const StructuredDAGNodePlacement &placement) {
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(operation);
  if (!tiling)
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> ranges;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation)) {
    ranges = linalg.getStaticLoopRanges();
  } else if (operation->getNumResults() == 1) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(
        operation->getResult(0).getType());
    if (!type || !type.hasStaticShape())
      return mlir::failure();
    ranges.assign(type.getShape().begin(), type.getShape().end());
  } else {
    return mlir::failure();
  }
  if (ranges.size() != placement.iteratorPartitionFactors.size())
    return mlir::failure();
  for (auto [dimension, factor] :
       llvm::enumerate(placement.iteratorPartitionFactors)) {
    if (factor == 0 || ranges[dimension] <= 0)
      return mlir::failure();
    ranges[dimension] = static_cast<int64_t>(ceilDivide(
        static_cast<uint64_t>(ranges[dimension]), factor));
  }
  return ranges;
}

} // namespace

mlir::LogicalResult setCardBaselineTemporalTiles(
    CardBaselineAssignment &assignment, const CardProgramAnalysis &program,
    std::string *failureReason) {
  assignment.mapping.operationTemporalTiles.clear();
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16> nodeTemporalTiles;
  nodeTemporalTiles.resize(program.dag.getNodes().size());
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    if (node.id >= assignment.nodePlacements.size()) {
      if (failureReason)
        *failureReason = "structured node has no spatial placement";
      return mlir::failure();
    }
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> ranges =
        getIteratorRanges(node.operation, assignment.nodePlacements[node.id]);
    if (mlir::failed(ranges)) {
      if (failureReason)
        *failureReason = "structured node has no static local iterator box";
      return mlir::failure();
    }
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> temporal =
        deriveStructuredOperationTemporalTileShape(node.operation, *ranges,
                                                   memory);
    if (mlir::failed(temporal)) {
      if (failureReason)
        *failureReason =
            "structured node has no proven lowering SPM wave bound";
      return mlir::failure();
    }
    nodeTemporalTiles[node.id] = *temporal;
    assignment.mapping.operationTemporalTiles.push_back(
        StructuredOpTemporalTile{node.operation, std::move(*temporal)});
  }

  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      program.dag.getFunction().getBody().front().getTerminator());
  if (!returnOp) {
    if (failureReason)
      *failureReason = "structured program has no functional return";
    return mlir::failure();
  }
  llvm::ArrayRef<llvm::SmallVector<StructuredDAGNodeID, 2>> outputRoots =
      program.dag.getObservableOutputRootNodes();
  for (OutputTileMapping &output : assignment.mapping.outputs) {
    if (output.outputIndex >= program.outputDomains.size() ||
        output.outputIndex >= outputRoots.size() ||
        output.outputIndex >= returnOp.getNumOperands() ||
        (output.shardDimension &&
         *output.shardDimension >=
             program.outputDomains[output.outputIndex].size()) ||
        output.activeTileIds.empty()) {
      if (failureReason)
        *failureReason = "output spatial shard is incomplete";
      return mlir::failure();
    }
    output.temporalTileSizes = program.outputDomains[output.outputIndex];
    if (output.shardDimension) {
      int64_t &extent = output.temporalTileSizes[*output.shardDimension];
      extent = static_cast<int64_t>(ceilDivide(
          static_cast<uint64_t>(extent), output.activeTileIds.size()));
    }

    if (outputRoots[output.outputIndex].size() != 1)
      continue;
    const StructuredDAGNodeID rootId = outputRoots[output.outputIndex].front();
    const StructuredDAGNode *root = program.dag.getNode(rootId);
    auto returned = mlir::dyn_cast<mlir::OpResult>(
        returnOp.getOperand(output.outputIndex));
    if (!root || !returned || returned.getOwner() != root->operation)
      continue;
    std::optional<llvm::SmallVector<int64_t, 4>> resultTile =
        getStructuredResultTileShape(root->operation,
                                     returned.getResultNumber(),
                                     nodeTemporalTiles[rootId]);
    if (!resultTile || resultTile->size() != output.temporalTileSizes.size())
      continue;
    for (auto [outputSize, rootSize] :
         llvm::zip_equal(output.temporalTileSizes, *resultTile))
      outputSize = std::min(outputSize, rootSize);
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
