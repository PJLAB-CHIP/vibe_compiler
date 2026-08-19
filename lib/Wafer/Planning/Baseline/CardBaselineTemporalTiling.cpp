//===- CardBaselineTemporalTiling.cpp --------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineTemporalTiling.h"

#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"

#include "Wafer/Target/Core/TargetMemory.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {

uint64_t ceilDivide(uint64_t numerator, uint64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

} // namespace

mlir::LogicalResult
setCardBaselineTemporalTiles(CardBaselineAssignment &assignment,
                             const CardProgramAnalysis &program,
                             std::string *failureReason) {
  assignment.mapping.operationTemporalTiles.clear();
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16> nodeTemporalTiles;
  nodeTemporalTiles.resize(program.dag.getNodes().size());
  const TargetMemoryPolicy memory = getTargetMemoryPolicy();
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    if (node.id >= assignment.nodePlacements.size()) {
      if (failureReason)
        *failureReason = "structured node has no spatial placement";
      return mlir::failure();
    }
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> ranges =
        deriveLocalIteratorExtents(
            node.operation,
            assignment.nodePlacements[node.id].iteratorPartitionFactors,
            failureReason);
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
      extent = static_cast<int64_t>(ceilDivide(static_cast<uint64_t>(extent),
                                               output.activeTileIds.size()));
    }

    if (outputRoots[output.outputIndex].size() != 1)
      continue;
    const StructuredDAGNodeID rootId = outputRoots[output.outputIndex].front();
    const StructuredDAGNode *root = program.dag.getNode(rootId);
    auto returned =
        mlir::dyn_cast<mlir::OpResult>(returnOp.getOperand(output.outputIndex));
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
