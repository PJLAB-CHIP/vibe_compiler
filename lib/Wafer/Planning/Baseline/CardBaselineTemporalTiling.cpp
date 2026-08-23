//===- CardBaselineTemporalTiling.cpp --------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineTemporalTiling.h"

#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
mlir::LogicalResult
setCardBaselineTemporalTiles(CardBaselineAssignment &assignment,
                             const CardProgramAnalysis &program,
                             std::string *failureReason) {
  assignment.mapping.operationTemporalTiles.clear();
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16> nodeTemporalTiles;
  nodeTemporalTiles.resize(program.dag.getNodes().size());
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
    nodeTemporalTiles[node.id] = *ranges;
    assignment.mapping.operationTemporalTiles.push_back(
        StructuredOpTemporalTile{node.operation, std::move(*ranges)});
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
        output.shards.empty()) {
      if (failureReason)
        *failureReason = "output spatial shard is incomplete";
      return mlir::failure();
    }
    output.temporalTileSizes.assign(
        program.outputDomains[output.outputIndex].size(), 0);
    for (const OutputTileShard &shard : output.shards) {
      if (shard.sizes.size() != output.temporalTileSizes.size())
        return mlir::failure();
      for (auto [dimension, size] : llvm::enumerate(shard.sizes))
        output.temporalTileSizes[dimension] =
            std::max(output.temporalTileSizes[dimension], size);
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
