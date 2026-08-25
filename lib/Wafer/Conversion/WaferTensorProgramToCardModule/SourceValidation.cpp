//===- SourceValidation.cpp - CardModule source validation ----------===//

#include "Internal.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::tensor_program_to_card_module {

mlir::FailureOr<TileMaterializationSourcePreparation>
prepareTileMaterializationSource(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<StructuredNodeRootGroup> operationRootGroups,
    std::string *failureReason) {
  wafer::support::ScopedCompileTimingSpan timing(
      "query", "tensor-program-to-card-module", "validate-source");
  TileMaterializationSourcePreparation preparation;
  if (failureReason)
    failureReason->clear();
  if (!sourceModule)
    return failCardModuleValue<TileMaterializationSourcePreparation>(
        failureReason, "source module is null");

  {
    mlir::ScopedDiagnosticHandler suppress(
        sourceModule.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    if (mlir::failed(mlir::verify(sourceModule)))
      return failCardModuleValue<TileMaterializationSourcePreparation>(
          failureReason, "source module is not verifier-legal");
  }

  std::string topologyFailure;
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(sourceModule, &topologyFailure);
  if (mlir::failed(topology))
    return failCardModuleValue<TileMaterializationSourcePreparation>(
        failureReason, topologyFailure);
  std::optional<llvm::ArrayRef<TileId>> availableTiles =
      topology->getAvailableTileIds(cardId);
  if (!availableTiles)
    return failCardModuleValue<TileMaterializationSourcePreparation>(
        failureReason, "requested card_id is outside target topology");
  if (availableTiles->empty())
    return failCardModuleValue<TileMaterializationSourcePreparation>(
        failureReason, "requested card has no available Tiles");
  preparation.availableTiles.assign(availableTiles->begin(),
                                    availableTiles->end());
  if (mlir::failed(verifyLogicalMesh(sourceModule, failureReason)))
    return mlir::failure();

  mlir::FailureOr<mlir::func::FuncOp> sourceProgram =
      getSourceTensorProgram(sourceModule, failureReason);
  if (mlir::failed(sourceProgram) ||
      mlir::failed(verifyDirectSourceMembers(sourceModule, *sourceProgram,
                                             failureReason)))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>
      outputDomains = getStaticOutputDomains(*sourceProgram, failureReason);
  if (mlir::failed(outputDomains))
    return mlir::failure();
  preparation.outputDomains = std::move(*outputDomains);
  preparation.sourceModule = sourceModule;
  preparation.sourceProgram = *sourceProgram;
  preparation.cardId = cardId;
  preparation.sourceArgumentCount = sourceProgram->getNumArguments();

  llvm::DenseSet<mlir::Operation *> nodeOperations;
  llvm::DenseSet<uint32_t> nodeIds;
  preparation.operationNodes.reserve(operationNodes.size());
  for (const StructuredOperationNodeMapping &node : operationNodes) {
    if (!node.operation || !nodeOperations.insert(node.operation).second ||
        node.operation->getParentOfType<mlir::ModuleOp>() != sourceModule ||
        (!node.coupledComponentIndices.empty() &&
         (node.coupledComponentIndices.size() !=
              node.operation->getNumResults() ||
          llvm::SmallDenseSet<unsigned, 4>(node.coupledComponentIndices.begin(),
                                           node.coupledComponentIndices.end())
                  .size() != node.coupledComponentIndices.size())))
      return failCardModuleValue<TileMaterializationSourcePreparation>(
          failureReason,
          "card structured operation-node mapping has a null or duplicated "
          "operation, invalid result identities, or is outside the tensor "
          "program");
    nodeIds.insert(node.structuredNodeId);
    preparation.operationNodes.push_back(node);
  }
  preparation.operationRootGroups.reserve(nodeIds.size());
  if (operationRootGroups.empty()) {
    llvm::DenseSet<uint32_t> groupedNodes;
    for (const StructuredOperationNodeMapping &node : operationNodes)
      if (groupedNodes.insert(node.structuredNodeId).second)
        preparation.operationRootGroups.push_back(
            {node.structuredNodeId, node.structuredNodeId});
  } else {
    llvm::DenseSet<uint32_t> groupedNodes;
    for (const StructuredNodeRootGroup &relation : operationRootGroups) {
      if (!nodeIds.contains(relation.structuredNodeId) ||
          !groupedNodes.insert(relation.structuredNodeId).second)
        return failCardModuleValue<TileMaterializationSourcePreparation>(
            failureReason,
            "card structured node/root-group relation is unknown or "
            "duplicated");
      preparation.operationRootGroups.push_back(relation);
    }
    if (groupedNodes.size() != nodeIds.size())
      return failCardModuleValue<TileMaterializationSourcePreparation>(
          failureReason,
          "card structured node/root-group relation is incomplete");
  }
  llvm::sort(preparation.operationRootGroups,
             [](const StructuredNodeRootGroup &lhs,
                const StructuredNodeRootGroup &rhs) {
               return lhs.structuredNodeId < rhs.structuredNodeId;
             });
  return preparation;
}

} // namespace wafer::tensor_program_to_card_module
