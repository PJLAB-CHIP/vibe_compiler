//===- SourceValidation.cpp - CardModule source validation ----------===//

#include "Internal.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"

namespace wafer::tensor_program_to_card_module {

mlir::FailureOr<TileMaterializationSourcePreparation>
prepareTileMaterializationSource(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    std::string *failureReason) {
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
        !nodeIds.insert(node.structuredNodeId).second ||
        node.operation->getParentOfType<mlir::ModuleOp>() != sourceModule)
      return failCardModuleValue<TileMaterializationSourcePreparation>(
          failureReason,
          "card structured operation-node mapping is null, duplicated or "
          "outside the tensor program");
    preparation.operationNodes.push_back(node);
  }
  return preparation;
}

} // namespace wafer::tensor_program_to_card_module
