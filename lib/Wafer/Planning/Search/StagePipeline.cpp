//===- StagePipeline.cpp - Selected stage event materialization --------===//

#include "Wafer/Planning/Search/StagePipeline.h"

#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace wafer::compiler::detail {

mlir::FailureOr<StagePipelineMaterialization> materializeStagePipelines(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<SelectedBufferingScope> scopes,
    StructuredMaterializationRelations materializationRelations,
    SelectedBufferMaterializationFailure *failure) {
  if (failure)
    *failure = {};
  auto fail = [&](size_t scopeIndex, llvm::StringRef message) {
    if (failure) {
      failure->kind = SelectedBufferMaterializationFailureKind::InvalidRequest;
      failure->scopeIndex = scopeIndex;
      failure->detail = message.str();
    }
    return mlir::failure();
  };
  if (!module || mlir::failed(checkStructuredBufferRelationsCurrent(
                     module->getOperation(), materializationRelations)))
    return fail(std::numeric_limits<size_t>::max(),
                "stage pipeline requires owned Instr and current relations");

  StagePipelineMaterialization result;
  for (auto [scopeIndex, scope] : llvm::enumerate(scopes)) {
    if (scope.requests.empty())
      return fail(scopeIndex, "stage pipeline scope has no exact logical edge");
    const uint32_t slotCount = scope.requests.front().bufferCount;
    auto materialized = materializeSelectedBuffering(
        std::move(module), scope.requests, std::move(materializationRelations),
        failure);
    if (mlir::failed(materialized)) {
      if (failure)
        failure->scopeIndex = scopeIndex;
      return mlir::failure();
    }
    if (materialized->stageCount < 2 ||
        materialized->maximumSlotCount != slotCount)
      return fail(scopeIndex,
                  "stage pipeline did not materialize selected stages and "
                  "slot multiplicity");
    if (result.slotAllocationCount > std::numeric_limits<unsigned>::max() -
                                         materialized->slotAllocationCount)
      return fail(scopeIndex, "stage pipeline slot allocation count overflows");
    result.pipelines.push_back(MaterializedStagePipeline{
        scopeIndex, slotCount, materialized->stageCount,
        materialized->slotAllocationCount});
    result.slotAllocationCount += materialized->slotAllocationCount;
    module = std::move(materialized->module);
    materializationRelations =
        std::move(materialized->materializationRelations);
  }
  result.module = std::move(module);
  result.materializationRelations = std::move(materializationRelations);
  return result;
}

} // namespace wafer::compiler::detail
