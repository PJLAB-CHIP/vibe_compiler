//===- StagePipeline.h - Selected stage event materialization -*- C++ -*-===//

#pragma once

#include "Wafer/Transforms/Bufferization/SelectedBufferMaterialization.h"

namespace wafer::compiler::detail {

struct MaterializedStagePipeline {
  size_t scopeIndex = 0;
  uint32_t slotCount = 0;
  unsigned stageCount = 0;
  unsigned slotAllocationCount = 0;
};

struct StagePipelineMaterialization {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
  llvm::SmallVector<MaterializedStagePipeline, 4> pipelines;
  unsigned slotAllocationCount = 0;
};

/// Consumes one prepared canonical Instr module and materializes every selected
/// buffering scope as an explicit SCF stage pipeline. Empty scopes are the
/// serialized identity and perform no work. Each nonempty scope must produce
/// at least two stages and exactly its selected slot multiplicity. The caller
/// discards this owned module on any failure.
mlir::FailureOr<StagePipelineMaterialization> materializeStagePipelines(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<SelectedBufferingScope> scopes,
    StructuredMaterializationRelations materializationRelations,
    SelectedBufferMaterializationFailure *failure = nullptr);

} // namespace wafer::compiler::detail
