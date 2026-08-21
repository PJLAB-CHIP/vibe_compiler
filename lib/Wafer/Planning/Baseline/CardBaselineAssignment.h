//===- CardBaselineAssignment.h -----------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Baseline/CardBaselineCompilation.h"
#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

namespace wafer::compiler::detail {

/// Complete deterministic assignment consumed by the one-shot CardModule
/// materializer. It contains no score, candidate ordinal or failure history.
struct CardBaselineAssignment {
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  llvm::SmallVector<StructuredDAGNodePlacement, 16> nodePlacements;
  TileMapping mapping;
};

struct CardBaselineModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

mlir::FailureOr<CardBaselineAssignment> computeCardBaselineAssignment(
    const CardProgramAnalysis &program, CardId cardId,
    BaselineStatistics *statistics, llvm::raw_ostream &diagnostics);

mlir::LogicalResult verifyCardBaselineMaterialization(
    mlir::ModuleOp cardModule, const CardBaselineAssignment &assignment,
    const StructuredDAGAnalysis &dag,
    const StructuredMaterializationRelations &relations,
    llvm::ArrayRef<TileId> expectedTileIds, std::string &failureReason);

mlir::FailureOr<CardBaselineModule> materializeCardBaseline(
    mlir::ModuleOp tensorProgram, CardId cardId,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    const CardBaselineAssignment &assignment,
    BaselineStatistics *statistics, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail
