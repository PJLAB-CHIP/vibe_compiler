//===- CardBaselineAssignment.h -----------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Planning/Baseline/CardBaselineCompilation.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

namespace wafer::compiler::detail {

struct CanonicalBaselinePlan;

struct BaselineNodeRootRelation {
  uint32_t structuredNodeId = 0;
  SemanticRootKey root;
};

/// Complete deterministic assignment consumed by the one-shot CardModule
/// materializer. It contains no score, candidate ordinal or failure history.
struct CardBaselineAssignment {
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  llvm::SmallVector<StructuredDAGNodePlacement, 16> nodePlacements;
  TileMapping mapping;
};

struct CardBaselineModule {
  /// Keeps a winner-owned selected TensorProgram alive when Card construction
  /// was driven by an actual attention decomposition. Empty for ordinary
  /// source materialization.
  mlir::OwningOpRef<mlir::ModuleOp> materializationSource;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
  llvm::SmallVector<BaselineNodeRootRelation, 64> nodeRoots;
  CardBaselineAssignment assignment;
};

mlir::FailureOr<CardBaselineAssignment>
computeCardBaselineAssignment(const CardProgramAnalysis &program, CardId cardId,
                              BaselineStatistics *statistics,
                              llvm::raw_ostream &diagnostics);

/// Projects one already resolved canonical plan into the current one-shot
/// CardModule materializer contract without re-running placement, demand, or
/// temporal selection.
mlir::FailureOr<CardBaselineAssignment>
buildCardBaselineMaterializationAssignment(const CardProgramAnalysis &program,
                                           const CanonicalBaselinePlan &plan,
                                           BaselineStatistics *statistics,
                                           llvm::raw_ostream &diagnostics);

mlir::LogicalResult verifyCardBaselineMaterialization(
    mlir::ModuleOp cardModule, const CardBaselineAssignment &assignment,
    const StructuredDAGAnalysis &dag,
    const StructuredMaterializationRelations &relations,
    llvm::ArrayRef<TileId> expectedTileIds, std::string &failureReason);

mlir::FailureOr<CardBaselineModule> materializeCardBaseline(
    mlir::ModuleOp tensorProgram, CardId cardId,
    const CardProgramAnalysis &program, const CanonicalBaselinePlan &plan,
    BaselineStatistics *statistics, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail
