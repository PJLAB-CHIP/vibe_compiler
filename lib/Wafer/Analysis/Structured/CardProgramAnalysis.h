//===- CardProgramAnalysis.h -------------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"

#include "Wafer/Analysis/Structured/StructuredMaterializationRelations.h"
#include "Wafer/Driver/Compilation.h"
#include "Wafer/Frontend/Program/Program.h"
#include "Wafer/IR/Target/TargetTopology.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace llvm {
class raw_ostream;
}

namespace wafer::compiler::detail {

using StaticOutputDomains = llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>;

/// Immutable current-IR facts shared by the baseline and search entrypoints.
/// Construction performs no placement, tiling, materialization or selection.
struct CardProgramAnalysis {
  CardProgramAnalysis(
      TargetTopology topology, llvm::SmallVector<TileId, 16> availableTileIds,
      StructuredDAGAnalysis dag, StaticOutputDomains outputDomains,
      llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes)
      : topology(std::move(topology)),
        availableTileIds(std::move(availableTileIds)), dag(std::move(dag)),
        outputDomains(std::move(outputDomains)),
        operationNodes(std::move(operationNodes)) {}

  TargetTopology topology;
  llvm::SmallVector<TileId, 16> availableTileIds;
  StructuredDAGAnalysis dag;
  StaticOutputDomains outputDomains;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
};

mlir::FailureOr<std::unique_ptr<CardProgramAnalysis>>
analyzeCardProgram(mlir::ModuleOp tensorProgram,
                   const frontend::FrontendProgramVerificationResult &program,
                   const ExecutionConfig &executionConfig,
                   llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail
