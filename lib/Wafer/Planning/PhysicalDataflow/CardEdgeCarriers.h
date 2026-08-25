//===- CardEdgeCarriers.h - Selected dependency carriers ----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDEDGECARRIERS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDEDGECARRIERS_H

#include "Wafer/Planning/PhysicalDataflow/MovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

#include <optional>
#include <utility>

namespace wafer::compiler::detail {

struct CoupledComponentResultMapping {
  mlir::Operation *operation = nullptr;
  unsigned result = 0;
  CoupledComponentValueId component;
};

struct StructuredOperationRootMapping {
  mlir::Operation *operation = nullptr;
  SemanticRootKey root;
  std::optional<ExecutionInstanceId> execution;
  llvm::SmallVector<std::pair<unsigned, unsigned>, 4> semanticOperandIndices;
};

mlir::LogicalResult
addCardEdgeCarriers(TileMapping &mapping, const SpatialAssignment &spatial,
                    const analysis::ExactDemandProof &demand,
                    const StructuredDAGAnalysis &dag,
                    const MovementPlan &movement,
                    llvm::ArrayRef<CoupledComponentResultMapping> components,
                    llvm::ArrayRef<StructuredOperationRootMapping> roots,
                    std::string *failureReason);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDEDGECARRIERS_H
