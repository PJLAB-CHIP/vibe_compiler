//===- CardEdgeCarriers.h - Selected dependency carriers ----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDEDGECARRIERS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDEDGECARRIERS_H

#include "Wafer/Planning/PhysicalDataflow/MovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

namespace wafer::compiler::detail {

mlir::LogicalResult
addCardEdgeCarriers(TileMapping &mapping, const SpatialAssignment &spatial,
                    const analysis::ExactDemandProof &demand,
                    const StructuredDAGAnalysis &dag,
                    const MovementPlan &movement, std::string *failureReason);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDEDGECARRIERS_H
