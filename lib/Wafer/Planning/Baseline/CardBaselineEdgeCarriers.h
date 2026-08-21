//===- CardBaselineEdgeCarriers.h ------------------------------------===//

#ifndef WAFER_COMPILER_BASELINE_CARDBASELINEEDGECARRIERS_H
#define WAFER_COMPILER_BASELINE_CARDBASELINEEDGECARRIERS_H

#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineEdgeCarriers(
    TileMapping &mapping, const SpatialAssignment &spatial,
    const analysis::ExactDemandProof &demand,
    const StructuredDAGAnalysis &dag, std::string *failureReason);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_BASELINE_CARDBASELINEEDGECARRIERS_H
