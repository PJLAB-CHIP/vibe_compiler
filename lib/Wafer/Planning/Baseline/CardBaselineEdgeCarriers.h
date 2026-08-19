//===- CardBaselineEdgeCarriers.h ------------------------------------===//

#ifndef WAFER_COMPILER_BASELINE_CARDBASELINEEDGECARRIERS_H
#define WAFER_COMPILER_BASELINE_CARDBASELINEEDGECARRIERS_H

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineEdgeCarriers(
    TileMapping &mapping, const analysis::LogicalShardTrial &trial,
    const StructuredDAGAnalysis &dag, StructuredDAGExactDemandQuery &query,
    std::string *failureReason);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_BASELINE_CARDBASELINEEDGECARRIERS_H
