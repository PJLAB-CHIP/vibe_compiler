//===- CardBaselineConsumerInputs.h ----------------------------------===//

#ifndef WAFER_COMPILER_BASELINE_CARDBASELINECONSUMERINPUTS_H
#define WAFER_COMPILER_BASELINE_CARDBASELINECONSUMERINPUTS_H

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineConsumerInputs(
    TileMapping &mapping, const StructuredDAGAnalysis &dag,
    const analysis::LogicalShardTrial &trial,
    StructuredDAGExactDemandQuery &query, std::string *failureReason);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_BASELINE_CARDBASELINECONSUMERINPUTS_H
