//===- CardBaselineConsumerInputs.h ----------------------------------===//

#ifndef WAFER_COMPILER_BASELINE_CARDBASELINECONSUMERINPUTS_H
#define WAFER_COMPILER_BASELINE_CARDBASELINECONSUMERINPUTS_H

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineConsumerInputs(
    TileMapping &mapping, const analysis::ExactDemandProof &demand,
    std::string *failureReason);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_BASELINE_CARDBASELINECONSUMERINPUTS_H
