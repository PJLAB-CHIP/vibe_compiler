//===- CardConsumerInputs.h - Selected boundary inputs ------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDCONSUMERINPUTS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDCONSUMERINPUTS_H

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

namespace wafer::compiler::detail {

mlir::LogicalResult
addCardConsumerInputs(TileMapping &mapping,
                      const analysis::ExactDemandProof &demand,
                      std::string *failureReason);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CARDCONSUMERINPUTS_H
