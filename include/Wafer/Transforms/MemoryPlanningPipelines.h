//===- MemoryPlanningPipelines.h - Instr memory preparation -*- C++ -*-===//
#pragma once

namespace mlir {
class OpPassManager;
}

namespace wafer {

void buildBufferizeInstrFunctionsPipeline(mlir::OpPassManager &pm);
void buildPrepareInstrForMemoryPlanningPipeline(mlir::OpPassManager &pm);

} // namespace wafer
