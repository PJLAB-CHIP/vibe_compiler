//===- Pipelines.h - StableHLO to structured tensor pipelines -*- C++ -*-===//
#pragma once

namespace mlir {
class OpPassManager;
}

namespace wafer {

void buildNormalizeImportedStablehloPipeline(mlir::OpPassManager &pm);
void buildLegalizeStablehloToStructuredTensorPipeline(
    mlir::OpPassManager &pm);
void buildSimplifyStructuredTensorPipeline(mlir::OpPassManager &pm);
void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm);

} // namespace wafer
