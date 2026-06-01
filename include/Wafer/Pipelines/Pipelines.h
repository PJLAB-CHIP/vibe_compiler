//===- Pipelines.h - Wafer named pipeline registration ---------*- C++ -*-===//

#ifndef WAFER_PIPELINES_PIPELINES_H
#define WAFER_PIPELINES_PIPELINES_H

#include <cstdint>

#include "llvm/ADT/StringRef.h"

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace wafer {

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm);

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm,
                                               int64_t defaultTileCount = 16);
#endif

void buildLinalgToCAbiPipeline(mlir::OpPassManager &pm,
                               llvm::StringRef target = "wafer",
                               llvm::StringRef tileMapping = "single");

void buildStablehloToCAbiPipeline(mlir::OpPassManager &pm,
                                  llvm::StringRef target = "wafer",
                                  llvm::StringRef tileMapping = "single");

void buildTileCommunicationToCAbiPipeline(mlir::OpPassManager &pm,
                                          llvm::StringRef target = "wafer");

void registerWaferPipelines();

} // namespace wafer

#endif // WAFER_PIPELINES_PIPELINES_H
