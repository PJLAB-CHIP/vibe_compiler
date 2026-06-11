//===- Passes.h - Wafer transform pass registration ------------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_H
#define WAFER_TRANSFORMS_PASSES_H

#include <cstdint>
#include <memory>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

#define GEN_PASS_DECL
#include "Wafer/Transforms/WaferPasses.h.inc"

std::unique_ptr<mlir::Pass> createLegalizeStablehloToLinalgPass();
mlir::LogicalResult planSPMMemoryModule(mlir::ModuleOp moduleOp,
                                        int64_t spmBase, int64_t spmLimit,
                                        int64_t spmAlignment);
mlir::LogicalResult planDDRMemoryModule(mlir::ModuleOp moduleOp,
                                        int64_t ddrAlignmentBytes,
                                        int64_t ddrCapacityBytes,
                                        int64_t ddrLargestContiguousBytes,
                                        int64_t ddrBandwidthLimitBytes);
#ifdef WAFER_ENABLE_SHARDY
std::unique_ptr<mlir::Pass> createApplyDefaultSpmdShardingPass();
std::unique_ptr<mlir::Pass>
createApplyDefaultSpmdShardingPass(int64_t tileCount);
#endif
void registerWaferTransformPasses();

} // namespace wafer

#endif // WAFER_TRANSFORMS_PASSES_H
