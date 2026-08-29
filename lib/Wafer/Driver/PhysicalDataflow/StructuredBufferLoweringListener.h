//===- StructuredBufferLoweringListener.h - Actual lowering adapter -----===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_STRUCTUREDBUFFERLOWERINGLISTENER_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_STRUCTUREDBUFFERLOWERINGLISTENER_H

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

namespace wafer::compiler::detail {

/// Candidate-actualization adapter joining current relation retargeting to
/// the Tile-to-Instr lowering recorder. It owns no IR and is scoped to one
/// caller-owned rewrite transaction.
class StructuredBufferLoweringListener final
    : public StructuredBufferReplacementListener,
      public TileRegionToInstrBufferRecorder {
public:
  using StructuredBufferReplacementListener::
      StructuredBufferReplacementListener;

  void recordScratchAllocation(mlir::Operation *sourceOperation,
                               mlir::Value allocation) final {
    StructuredBufferReplacementListener::recordScratchAllocation(
        sourceOperation, allocation);
  }

  void recordLoweredOperation(mlir::Operation *sourceOperation,
                              mlir::Operation *loweredOperation) final {
    StructuredBufferReplacementListener::recordLoweredOperation(
        sourceOperation, loweredOperation);
  }
};

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_STRUCTUREDBUFFERLOWERINGLISTENER_H
