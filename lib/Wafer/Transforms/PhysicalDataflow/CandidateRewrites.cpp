//===- CandidateRewrites.cpp - Recompute derived physical facts --------===//

#include "Wafer/Transforms/PhysicalDataflow.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace wafer {

void clearPhysicalTileCandidateFacts(mlir::ModuleOp module) {
  module.walk([](mlir::memref::AllocOp allocation) {
    allocation->removeAttr(kWaferSPMOffsetAttrName);
    allocation->removeAttr(kWaferDDROffsetAttrName);
  });
  module.walk([](mlir::Operation *operation) {
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation))
      operation->removeAttr("binding");
  });
}

} // namespace wafer
