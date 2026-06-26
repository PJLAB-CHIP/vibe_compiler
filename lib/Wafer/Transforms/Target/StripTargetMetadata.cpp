//===- StripTargetMetadata.cpp - Strip consumed Wafer target metadata -----===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace wafer {
#define GEN_PASS_DEF_STRIPTARGETMETADATAPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

static bool isWaferTargetMetadata(mlir::Operation *op) {
  return mlir::isa<TargetTopologyOp, ExecutionMeshOp>(op);
}

static bool isWaferOp(mlir::Operation *op) {
  mlir::Dialect *dialect = op->getDialect();
  return dialect &&
         dialect->getNamespace() == WaferDialect::getDialectNamespace();
}

struct StripTargetMetadataPass
    : public impl::StripTargetMetadataPassBase<StripTargetMetadataPass> {
  using impl::StripTargetMetadataPassBase<
      StripTargetMetadataPass>::StripTargetMetadataPassBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::Operation *remainingWaferOp = nullptr;
    moduleOp.walk([&](mlir::Operation *op) {
      if (op == moduleOp || isWaferTargetMetadata(op))
        return mlir::WalkResult::advance();
      if (!isWaferOp(op))
        return mlir::WalkResult::advance();
      remainingWaferOp = op;
      return mlir::WalkResult::interrupt();
    });
    if (remainingWaferOp) {
      remainingWaferOp->emitError()
          << "target_metadata_strip_failure: cannot strip target metadata "
             "while non-metadata Wafer ops remain";
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::Operation *, 4> toErase;
    moduleOp.walk([&](ExecutionMeshOp meshOp) {
      toErase.push_back(meshOp.getOperation());
    });
    for (mlir::Operation *op : toErase)
      op->erase();

    toErase.clear();
    moduleOp.walk([&](TargetTopologyOp topologyOp) {
      toErase.push_back(topologyOp.getOperation());
    });
    for (mlir::Operation *op : toErase)
      op->erase();
  }
};

} // namespace
} // namespace wafer
