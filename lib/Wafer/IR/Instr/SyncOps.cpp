//===- SyncOps.cpp - Wafer sync verifier implementation ---------------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

using namespace wafer;
using namespace wafer::detail;

// `wafer.sync.*` ops currently rely on ODS traits only. This file owns
// future sync-family verifier code so verifier ownership stays explicit.

void SyncLocalDrainOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::Sync,
                       WaferResourceAccess::Drain, WaferValueRole::None, 0, -1);
}

mlir::LogicalResult SyncLocalDrainOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
