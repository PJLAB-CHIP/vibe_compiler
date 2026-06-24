//===- SyncOps.cpp - Wafer sync verifier implementation ---------------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

using namespace wafer;
using namespace wafer::detail;

// Instruction ordering ops currently rely on ODS traits only. This file owns
// future verifier code so verifier ownership stays explicit.

void SyncLocalFenceOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::Sync,
                       WaferResourceAccess::Fence, WaferValueRole::None, 0, -1);
}

mlir::LogicalResult SyncLocalFenceOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
