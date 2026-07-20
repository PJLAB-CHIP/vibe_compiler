//===- SyncOps.cpp - Wafer sync verifier implementation ---------------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

using namespace wafer;
using namespace wafer::detail;

// Instruction ordering ops currently rely on ODS traits only. This file owns
// future verifier code so verifier ownership stays explicit.
