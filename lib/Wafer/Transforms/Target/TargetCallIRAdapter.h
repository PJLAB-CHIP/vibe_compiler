//===- TargetCallIRAdapter.h - Instr IR to target protocol ----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TARGET_TARGETCALLIRADAPTER_H
#define WAFER_TRANSFORMS_TARGET_TARGETCALLIRADAPTER_H

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/Core/TargetCall.h"

namespace wafer {

/// Map one verified Instr enum to the pure target-call protocol. These
/// overloads deliberately live in the MLIR lowering library rather than the
/// target protocol library, keeping WaferTarget independent of WaferIR.
const TargetCallDescriptor &
getTargetCallDescriptor(InstrElementwiseKind operation);
const TargetCallDescriptor &getTargetCallDescriptor(InstrReduceKind operation);
const TargetCallDescriptor &
getTargetCallDescriptor(InstrConvertKind operation);
const TargetCallDescriptor &getTargetCallDescriptor(InstrConvKind operation);
const TargetCallDescriptor &getTargetCallDescriptor(InstrPoolKind operation);
const TargetCallDescriptor &getTargetCallDescriptor(InstrUnpoolKind operation);
const TargetCallDescriptor &
getTargetCallDescriptor(InstrPeripheralKind operation);

} // namespace wafer

#endif // WAFER_TRANSFORMS_TARGET_TARGETCALLIRADAPTER_H
