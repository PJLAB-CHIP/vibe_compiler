//===- WaferInterfaces.cpp - Wafer operation interfaces ------------------===//

#include "Wafer/IR/WaferInterfaces.h"

#include "Wafer/IR/WaferDialect.h"

#include "Wafer/IR/WaferInterfaces.cpp.inc"

wafer::LocalInstructionCompletion
wafer::classifyLocalInstructionCompletion(mlir::Operation *operation) {
  if (!operation)
    return LocalInstructionCompletion::None;
  if (mlir::isa<SyncLocalFenceOp>(operation))
    return LocalInstructionCompletion::BarrierAndComplete;

  if (auto peripheral = mlir::dyn_cast<InstrPeripheralOp>(operation)) {
    switch (peripheral.getKindAttr().getValue()) {
    case InstrPeripheralKind::ArgMax:
    case InstrPeripheralKind::ArgMin:
      // The production CRT issues CT, drains the default worker's local NCC
      // queues with TsmWaitfinish(), then writes both scalar results to SPM.
      return LocalInstructionCompletion::BarrierAndComplete;
    default:
      break;
    }
  }

  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return LocalInstructionCompletion::None;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  bool hasLocalIssue = llvm::any_of(instances, [](const auto &effect) {
    return llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect()) &&
           llvm::isa<WaferComputeResource, WaferMovementResource>(
               effect.getResource());
  });
  return hasLocalIssue ? LocalInstructionCompletion::PendingUntilFence
                       : LocalInstructionCompletion::None;
}

llvm::StringRef
wafer::stringifyTargetImplementationKind(TargetImplementationKind kind) {
  switch (kind) {
  case TargetImplementationKind::Fill:
    return "fill";
  case TargetImplementationKind::Gemm:
    return "gemm";
  case TargetImplementationKind::BatchGemm:
    return "batch-gemm";
  case TargetImplementationKind::Generic:
    return "generic";
  case TargetImplementationKind::GenericReciprocal:
    return "generic-reciprocal";
  }
  llvm_unreachable("unknown target implementation kind");
}
