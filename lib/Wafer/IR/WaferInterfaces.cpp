//===- WaferInterfaces.cpp - Wafer operation interfaces ------------------===//

#include "Wafer/IR/WaferInterfaces.h"

#include "Wafer/IR/WaferInterfaces.cpp.inc"

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
