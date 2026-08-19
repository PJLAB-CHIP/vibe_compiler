//===- WaferInterfaces.h - Wafer operation interfaces ---------*- C++ -*-===//

#ifndef WAFER_IR_WAFERINTERFACES_H
#define WAFER_IR_WAFERINTERFACES_H

#include "Wafer/IR/NCCCompletion.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
namespace wafer {

enum class WaferLinalgExtCollectiveKind {
  AllGather,
  ReduceScatter,
  AllReduce,
  AllToAll,
  CollectivePermute,
};

struct WaferSPMResource
    : public mlir::SideEffects::Resource::Base<WaferSPMResource> {
  llvm::StringRef getName() final { return "WaferSPM"; }
};

struct WaferDDRResource
    : public mlir::SideEffects::Resource::Base<WaferDDRResource> {
  llvm::StringRef getName() final { return "WaferDDR"; }
};

struct WaferComputeResource
    : public mlir::SideEffects::Resource::Base<WaferComputeResource> {
  llvm::StringRef getName() final { return "WaferCompute"; }
};

struct WaferMovementResource
    : public mlir::SideEffects::Resource::Base<WaferMovementResource> {
  llvm::StringRef getName() final { return "WaferMovement"; }
};

struct WaferCommunicationResource
    : public mlir::SideEffects::Resource::Base<WaferCommunicationResource> {
  llvm::StringRef getName() final { return "WaferCommunication"; }
};

struct WaferSyncResource
    : public mlir::SideEffects::Resource::Base<WaferSyncResource> {
  llvm::StringRef getName() final { return "WaferSync"; }
};

} // namespace wafer

#include "Wafer/IR/WaferInterfaces.h.inc"

#endif // WAFER_IR_WAFERINTERFACES_H
