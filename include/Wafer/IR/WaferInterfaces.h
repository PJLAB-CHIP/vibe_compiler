//===- WaferInterfaces.h - Wafer operation interfaces ---------*- C++ -*-===//

#ifndef WAFER_IR_WAFERINTERFACES_H
#define WAFER_IR_WAFERINTERFACES_H

#include "Wafer/IR/NCCCompletion.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
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

/// Iterator groups inferred from the four attention indexing maps. The groups
/// are pairwise disjoint and cover the complete iteration domain for a
/// verifier-valid attention operation.
struct AttentionIterationRoles {
  llvm::SmallVector<unsigned, 4> batch;
  llvm::SmallVector<unsigned, 2> query;
  llvm::SmallVector<unsigned, 2> queryKeyReduction;
  llvm::SmallVector<unsigned, 2> keyValueReduction;
  llvm::SmallVector<unsigned, 2> valueOutput;
};

enum class CoupledReductionComponentKind : uint8_t {
  Maximum,
  Sum,
  Accumulator,
};

struct CoupledReductionComponent {
  CoupledReductionComponentKind kind;
  mlir::AffineMap indexingMap;
  mlir::Type elementType;
};

enum class CoupledReductionMergeKind : uint8_t {
  OnlineAttention,
};

enum class CoupledReductionFinalizationKind : uint8_t {
  NormalizeAccumulator,
};

/// Read-only source semantics for a coupled reduction. It intentionally owns
/// no planning, placement, storage, event, or target object.
struct CoupledReductionDescription {
  llvm::SmallVector<unsigned, 2> reductionIterators;
  llvm::SmallVector<CoupledReductionComponent, 3> components;
  CoupledReductionMergeKind mergeKind =
      CoupledReductionMergeKind::OnlineAttention;
  CoupledReductionFinalizationKind finalizationKind =
      CoupledReductionFinalizationKind::NormalizeAccumulator;
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
