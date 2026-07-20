//===- WaferInterfaces.h - Wafer operation interfaces ---------*- C++ -*-===//

#ifndef WAFER_IR_WAFERINTERFACES_H
#define WAFER_IR_WAFERINTERFACES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace wafer {

enum class MemLayout : uint32_t;
enum class MemorySpace : uint32_t;
enum class InstrFamily : uint32_t;

/// Closed target capabilities consumed while enumerating source
/// implementations.  These values are compiler inputs, not source-IR attrs or
/// serialized candidate state.
struct WaferTargetCapabilities {
  bool supportsElementwiseReciprocal = true;
  bool supportsElementwiseDivision = true;
};

/// A target implementation form that the current tile/instruction pipeline
/// can materialize and verify.  The first candidate returned by a source
/// interface is its production baseline.
enum class TargetImplementationKind : uint32_t {
  Fill,
  Gemm,
  BatchGemm,
  Generic,
  GenericReciprocalViaDivision,
};

llvm::StringRef
stringifyTargetImplementationKind(TargetImplementationKind kind);

struct TargetImplementationCandidate {
  TargetImplementationKind kind;

  friend bool operator==(const TargetImplementationCandidate &lhs,
                         const TargetImplementationCandidate &rhs) {
    return lhs.kind == rhs.kind;
  }
};

/// Conversion-owned materialization context.  The source OpInterface owns
/// candidate enumeration and the selected hook; the context owns physical
/// operands/results and creates typed wafer.tile IR in the isolated clone.
class WaferTargetImplementationMaterializer {
public:
  virtual ~WaferTargetImplementationMaterializer() = default;

  virtual mlir::LogicalResult materializeTargetImplementation(
      mlir::Operation *source, const TargetImplementationCandidate &candidate,
      mlir::OpBuilder &builder) = 0;
};

enum class WaferValueRole {
  None,
  Operand,
  Result,
};

enum class WaferLinalgExtCollectiveKind {
  AllGather,
  ReduceScatter,
  AllReduce,
  AllToAll,
  CollectivePermute,
};

struct WaferLinalgExtCollectiveInfo {
  WaferLinalgExtCollectiveKind kind;
  llvm::SmallVector<int64_t, 8> rankGroup;
  llvm::SmallVector<int64_t, 8> rankGroups;
  llvm::SmallVector<int64_t, 8> sourceTargetPairs;
  int64_t axis = -1;
  int64_t splitAxis = -1;
  int64_t concatAxis = -1;
  int64_t splitCount = -1;
  int64_t rankGroupSize = -1;
  int64_t channelId = -1;
  bool hasAxis = false;
  bool hasSplitAxis = false;
  bool hasConcatAxis = false;
  bool hasSplitCount = false;
  bool hasRankGroups = false;
  bool hasChannelId = false;
  bool useGlobalDeviceIds = false;
  bool hasCombiner = false;
  bool hasCommunicationEffect = false;
};

struct WaferLayoutRequirement {
  WaferValueRole role;
  unsigned index;
  MemLayout layout;
  MemorySpace memorySpace;
};

enum class WaferResourceKind {
  SPM,
  DDR,
  Movement,
  Compute,
  Communication,
  Sync,
};

enum class WaferResourceAccess {
  Read,
  Write,
  Issue,
  Wait,
  Fence,
};

struct WaferResourceEffect {
  WaferResourceKind resource;
  WaferResourceAccess access;
  WaferValueRole role;
  unsigned index;
  int64_t bytes;
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
