//===- WaferInterfaces.h - Wafer operation interfaces ---------*- C++ -*-===//

#ifndef WAFER_IR_WAFERINTERFACES_H
#define WAFER_IR_WAFERINTERFACES_H

#include "Wafer/ABI/Tx81NCCABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace wafer {

enum class MemLayout : uint32_t;
enum class MemorySpace : uint32_t;
enum class InstrFamily : uint32_t;
enum class NCCWorker : uint32_t;

inline constexpr uint32_t kNCCWorkerCount = WAFER_TX81_NCC_WORKER_COUNT;
inline constexpr uint32_t kAllNCCWorkersMask =
    WAFER_TX81_NCC_ALL_WORKERS_MASK;

/// Completion behavior of one local NCC instruction.
///
/// Ordinary issues are ordered within their worker domain but remain
/// externally pending. A typed join completes exactly its participant worker
/// set. Synchronous writeback operations both issue and drain internally.
/// `None` covers operations outside NCC completion, including Direct DTE.
enum class LocalInstructionCompletion : uint32_t {
  None,
  OrderedPending,
  ParticipantJoin,
  SynchronousWriteback,
};

struct NCCCompletionContract {
  LocalInstructionCompletion behavior = LocalInstructionCompletion::None;
  std::optional<NCCWorker> issueWorker;
  uint32_t participantMask = 0;
};

/// Derive the complete typed NCC issue/completion contract from one operation.
/// `participantMask` is nonzero only for a join or synchronous writeback.
NCCCompletionContract getNCCCompletionContract(mlir::Operation *operation);

/// Derive the local completion contract from the typed operation and its
/// standard resource effects. This is the shared scheduling/lifetime boundary;
/// callers must not infer completion from an operation or symbol name.
LocalInstructionCompletion
classifyLocalInstructionCompletion(mlir::Operation *operation);

/// Return the typed issue worker for ordinary NCC issue and synchronous
/// writeback operations.
std::optional<NCCWorker> getNCCIssueWorker(mlir::Operation *operation);

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
  GenericReciprocal,
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
