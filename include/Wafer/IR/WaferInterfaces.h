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

namespace mlir {
class ModuleOp;
}

namespace wafer {

enum class MemLayout : uint32_t;
enum class MemorySpace : uint32_t;
enum class InstrFamily : uint32_t;
enum class NCCWorker : uint32_t;

inline constexpr uint32_t kNCCWorkerCount = WAFER_TX81_NCC_WORKER_COUNT;
inline constexpr uint32_t kAllNCCWorkersMask = WAFER_TX81_NCC_ALL_WORKERS_MASK;
inline constexpr llvm::StringLiteral kWaferNCCWorkerAttrName = "worker";

/// Completion behavior of one local NCC instruction.
///
/// Ordinary issues are ordered within their worker domain but remain
/// externally pending. A typed join completes exactly its participant worker
/// set. Synchronous writeback operations both issue and drain internally.
/// `None` covers operations outside NCC completion, including Direct DTE.
enum class NCCSynchronizationBehavior : uint32_t {
  None,
  OrderedAsynchronousIssue,
  ParticipantJoin,
  SynchronousWriteback,
};

struct NCCSynchronizationContract {
  NCCSynchronizationBehavior behavior = NCCSynchronizationBehavior::None;
  std::optional<NCCWorker> issueWorker;
  uint32_t participantMask = 0;
};

/// Recomputable summary of actual pending NCC worker windows in structured
/// instruction IR. `issuedWorkerMask` names every typed worker used by an
/// issue; `hasCrossWorkerWindow` is true only when at least two worker domains
/// are simultaneously pending along one structured execution path.
struct NCCWorkerWindowSummary {
  uint32_t issuedWorkerMask = 0;
  bool hasCrossWorkerWindow = false;
};

/// Derive the complete typed NCC issue/completion contract from one operation.
/// `participantMask` is nonzero only for a join or synchronous writeback.
NCCSynchronizationContract getNCCSynchronizationContract(mlir::Operation *operation);

/// Analyze typed issue/join order through func, tile-region, and structured
/// control-flow regions. The result is derived solely from current IR and may
/// be recomputed after every scheduling rewrite.
NCCWorkerWindowSummary analyzeNCCWorkerWindows(mlir::ModuleOp module);

/// Derive the local completion contract from the typed operation and its
/// standard resource effects. This is the shared scheduling/lifetime boundary;
/// callers must not infer completion from an operation or symbol name.
NCCSynchronizationBehavior
classifyNCCSynchronizationBehavior(mlir::Operation *operation);

/// Return the typed issue worker for ordinary NCC issue and synchronous
/// writeback operations.
std::optional<NCCWorker> getNCCIssueWorker(mlir::Operation *operation);

/// Update the explicit worker domain of one typed NCC issue. This is the
/// shared mutation boundary for a whole-DAG candidate that explicitly selects
/// a worker; it rejects non-issue operations and out-of-domain values.
mlir::LogicalResult setNCCIssueWorker(mlir::Operation *operation,
                                      NCCWorker worker);

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
