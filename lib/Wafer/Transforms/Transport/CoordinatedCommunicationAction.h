//===- CoordinatedCommunicationAction.h - All-rank actions -*- C++ -*-===//

#ifndef WAFER_COMPILER_COORDINATEDCOMMUNICATIONACTION_H
#define WAFER_COMPILER_COORDINATEDCOMMUNICATIONACTION_H

#include "Wafer/Frontend/Program/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

/// Invocation-local upper bound across all registered communication
/// providers. Providers describe cheap, read-only structural recipes; this
/// bound keeps their cross-product with the common schedule axes finite
/// without turning a provider point into an actual IR clone.
inline constexpr size_t kMaximumCoordinatedCommunicationActionPoints = 8;

/// Opaque key of one provider-local point in a canonical query. The key
/// is for deterministic equality and diagnostics only; common coordination
/// must never parse it to recover provider semantics. A point is query-local
/// and carries no pointer into the queried IR.
struct CommunicationActionKey {
  std::string providerKey;
  uint32_t stableOrdinal = 0;

  friend bool operator==(const CommunicationActionKey &lhs,
                         const CommunicationActionKey &rhs) {
    return lhs.providerKey == rhs.providerKey &&
           lhs.stableOrdinal == rhs.stableOrdinal;
  }
};

/// One immutable recipe for an all-rank canonical Instr parent. Common search
/// supplies discardable complete-rank clones. Materialization must
/// re-prove its capability from those clones and either mutate every required
/// rank atomically or fail; it cannot select a winner or run physical gates.
class CoordinatedCommunicationActionPoint {
public:
  virtual ~CoordinatedCommunicationActionPoint() = default;

  const CommunicationActionKey &getKey() const { return key; }

  virtual mlir::LogicalResult materialize(
      llvm::MutableArrayRef<mlir::ModuleOp> isolatedCanonicalInstrModules,
      const frontend::FrontendProgramVerificationResult &program,
      std::string *failureReason = nullptr) const = 0;

protected:
  explicit CoordinatedCommunicationActionPoint(CommunicationActionKey key);

private:
  CommunicationActionKey key;
};

using CoordinatedCommunicationActionPoints =
    llvm::SmallVector<std::unique_ptr<CoordinatedCommunicationActionPoint>, 4>;

/// Compiler-private capability provider over one immutable, complete-rank,
/// canonical/unplaced Instr parent. A semantic non-match is success with no
/// appended points. Failure is reserved for an invalid query contract or a
/// provider invariant. Querying is read-only and must not retain Operation
/// pointers in a returned point.
class CoordinatedCommunicationActionProvider {
public:
  virtual ~CoordinatedCommunicationActionProvider() = default;

  virtual llvm::StringRef getStableKey() const = 0;

  virtual mlir::LogicalResult
  query(llvm::ArrayRef<mlir::ModuleOp> currentCanonicalInstrModules,
        const frontend::FrontendProgramVerificationResult &program,
        CoordinatedCommunicationActionPoints &points,
        std::string *failureReason = nullptr) const = 0;
};

/// Verify the common provider domain without modifying it: one complete rank
/// set in one context, no Tile dataflow operation, no evaluation-owned
/// allocation/Direct-DTE binding or nonzero worker assignment, and valid
/// canonical Instr IR.
mlir::LogicalResult verifyCoordinatedCommunicationActionDomain(
    llvm::ArrayRef<mlir::ModuleOp> currentCanonicalInstrModules,
    const frontend::FrontendProgramVerificationResult &program,
    std::string *failureReason = nullptr);

/// One materialized provider point. The key is transferred directly from
/// the point and is never reconstructed from the rewritten modules.
struct CoordinatedCommunicationAction {
  CommunicationActionKey key;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> rankModules;
};

/// Clone and materialize one point after reserving common action work. Source
/// parents remain immutable. The returned modules are verified canonical
/// Instr and contain no Tile dataflow operations; physical/resource validation
/// remains the caller's responsibility.
mlir::FailureOr<CoordinatedCommunicationAction>
materializeCoordinatedCommunicationAction(
    llvm::ArrayRef<mlir::ModuleOp> currentCanonicalInstrModules,
    const frontend::FrontendProgramVerificationResult &program,
    const CoordinatedCommunicationActionPoint &point,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COORDINATEDCOMMUNICATIONACTION_H
