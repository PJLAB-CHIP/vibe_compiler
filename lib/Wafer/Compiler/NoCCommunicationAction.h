//===- NoCCommunicationAction.h - Typed NoC action provider -*- C++ -*-===//

#ifndef WAFER_COMPILER_NOCCOMMUNICATIONACTION_H
#define WAFER_COMPILER_NOCCOMMUNICATIONACTION_H

#include "CoordinatedCommunicationAction.h"
#include "NoCIntermediateDataflow.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace wafer::compiler::detail {

/// Returns the first communication identity not used by current Tile peer /
/// collective operations or Instr DTE messages across the complete rank
/// domain. A negative result means the signed identity space is exhausted.
int64_t findNextNoCCommunicationId(llvm::ArrayRef<mlir::ModuleOp> rankModules);

/// Read-only exact capability query for replicated or repeated partitioned
/// boundary tiles. It derives rank/global tile equivalence solely from typed
/// frontend bindings and current canonical Instr RDMA operands.
bool hasNoCTypedBoundaryFanoutOpportunity(
    llvm::ArrayRef<mlir::ModuleOp> rankModules,
    const frontend::FrontendProgramVerificationResult &program);

/// Materialize every exact boundary fanout opportunity in deterministic
/// current-IR order. The caller owns a discardable complete-rank clone.
unsigned materializeNoCTypedBoundaryFanouts(
    llvm::MutableArrayRef<mlir::ModuleOp> rankModules,
    const frontend::FrontendProgramVerificationResult &program,
    int64_t &communicationId, NoCFanoutKind kind);

/// Put each newly inserted peer receive and its static allocation ahead of
/// every transport issue in the same structured block when current SSA/effects
/// prove that movement safe. Cross-block progress remains an all-rank Direct
/// DTE admission responsibility.
mlir::LogicalResult
normalizeNoCPeerReceivePreparation(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                                   std::string *failureReason = nullptr);

/// Provider-local points are PartialOnly, DirectFanout and
/// ReceiveForwardFanout. Their typed recipes remain private; common search
/// observes only the opaque provider-point identity.
class NoCCommunicationActionProvider final
    : public CoordinatedCommunicationActionProvider {
public:
  llvm::StringRef getStableKey() const final;

  mlir::LogicalResult
  query(llvm::ArrayRef<mlir::ModuleOp> currentCanonicalInstrModules,
        const frontend::FrontendProgramVerificationResult &program,
        CoordinatedCommunicationActionPoints &points,
        std::string *failureReason = nullptr) const final;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_NOCCOMMUNICATIONACTION_H
