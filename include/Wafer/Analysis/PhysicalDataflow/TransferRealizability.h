//===- TransferRealizability.h - Exact physical transfer proofs -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_PHYSICALDATAFLOW_TRANSFERREALIZABILITY_H
#define WAFER_ANALYSIS_PHYSICALDATAFLOW_TRANSFERREALIZABILITY_H

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace wafer::analysis {

struct TransferRealizabilityLimits {
  int64_t maxEnumeratedElements = 65536;
};

/// Exact, invocation-local proofs over the current source/destination types,
/// logical relation, and physical encoding interfaces. The proof is consumed
/// immediately by a rewrite; no route report, descriptor cache, or accepted
/// side state is retained.
class TransferRealizability {
public:
  /// Prove two exact mappings from one rectangular iteration domain to the
  /// source and destination logical tensors.  This is the general movement
  /// form used by descriptor lowering: destination mapping must be injective
  /// (each physical result element is written once), while source mapping may
  /// be non-injective for broadcast.  The proof is symbolic and does not
  /// enumerate the iteration domain.
  static mlir::LogicalResult proveMappedTransfer(
      mlir::MemRefType sourceType, mlir::MemRefType destType,
      llvm::ArrayRef<int64_t> iterationShape,
      const IndexRelation &iterationToSource,
      const IndexRelation &iterationToDest);

  static mlir::LogicalResult
  proveMetadataView(mlir::MemRefType sourceType, mlir::MemRefType destType,
                    const IndexRelation &relation, bool destinationMayWrite,
                    const TransferRealizabilityLimits &limits = {});

  /// Prove the canonical row-major logical reshape between two static views.
  /// Unlike the general relation entry point, this preserves the construction
  /// guarantee instead of rebuilding and comparing equivalent Presburger
  /// relations.  Non-compact physical maps still use the bounded exact
  /// element-span proof and fail closed when the enumeration budget is
  /// exceeded.
  static mlir::LogicalResult proveStaticReshapeMetadataView(
      mlir::MemRefType sourceType, mlir::MemRefType destType,
      bool destinationMayWrite,
      const TransferRealizabilityLimits &limits = {});

  static mlir::LogicalResult
  proveCompactDma(mlir::MemRefType sourceType, mlir::MemRefType destType,
                  const IndexRelation &relation,
                  const TransferRealizabilityLimits &limits = {});

  /// Proves an exact cross-space logical transfer whose physical element
  /// spans can be represented by root-relative mapped RDMA/WDMA descriptors.
  static mlir::LogicalResult
  proveMappedDma(mlir::MemRefType sourceType, mlir::MemRefType destType,
                 const IndexRelation &relation,
                 const TransferRealizabilityLimits &limits = {});

  static mlir::LogicalResult
  proveGatherScatter(mlir::MemRefType sourceType, mlir::MemRefType destType,
                     const IndexRelation &relation,
                     const TransferRealizabilityLimits &limits = {});

  static mlir::LogicalResult
  proveStagedMovement(mlir::MemRefType sourceType,
                      mlir::MemRefType temporaryType, mlir::MemRefType destType,
                      const IndexRelation &sourceToTemporary,
                      const IndexRelation &temporaryToDest,
                      const TransferRealizabilityLimits &limits = {});
};

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_PHYSICALDATAFLOW_TRANSFERREALIZABILITY_H
