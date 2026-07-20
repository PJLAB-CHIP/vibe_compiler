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
  static mlir::LogicalResult
  proveMetadataView(mlir::MemRefType sourceType, mlir::MemRefType destType,
                    const IndexRelation &relation, bool destinationMayWrite,
                    const TransferRealizabilityLimits &limits = {});

  static mlir::LogicalResult
  proveCompactDma(mlir::MemRefType sourceType, mlir::MemRefType destType,
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
