//===- TransferRealizability.h - Exact physical transfer proofs -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_PHYSICALDATAFLOW_TRANSFERREALIZABILITY_H
#define WAFER_ANALYSIS_PHYSICALDATAFLOW_TRANSFERREALIZABILITY_H

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::analysis {

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
  /// be non-injective for broadcast. Bit-packed mappings are represented in
  /// exact physical bit offsets; concrete route proofs separately decide
  /// whether a byte-addressable engine can consume them. The proof is symbolic
  /// and does not enumerate the iteration domain.
  static mlir::LogicalResult
  proveMappedTransfer(mlir::MemRefType sourceType, mlir::MemRefType destType,
                      llvm::ArrayRef<int64_t> iterationShape,
                      const IndexRelation &iterationToSource,
                      const IndexRelation &iterationToDest);

  /// Prove that source and destination visit corresponding logical elements
  /// at the same normalized physical element ordinal and that the source
  /// footprint covers the complete destination traversal. Element bit widths
  /// may differ; this is the exact precondition used by dtype-changing CT
  /// routes.
  static mlir::LogicalResult
  provePhysicalTraversal(mlir::MemRefType sourceType, mlir::MemRefType destType,
                         llvm::ArrayRef<int64_t> iterationShape,
                         const IndexRelation &iterationToSource,
                         const IndexRelation &iterationToDest);

  static mlir::LogicalResult proveMetadataView(mlir::MemRefType sourceType,
                                               mlir::MemRefType destType,
                                               const IndexRelation &relation,
                                               bool destinationMayWrite);

  /// Prove the canonical row-major logical reshape between two static views.
  /// Unlike the general relation entry point, this preserves the construction
  /// guarantee instead of rebuilding and comparing equivalent Presburger
  /// relations. Physical equality is proved by comparing the composed
  /// Presburger bit-offset relations over the complete destination domain; it
  /// never enumerates logical elements.
  static mlir::LogicalResult
  proveStaticReshapeMetadataView(mlir::MemRefType sourceType,
                                 mlir::MemRefType destType,
                                 bool destinationMayWrite);

  static mlir::LogicalResult proveCompactDma(mlir::MemRefType sourceType,
                                             mlir::MemRefType destType,
                                             const IndexRelation &relation);

  /// Proves an exact cross-space logical transfer whose physical element
  /// spans can be represented by root-relative mapped RDMA/WDMA descriptors.
  static mlir::LogicalResult proveMappedDma(mlir::MemRefType sourceType,
                                            mlir::MemRefType destType,
                                            const IndexRelation &relation);

  static mlir::LogicalResult proveGatherScatter(mlir::MemRefType sourceType,
                                                mlir::MemRefType destType,
                                                const IndexRelation &relation);

  static mlir::LogicalResult
  proveStagedMovement(mlir::MemRefType sourceType,
                      mlir::MemRefType temporaryType, mlir::MemRefType destType,
                      const IndexRelation &sourceToTemporary,
                      const IndexRelation &temporaryToDest);
};

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_PHYSICALDATAFLOW_TRANSFERREALIZABILITY_H
