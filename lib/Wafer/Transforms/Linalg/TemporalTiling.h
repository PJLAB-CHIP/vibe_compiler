//===- TemporalTiling.h - Apply live-operation temporal choices -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_TEMPORALTILING_H
#define WAFER_TRANSFORMS_LINALG_TEMPORALTILING_H

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace wafer {

enum class TensorAssemblyOpportunityKind : uint8_t {
  Available,
  NotApplicable,
  Unsupported,
  ResourceExhausted,
  BrokenContract,
};

struct TensorAssemblyOpportunity {
  TensorAssemblyOpportunityKind kind =
      TensorAssemblyOpportunityKind::NotApplicable;
  std::string detail;
};

enum class TemporalTilingFailureKind : uint8_t {
  None,
  Unsupported,
  BrokenContract,
  ResourceExhausted,
  CompilerFailure,
};

struct TemporalTilingFailure {
  TemporalTilingFailureKind kind = TemporalTilingFailureKind::None;
  std::string detail;
};

struct TemporalTilingStatistics {
  uint64_t tiledTraversals = 0;
  uint64_t loops = 0;
  uint64_t specializedTails = 0;
  uint64_t specializedConcatBoundaries = 0;
  uint64_t fusedProducers = 0;
  uint64_t viewTransparentProducers = 0;
  uint64_t tileLocalAssemblies = 0;
  uint64_t assembledSegments = 0;
  uint64_t decomposedPads = 0;
  uint64_t decomposedConstantGenerates = 0;
};

struct TemporalTilingRequest {
  const compiler::detail::TemporalDomain &domain;
  const compiler::detail::TemporalChoice &choice;
};

/// Applies a batch of choices to distinct regions of one current candidate.
/// All choices and global relations are checked before the first mutation;
/// global output checks and replacement-listener finalization run once after
/// the batch. Each region still runs its local verifier and reports every
/// replacement to the same invocation-owned listener. Domains/choices are
/// invalid after this call; on post-mutation failure discard the complete
/// candidate owner.
mlir::FailureOr<TemporalTilingStatistics>
applyTemporalTiling(llvm::ArrayRef<TemporalTilingRequest> requests,
                    StructuredMaterializationRelations &relations,
                    TemporalTilingFailure *failure = nullptr,
                    mlir::RewriterBase::Listener *listener = nullptr);

/// Inspect one actual Tensor subset after structural expansion. Unsupported
/// and exhausted proofs remain distinct from a read with no assembly source.
/// Handles borrow the current unchanged IR epoch.
TensorAssemblyOpportunity
queryLocalTensorAssemblyRead(mlir::tensor::ExtractSliceOp read,
                             const analysis::IndexRelationLimits &limits =
                                 analysis::IndexRelationLimits());

/// Explicit local implementation of current assembly reads. Keep the shared
/// value and its other observers intact. On failure, discard the candidate.
mlir::FailureOr<TemporalTilingStatistics> materializeLocalTensorAssemblyReads(
    TileRegionOp region, llvm::ArrayRef<mlir::tensor::ExtractSliceOp> reads,
    StructuredMaterializationRelations &relations,
    TemporalTilingFailure *failure = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_LINALG_TEMPORALTILING_H
