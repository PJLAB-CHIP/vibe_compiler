//===- StructuredCandidateDerivation.h --------------*- C++ -*-===//

#ifndef WAFER_COMPILER_STRUCTUREDCANDIDATEDERIVATION_H
#define WAFER_COMPILER_STRUCTUREDCANDIDATEDERIVATION_H

#include "StructuredImplementationAlternative.h"
#include "RankCandidateSearch.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/Support/OptimizationConfig.h"
#include "Wafer/Transforms/CompleteRankMaterialization.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

/// One query-local, typed way to materialize a complete structured candidate.
/// Every candidate family enters the same coordinated candidates through this
/// representation; this is neither persistent IR nor a family-local winner.
struct StructuredTraversalProposal {
  int64_t stableSemanticOrdinal = 0;
  CompleteRankTraversalComposition composition =
      CompleteRankTraversalComposition::Coupled;
  CandidateTileTraversalKind traversalKind =
      CandidateTileTraversalKind::ResultDriven;
  CandidateTileResidencyAction residencyAction =
      CandidateTileResidencyAction::KeepSingleRegion;
  CandidateBoundaryMovementAction boundaryMovementAction =
      CandidateBoundaryMovementAction::Staged;
  CandidateLoopMovementAction loopMovementAction =
      CandidateLoopMovementAction::AsConstructed;
  llvm::SmallVector<int64_t, 4> tileSizes;
  llvm::SmallVector<int64_t, 2> reductionTileSizes;
  std::optional<TargetImplementationKind> selectedImplementation;
  std::optional<unsigned> physicalLayoutProposalOrdinal;
  bool cloneReservedBaseline = false;
};

/// The complete materialization recipe carried by one query-local structural
/// proposal. It owns only typed decisions and stable connection-order
/// parameters. It never owns an IR clone, an executable-evaluation
/// reservation, or a pointer into a mutable source operation.
enum class CoordinatedStructuralRecipeKind : uint8_t {
  MandatoryBaseline,
  Traversal,
  Connections,
  ImplementationAlternative,
};

struct CoordinatedStructuralRecipe {
  CoordinatedStructuralRecipeKind kind =
      CoordinatedStructuralRecipeKind::MandatoryBaseline;
  StructuredTraversalProposal traversal;
  llvm::SmallVector<CandidateTraversalConnectionChoice, 16> connections;
  std::shared_ptr<const StructuredImplementationAlternativePoint>
      implementationAlternative;
  CandidateTileResidencyAction implementationPostResidencyAction =
      CandidateTileResidencyAction::KeepSingleRegion;
};

/// One pre-materialization candidates member. Facts are conservative
/// projections from the immutable source and typed recipe. A dimension that
/// cannot be proved before replay remains Unknown with a stable disposition; it
/// is never treated as zero. canonicalKey and coverageClass are
/// invocation-local search keys and are not serialized into selected IR or
/// output files.
struct CoordinatedStructuralProposal {
  int64_t stableSemanticOrdinal = 0;
  bool reservedBaseline = false;
  CoordinatedStructuralRecipe recipe;
  StructuredCandidateCost facts;
  std::string canonicalKey;
  std::string coverageClass;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_STRUCTUREDCANDIDATEDERIVATION_H
