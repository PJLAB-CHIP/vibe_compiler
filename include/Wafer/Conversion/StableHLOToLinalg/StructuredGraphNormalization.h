//===- StructuredGraphNormalization.h - Graph normalization -*- C++ -*-===//

#ifndef WAFER_CONVERSION_STABLEHLOTOLINALG_STRUCTUREDGRAPHNORMALIZATION_H
#define WAFER_CONVERSION_STABLEHLOTOLINALG_STRUCTUREDGRAPHNORMALIZATION_H

#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace wafer {

struct StructuredGraphNormalizationOptions {
  uint64_t maximumRelationQueries = 8192;
  uint64_t maximumENodes = 4096;
  uint64_t maximumMatches = 8192;
  uint32_t maximumIterations = 8;
};

struct StructuredGraphNormalizationStatistics {
  uint64_t components = 0;
  uint64_t changedComponents = 0;
  uint64_t multiRuleChangedComponents = 0;
  uint64_t unchangedComponents = 0;
  uint64_t budgetExhaustedComponents = 0;
  uint64_t inputOperations = 0;
  uint64_t outputOperations = 0;
  uint64_t accessTransformsRemoved = 0;
  uint64_t multiUseAccessPropagations = 0;
  uint64_t concatTransformsRemoved = 0;
  uint64_t relationQueries = 0;
  uint64_t eNodes = 0;
  uint64_t eClasses = 0;
  uint64_t rewriteMatches = 0;
  uint64_t eClassMerges = 0;
  uint64_t rebuildWork = 0;
  uint64_t iterations = 0;
  uint64_t extractionWork = 0;
  uint64_t identityApplications = 0;
  uint64_t compositionApplications = 0;
  uint64_t computeAbsorptionApplications = 0;
  uint64_t concatApplications = 0;
  uint64_t resultReindexApplications = 0;
  uint64_t abiInputRecords = 0;
  uint64_t abiOutputRecords = 0;
  uint64_t abiInputBytes = 0;
  uint64_t abiOutputBytes = 0;
};

enum class StructuredGraphNormalizationOutcome {
  Changed,
  Unchanged,
  BudgetExhausted,
};

/// Runs the request-local access-relation e-graph over ordinary pure
/// single-root components in `function`. Standard pinned canonicalization is
/// owned by the immediately preceding pipeline stage.
mlir::FailureOr<StructuredGraphNormalizationOutcome>
normalizeStructuredTensorGraph(
    mlir::func::FuncOp function,
    const StructuredGraphNormalizationOptions &options = {},
    StructuredGraphNormalizationStatistics *statistics = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_STABLEHLOTOLINALG_STRUCTUREDGRAPHNORMALIZATION_H
