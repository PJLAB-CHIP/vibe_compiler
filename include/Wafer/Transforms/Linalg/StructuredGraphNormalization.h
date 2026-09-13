//===- StructuredGraphNormalization.h - Graph normalization -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHNORMALIZATION_H
#define WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHNORMALIZATION_H

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
  uint32_t maximumIterations = 32;
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
  uint64_t multiRootComponents = 0;
  uint64_t concatTransformsRemoved = 0;
  uint64_t relationQueries = 0;
  uint64_t eNodes = 0;
  uint64_t eClasses = 0;
  uint64_t rewriteMatches = 0;
  uint64_t applicationChecks = 0;
  uint64_t unchangedApplications = 0;
  uint64_t eClassMerges = 0;
  uint64_t rebuildWork = 0;
  uint64_t iterations = 0;
  uint64_t extractionWork = 0;
  uint64_t identityApplications = 0;
  uint64_t compositionApplications = 0;
  uint64_t computeAbsorptionApplications = 0;
  uint64_t concatApplications = 0;
  uint64_t resultReindexApplications = 0;
  uint64_t reshapeThroughComputeApplications = 0;
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
/// ordered multi-root components in `function`. Search limits retain proven
/// improvements when complete extraction succeeds. Pinned canonicalization is
/// owned by the immediately preceding pipeline stage.
mlir::FailureOr<StructuredGraphNormalizationOutcome>
normalizeStructuredTensorGraph(
    mlir::func::FuncOp function,
    const StructuredGraphNormalizationOptions &options = {},
    StructuredGraphNormalizationStatistics *statistics = nullptr);

/// Makes every static shaped function result a direct DPS/Tiling producer for
/// the physical-dataflow boundary. This is a policy-free current-IR
/// legalization performed after graph normalization; it is not an e-graph
/// rule or a persistent side plan.
mlir::LogicalResult closeStructuredProgramOutputs(mlir::func::FuncOp function);

} // namespace wafer

#endif // WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHNORMALIZATION_H
