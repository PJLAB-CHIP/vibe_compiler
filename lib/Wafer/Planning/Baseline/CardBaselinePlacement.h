//===- CardBaselinePlacement.h ------------------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_BASELINE_CARDBASELINEPLACEMENT_H
#define WAFER_COMPILER_BASELINE_CARDBASELINEPLACEMENT_H

#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer::compiler::detail {

/// One parallel iterator whose first result can be partitioned by the
/// deterministic baseline. The complete Q50.B domain is deliberately not
/// represented here.
struct CardBaselineSpatialAxis {
  unsigned iteratorDimension = 0;
  unsigned resultDimension = 0;
  uint64_t extent = 0;
  llvm::SmallVector<uint32_t, 4> unitPartitionFactors;
};

std::optional<llvm::SmallVector<CardBaselineSpatialAxis, 4>>
getCardBaselineSpatialAxes(const StructuredDAGNode &node);

struct CardBaselineObservablePlacement {
  uint32_t outputIndex = 0;
  std::optional<unsigned> shardDimension;
  llvm::SmallVector<TileId, 16> tiles;
};

struct CardBaselinePlacementVerdict {
  analysis::ExactDemandStatus status = analysis::ExactDemandStatus::Satisfied;
  std::string detail;
};

struct CardBaselinePlacementClosure {
  llvm::SmallVector<StructuredDAGNodePlacement, 16> nodePlacements;
  llvm::SmallVector<CardBaselineObservablePlacement, 4> outputPlacements;
};

/// Closes one already chosen baseline coordinate through observable output
/// ownership and exact logical edge demand. It enumerates no alternatives and
/// builds no movement, resource schedule, or candidate score.
mlir::FailureOr<CardBaselinePlacementClosure> closeCardBaselinePlacement(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    analysis::IREpoch epoch, std::string *failureReason = nullptr,
    CardBaselinePlacementVerdict *verdict = nullptr);

enum class DeterministicSpatialAdvance : uint8_t {
  Advanced,
  Exhausted,
};

mlir::FailureOr<DeterministicSpatialAdvance>
advanceDeterministicSpatialCoordinate(
    llvm::SmallVectorImpl<StructuredDAGNodePlacement> &placements,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_BASELINE_CARDBASELINEPLACEMENT_H
