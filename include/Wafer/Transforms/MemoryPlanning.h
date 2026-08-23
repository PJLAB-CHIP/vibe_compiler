//===- MemoryPlanning.h - Typed Wafer memory planning API -----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_MEMORYPLANNING_H
#define WAFER_TRANSFORMS_MEMORYPLANNING_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer {

class TileRegionOp;

enum class SPMMemoryPlanningFailureKind : uint8_t {
  None,
  CapacityOverflow,
  UnsupportedLifetime,
  Other,
};

struct SPMMemoryPlanningFailure {
  SPMMemoryPlanningFailureKind kind = SPMMemoryPlanningFailureKind::None;
  mlir::LocationAttr largestDemandLocation;
  mlir::Type largestDemandType;
  uint64_t largestDemandBytes = 0;
  uint64_t demandCount = 0;

  struct DemandEvidence {
    mlir::LocationAttr location;
    /// Invocation-local allocation root; never retain it past the queried IR
    /// epoch.
    mlir::Value allocation;
    mlir::Type type;
    uint64_t bytes = 0;
    llvm::SmallVector<mlir::LocationAttr, 4> userLocations;
    /// Diagnostic-only operation names captured before the candidate IR is
    /// destroyed. They must never participate in attribution or control flow.
    llvm::SmallVector<mlir::OperationName, 4> userOperationNames;
  };

  llvm::SmallVector<DemandEvidence, 4> largestDemands;
  llvm::SmallVector<DemandEvidence, 8> capacityConflictDemands;
  llvm::SmallVector<DemandEvidence, 8> individuallyOversizedDemands;
};

/// Direct query/apply kernel for callers that already own a private Module
/// transaction. Production pass pipelines should use the AnalysisManager-aware
/// SPM pass adapter.
mlir::LogicalResult
planSPMMemoryModule(mlir::ModuleOp moduleOp, int64_t spmBase, int64_t spmLimit,
                    int64_t spmAlignment,
                    SPMMemoryPlanningFailure *failure = nullptr);

/// Prove fixed-capacity SPM feasibility for one isolated TileRegion without
/// assigning offsets or inspecting sibling regions.
mlir::LogicalResult
checkTileRegionSPMCapacity(TileRegionOp region, int64_t spmBase,
                           int64_t spmLimit, int64_t spmAlignment,
                           SPMMemoryPlanningFailure *failure = nullptr);

/// Direct query/apply kernel for a caller-owned private Module. The caller
/// discards the Module on failure. Whole-executable lowering uses the
/// AnalysisManager-aware DDR pass adapter.
mlir::LogicalResult planDDRMemoryModule(mlir::ModuleOp moduleOp,
                                        int64_t ddrAlignmentBytes,
                                        int64_t ddrCapacityBytes,
                                        int64_t ddrLargestContiguousBytes,
                                        int64_t ddrBandwidthLimitBytes);

} // namespace wafer

#endif // WAFER_TRANSFORMS_MEMORYPLANNING_H
