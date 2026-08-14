//===- Passes.h - Wafer transform pass registration ------------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_H
#define WAFER_TRANSFORMS_PASSES_H

#include <cstdint>
#include <memory>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

class TileRegionOp;

#define GEN_PASS_DECL
#include "Wafer/Transforms/WaferPasses.h.inc"

std::unique_ptr<mlir::Pass> createLegalizeStablehloToLinalgPass();
/// Preserve static concatenate semantics as canonical tensor insertion SSA
/// before the generic StableHLO-to-Linalg conversion expands it into scalar
/// index control flow.
std::unique_ptr<mlir::Pass> createLowerStaticStablehloConcatenatePass();

enum class SPMMemoryPlanningFailureKind : uint8_t {
  None,
  CapacityOverflow,
  UnsupportedLifetime,
  Other,
};

struct SPMMemoryPlanningFailure {
  SPMMemoryPlanningFailureKind kind = SPMMemoryPlanningFailureKind::None;
  /// Exact IR evidence from a proven capacity failure.  The location remains
  /// query-local and lets an upstream joint search identify which structured
  /// coordinate produced the largest actual SPM lifetime demand.  It is
  /// diagnostic/feedback evidence only; it never accepts a placement.
  mlir::LocationAttr largestDemandLocation;
  mlir::Type largestDemandType;
  uint64_t largestDemandBytes = 0;
  uint64_t demandCount = 0;
  struct DemandEvidence {
    mlir::LocationAttr location;
    mlir::Type type;
    uint64_t bytes = 0;
    /// Direct use-site locations of the actual allocation.  Buffer/view
    /// construction can legitimately keep the allocation at a support-op
    /// location while its typed consumers retain the causal source lineage.
    /// These locations are query-local rejection evidence only.
    llvm::SmallVector<mlir::LocationAttr, 4> userLocations;
  };
  /// Every actual IR demand tied for the largest byte size.  A caller may
  /// need more than one because equal-size allocations can carry different
  /// structured lineage and therefore control different joint-search
  /// coordinates.  The singular fields above remain the deterministic first
  /// representative for diagnostics and compatibility.
  llvm::SmallVector<DemandEvidence, 4> largestDemands;
  /// A sufficient exact capacity-conflict certificate from the final
  /// lifetime packing problem.  Every entry pair conflicts and the total
  /// bytes strictly exceed usable SPM.  Unlike `largestDemands`, this is a
  /// causal set rather than a same-size diagnostic cohort.
  llvm::SmallVector<DemandEvidence, 8> capacityConflictDemands;
  /// All actual lifetime demands that individually exceed usable SPM.  Each
  /// entry is independently impossible; unlike a clique certificate they do
  /// not assert pairwise lifetime overlap.
  llvm::SmallVector<DemandEvidence, 8> individuallyOversizedDemands;
};

mlir::LogicalResult
planSPMMemoryModule(mlir::ModuleOp moduleOp, int64_t spmBase, int64_t spmLimit,
                    int64_t spmAlignment,
                    SPMMemoryPlanningFailure *failure = nullptr);
/// Prove fixed-capacity SPM feasibility for one already isolated TileRegion.
/// This query derives lifetimes and packing from the region's current IR,
/// does not assign offsets, and does not inspect or mutate sibling regions.
mlir::LogicalResult
checkTileRegionSPMCapacity(TileRegionOp region, int64_t spmBase,
                           int64_t spmLimit, int64_t spmAlignment,
                           SPMMemoryPlanningFailure *failure = nullptr);
mlir::LogicalResult planDDRMemoryModule(mlir::ModuleOp moduleOp,
                                        int64_t ddrAlignmentBytes,
                                        int64_t ddrCapacityBytes,
                                        int64_t ddrLargestContiguousBytes,
                                        int64_t ddrBandwidthLimitBytes);
#ifdef WAFER_ENABLE_SHARDY
std::unique_ptr<mlir::Pass> createApplyDefaultSpmdShardingPass();
#endif
void registerWaferTransformPasses();

} // namespace wafer

#endif // WAFER_TRANSFORMS_PASSES_H
