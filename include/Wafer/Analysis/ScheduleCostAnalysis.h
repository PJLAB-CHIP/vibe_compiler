//===- ScheduleCostAnalysis.h - Instruction program cost -------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_SCHEDULECOSTANALYSIS_H
#define WAFER_ANALYSIS_SCHEDULECOSTANALYSIS_H

#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetProfile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>

namespace mlir {
class Operation;
} // namespace mlir

namespace wafer::analysis {

/// The knowledge state is carried independently for every cost dimension.
/// Exact traffic can therefore remain usable when, for example, a NoC route
/// has not yet been selected.
enum class ScheduleCostKnowledge {
  Known,
  Unknown,
  Unsupported,
  Overflow,
};

enum class ScheduleCostReason {
  None,
  DynamicLoopTripCount,
  InvalidLoopStep,
  ConditionalControlFlow,
  UnsupportedControlFlow,
  UnknownPhysicalGeometry,
  UnknownResourceBytes,
  MissingAcceptedSPMOffset,
  InvalidAcceptedSPMOffset,
  UnsupportedSPMRoot,
  UnresolvedNoCRoute,
  UnsupportedInstructionSemantics,
  UnsupportedComputeType,
  ArithmeticOverflow,
};

struct ScheduleCostMetric {
  uint64_t value = 0;
  ScheduleCostKnowledge knowledge = ScheduleCostKnowledge::Known;
  ScheduleCostReason reason = ScheduleCostReason::None;

  bool isKnown() const { return knowledge == ScheduleCostKnowledge::Known; }
};

enum class NoCDirection : uint8_t { North, South, East, West };

enum class NoCCollectiveKind : uint8_t {
  CollectivePermute,
  AllToAll,
  AllGather,
  ReduceScatter,
  AllReduce,
};

/// Hardware facts used to interpret the logical cost dimensions. This policy
/// deliberately has no SPM bandwidth, issue latency, clock, or cycle estimate:
/// those facts are not established by the current target contract.
struct TargetScheduleCostPolicy {
  explicit constexpr TargetScheduleCostPolicy(TargetProfileId targetProfile)
      : targetProfile(targetProfile) {}

  TargetProfileId targetProfile;
  uint64_t cardDDRBytesPerSecond = 200'000'000'000ULL;
  uint64_t directionalNoCBytesPerSecond = 128'000'000'000ULL;
  uint64_t f16Bf16NPULogicalOpsPerSecondPerTile = 8'000'000'000'000ULL;
  uint64_t f16Bf16VectorLogicalOpsPerSecondPerTile = 64'000'000'000ULL;
  uint64_t f32VectorLogicalOpsPerSecondPerTile = 32'000'000'000ULL;
  uint64_t spmAddressBase = static_cast<uint64_t>(TargetMemoryPolicy{}.spmBase);
  uint64_t spmAddressLimit =
      static_cast<uint64_t>(TargetMemoryPolicy{}.spmLimit);
};

TargetScheduleCostPolicy
getTargetScheduleCostPolicy(TargetProfileId targetProfile);

struct ScheduleComputeCost {
  ScheduleCostMetric npuF16Bf16LogicalOps;
  ScheduleCostMetric npuOtherLogicalOps;
  ScheduleCostMetric vectorF16Bf16LogicalOps;
  ScheduleCostMetric vectorF32LogicalOps;
  ScheduleCostMetric vectorOtherLogicalOps;
};

struct ScheduleNoCCost {
  /// Bytes injected by send instructions, counted once per logical payload.
  ScheduleCostMetric aggregateTransmitBytes;
  /// Bytes consumed by receive instructions. Kept separate to avoid treating a
  /// send/receive pair as two traversals of the fabric.
  ScheduleCostMetric aggregateReceiveBytes;
  std::array<ScheduleCostMetric, 4> directionalTransmitBytes;
  std::array<ScheduleCostMetric, 5> collectiveTransmitBytes;

  const ScheduleCostMetric &directional(NoCDirection direction) const;
  const ScheduleCostMetric &collective(NoCCollectiveKind kind) const;
};

struct InstructionProgramCost {
  ScheduleComputeCost compute;
  ScheduleCostMetric ddrReadBytes;
  ScheduleCostMetric ddrWriteBytes;

  /// Payload bytes explicitly issued to the local movement resource. DDR
  /// traffic may also appear here because the two dimensions account for
  /// different constrained resources.
  ScheduleCostMetric spmMovementBytes;
  ScheduleNoCCost noc;
  ScheduleCostMetric instructionCount;
  /// Number of asynchronous completion-token results materialized by the
  /// instruction program, not the number of waits consuming those tokens.
  ScheduleCostMetric eventCount;

  /// Maximum accepted SPM address end relative to the target SPM base. This is
  /// address-space high-water, not liveness-aware peak allocation.
  ScheduleCostMetric spmHighWaterBytes;
};

/// Exact all-rank aggregation of independently lowered instruction programs.
/// Work and traffic dimensions are summed over the complete variant. SPM
/// remains private to a tile, so both the maximum per-rank high-water and the
/// sum of rank-local high-waters are retained. No bandwidth-to-time conversion
/// is performed here: the current target contract does not establish issue
/// timing or cross-resource overlap.
struct WholeCardInstructionProgramCost {
  llvm::SmallVector<InstructionProgramCost, 16> rankCosts;

  ScheduleComputeCost aggregateCompute;
  ScheduleCostMetric aggregateDDRReadBytes;
  ScheduleCostMetric aggregateDDRWriteBytes;
  ScheduleCostMetric aggregateSPMMovementBytes;
  ScheduleNoCCost aggregateNoC;
  ScheduleCostMetric aggregateInstructionCount;
  ScheduleCostMetric aggregateEventCount;

  ScheduleCostMetric maximumRankSPMHighWaterBytes;
  ScheduleCostMetric summedRankSPMHighWaterBytes;
};

/// Analyze a lowered instruction program without mutating it. Static scf.for
/// trip counts multiply execution costs. Dynamic or unsupported control flow,
/// missing accepted offsets, and arithmetic overflow are surfaced per metric
/// instead of being guessed or saturated.
InstructionProgramCost
analyzeInstructionProgramCost(mlir::Operation *root,
                              const TargetScheduleCostPolicy &policy);

/// Recompute every rank cost in canonical caller order and aggregate the
/// complete card variant. Unknown, unsupported and overflow states propagate
/// independently for each metric instead of being replaced with estimates.
WholeCardInstructionProgramCost analyzeWholeCardInstructionProgramCost(
    llvm::ArrayRef<mlir::Operation *> rankRoots,
    const TargetScheduleCostPolicy &policy);

llvm::StringRef stringifyScheduleCostKnowledge(ScheduleCostKnowledge knowledge);
llvm::StringRef stringifyScheduleCostReason(ScheduleCostReason reason);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_SCHEDULECOSTANALYSIS_H
