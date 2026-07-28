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
  InvalidExecutionTopology,
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

  /// Explicit NCC participant-join operations after static execution
  /// multiplicity. This is kept separate from instructionCount because every
  /// join lowers to one or more blocking worker drains.
  ScheduleCostMetric nccJoinCount;
  /// Explicit NCC joins executed from a structured loop body. A legal
  /// zero-steady-state-join candidate is always preferred over one that drains
  /// an NCC worker in the steady kernel.
  ScheduleCostMetric steadyStateNCCJoinCount;
  /// Explicit NCC joins that are followed by more executable work on the same
  /// path, including joins in a loop body. Terminal joins are excluded.
  ScheduleCostMetric nonTerminalNCCJoinCount;
  /// Number of blocking worker waits implied by explicit participant joins and
  /// synchronous NCC writeback islands. A join of workers {0, 2} contributes
  /// two; a narrower scope is not treated as a cheaper wait.
  ScheduleCostMetric nccParticipantWaitCount;
  /// Blocking worker waits executed from a structured loop body. This is the
  /// primary steady-state drain metric: one join operation with three
  /// participants is three heavy waits, not one.
  ScheduleCostMetric steadyStateNCCParticipantWaitCount;
  /// Blocking worker waits followed by more executable work on the same path,
  /// including waits in a loop body. Terminal waits are excluded.
  ScheduleCostMetric nonTerminalNCCParticipantWaitCount;
  /// Blocking NCC drains hidden inside otherwise ordinary instructions, such
  /// as the current synchronous ArgMax/ArgMin host writeback path.
  ScheduleCostMetric intrinsicNCCDrainCount;

  /// Longest value/data-effect dependency chain in the accepted instruction
  /// IR. This is a structural count, not a cycle, latency, or overlap model.
  /// Unsupported control flow leaves it unknown without degrading the exact
  /// resource dimensions above.
  ScheduleCostMetric dataDependencyDepth;

  /// Count of target-static ready-priority inversions in fence-bounded final
  /// instruction order. This describes an actual order, not latency or
  /// overlap, and is used only after exact resources and dependency depth are
  /// equivalent.
  ScheduleCostMetric readyOrderPriorityInversions;

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
  /// Sum over final send instructions of
  /// payload bytes * static execution multiplicity * minimum topology hops.
  /// This is a whole-domain link-byte demand lower bound, not an actual route,
  /// directional link load, congestion estimate, or execution time.
  ScheduleCostMetric minimumHopLinkByteDemand;
  ScheduleCostMetric aggregateInstructionCount;
  ScheduleCostMetric aggregateEventCount;
  ScheduleCostMetric aggregateNCCJoinCount;
  ScheduleCostMetric aggregateSteadyStateNCCJoinCount;
  ScheduleCostMetric aggregateNonTerminalNCCJoinCount;
  ScheduleCostMetric aggregateNCCParticipantWaitCount;
  ScheduleCostMetric aggregateSteadyStateNCCParticipantWaitCount;
  ScheduleCostMetric aggregateNonTerminalNCCParticipantWaitCount;
  ScheduleCostMetric aggregateIntrinsicNCCDrainCount;

  /// Maximum rank-local structural data-dependency depth. It is retained for
  /// exact-resource-equivalent static policy tie-breaking, not summed as
  /// consumed work and not converted to time.
  ScheduleCostMetric maximumRankDataDependencyDepth;

  /// Sum of rank-local ready-priority inversions for exact-resource and
  /// dependency-depth-equivalent static policy tie-breaking.
  ScheduleCostMetric aggregateReadyOrderPriorityInversions;

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
