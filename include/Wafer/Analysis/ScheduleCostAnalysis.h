//===- ScheduleCostAnalysis.h - Instruction program cost -------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_SCHEDULECOSTANALYSIS_H
#define WAFER_ANALYSIS_SCHEDULECOSTANALYSIS_H

#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/PhysicalIds.h"

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
  Unavailable,
  Unsupported,
  Overflow,
};

enum class ScheduleCostReason {
  None,
  DynamicLoopTripCount,
  InvalidLoopStep,
  ConditionalControlFlow,
  UnsupportedControlFlow,
  UnavailablePhysicalGeometry,
  UnavailableResourceBytes,
  MissingAcceptedSPMOffset,
  InvalidAcceptedSPMOffset,
  MissingAcceptedDDROffset,
  InvalidAcceptedDDROffset,
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

/// Hardware facts used to interpret the logical cost dimensions. Peak,
/// nominal, and conservative-bound fields are deliberately distinct:
/// reporting references must not silently become production bounds.
struct TargetScheduleCostPolicy {
  /// Whole-card peak/reference bandwidth. This remains the profiler's
  /// theoretical traffic-floor rate.
  uint64_t cardDDRBytesPerSecond = 200'000'000'000ULL;
  /// Whole-card observed operating point shared by all 16 tiles. It is a
  /// nominal estimate, not 16 independent per-tile rates and not a guaranteed
  /// throughput lower bound.
  uint64_t cardDDRNominalBytesPerSecond = 150'000'000'000ULL;
  /// Single-direction payload serialization reference. It is not a fabric
  /// aggregate, endpoint sustained rate, route estimate, or latency.
  uint64_t directionalNoCBytesPerSecond = 128'000'000'000ULL;
  /// Point estimates used by the numeric whole-card schedule model. These are
  /// compiler policy priors, not measured lower/upper bounds.
  ///
  /// The endpoint prior starts from one documented directional link. The
  /// startup prior is deliberately conservative for the current uncalibrated
  /// Direct-DTE software/handshake path; later matched board calibration may
  /// replace the value without changing the analytical formula.
  uint64_t dteEndpointBytesPerSecondEstimate = 128'000'000'000ULL;
  uint64_t dteMessageStartupPicosecondsEstimate = 10'000'000ULL;
  /// One model quantum per hop on the maximum modeled route. It is charged
  /// once as the route-fill/dilation term after link-congestion and endpoint
  /// service, not once per physical packet traversal.
  uint64_t noCHopPicosecondsEstimate = 1'000ULL;
  /// Fixed issue/release priors. Blocking resource service is accounted by the
  /// DDR/NoC/compute terms and is not charged again here.
  uint64_t instructionFixedPicosecondsEstimate = 1'000ULL;
  uint64_t dteWaitedEventPicosecondsEstimate = 1'000ULL;
  uint64_t nccParticipantWaitPicosecondsEstimate = 1'000ULL;
  uint64_t f16Bf16NPULogicalOpsPerSecondPerTile = 8'000'000'000'000ULL;
  uint64_t f16Bf16VectorLogicalOpsPerSecondPerTile = 64'000'000'000ULL;
  uint64_t f32VectorLogicalOpsPerSecondPerTile = 32'000'000'000ULL;
  /// Nominal service point for explicit SPM1 movement accounted by the
  /// instruction cost model. It comes from one 2048-bit bank at 1 GHz
  /// (256 GB/s per tile); it is a planning prior, not a sustained lower bound.
  uint64_t spmExplicitMovementBytesPerSecondPerTileEstimate =
      256'000'000'000ULL;

  uint64_t spmAddressBase = static_cast<uint64_t>(TargetMemoryPolicy{}.spmBase);
  uint64_t spmAddressLimit =
      static_cast<uint64_t>(TargetMemoryPolicy{}.spmLimit);
};

TargetScheduleCostPolicy getTargetScheduleCostPolicy();

struct ScheduleComputeCost {
  ScheduleCostMetric npuF16Bf16LogicalOps;
  ScheduleCostMetric npuOtherLogicalOps;
  ScheduleCostMetric vectorF16Bf16LogicalOps;
  ScheduleCostMetric vectorF32LogicalOps;
  ScheduleCostMetric vectorOtherLogicalOps;
};

struct ScheduleNoCCost {
  /// Exact executable send/receive operation sites, independent of dynamic
  /// loop multiplicity. This distinguishes a NoC-free program from one whose
  /// traffic multiplicity is unavailable.
  ScheduleCostMetric staticIssueSiteCount;
  /// Bytes injected by send instructions, counted once per logical payload.
  ScheduleCostMetric aggregateTransmitBytes;
  /// Bytes consumed by receive instructions. Kept separate to avoid treating a
  /// send/receive pair as two traversals of the fabric.
  ScheduleCostMetric aggregateReceiveBytes;
  /// Static execution multiplicity of actual Direct-DTE send/receive sites.
  /// Message startup must use these facts rather than guessing from event or
  /// instruction counts.
  ScheduleCostMetric transmitMessageCount;
  ScheduleCostMetric receiveMessageCount;
  /// Static wait operations and exact event operands consumed by them.
  ScheduleCostMetric waitOperationCount;
  ScheduleCostMetric waitedEventCount;
  std::array<ScheduleCostMetric, 4> directionalTransmitBytes;

  const ScheduleCostMetric &directional(NoCDirection direction) const;
};

/// One kind of final instruction-program work under structured control flow.
/// `staticSites` counts reachable work sites in the statically traversed entry
/// closure. `exactExecutions` is populated only when every enclosing trip count
/// and path is statically determined. The lower and upper bounds remain
/// independently useful when exact execution is not known; an unbounded or
/// unsupported upper bound is represented by its knowledge state, never zero.
struct InstructionExecutionCount {
  ScheduleCostMetric staticSites;
  ScheduleCostMetric exactExecutions;
  ScheduleCostMetric lowerBound;
  ScheduleCostMetric upperBound;
};

/// Same-source execution counters for the final instruction IR. Engine-family
/// counters describe issued work; operation-specific counters retain the
/// distinctions needed to audit movement, transport, and completion. A gather
/// or scatter therefore contributes to both `tdmaIssues` and
/// `gatherScatterOperations`, while a Direct-DTE send/receive contributes to
/// both `dteOperations` and its concrete send/receive counter.
struct InstructionProgramWork {
  InstructionExecutionCount instructions;
  InstructionExecutionCount asynchronousEvents;
  InstructionExecutionCount rdmaIssues;
  InstructionExecutionCount wdmaIssues;
  InstructionExecutionCount tdmaIssues;
  InstructionExecutionCount ctIssues;
  InstructionExecutionCount neIssues;
  InstructionExecutionCount dteOperations;
  InstructionExecutionCount gatherScatterOperations;
  InstructionExecutionCount dteSendOperations;
  InstructionExecutionCount dteReceiveOperations;
  InstructionExecutionCount dteWaitOperations;
  InstructionExecutionCount nccJoins;
  InstructionExecutionCount steadyStateNCCJoins;
  InstructionExecutionCount nonTerminalNCCJoins;
  InstructionExecutionCount nccParticipantWaits;
  InstructionExecutionCount steadyStateNCCParticipantWaits;
  InstructionExecutionCount nonTerminalNCCParticipantWaits;
  InstructionExecutionCount intrinsicNCCDrains;
};

struct InstructionProgramCost {
  InstructionProgramWork work;
  ScheduleComputeCost compute;
  ScheduleCostMetric ddrReadBytes;
  ScheduleCostMetric ddrWriteBytes;

  /// Payload bytes explicitly issued to the local movement resource. DDR
  /// traffic may also appear here because the two dimensions account for
  /// different constrained resources.
  ScheduleCostMetric spmMovementBytes;
  /// Bytes moved specifically by final gather/scatter instructions. This is a
  /// subset of `spmMovementBytes`, retained so layout movement is auditable
  /// without reparsing an IR dump.
  ScheduleCostMetric gatherScatterBytes;
  /// Maximum end offset among compiler-owned DDR allocations reachable from
  /// the analyzed entry closure. External invocation buffers are not part of
  /// this arena.
  ScheduleCostMetric ddrHighWaterBytes;
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

  /// Maximum accepted SPM address end relative to the target SPM base. This is
  /// address-space high-water, not liveness-aware peak allocation.
  ScheduleCostMetric spmHighWaterBytes;
  ScheduleCostMetric compilerOwnedSPMBufferCount;
  ScheduleCostMetric compilerOwnedDDRBufferCount;
};

enum class ModeledNoCRouteKind : uint8_t {
  CanonicalShortestPath,
};

/// A deterministic route-model result derived from final instruction traffic
/// and the typed topology. A Known metric means that every model input was
/// available and arithmetic completed; it does not assert that target hardware
/// uses this physical route.
struct ModeledNoCRouteCost {
  ModeledNoCRouteKind kind = ModeledNoCRouteKind::CanonicalShortestPath;
  /// Maximum accumulated payload bytes on any directed link when every final
  /// send follows the canonical shortest path selected by topology analysis.
  ScheduleCostMetric peakDirectedLinkByteDemand;
};

/// Exact whole-card aggregation of independently lowered physical-Tile
/// instruction programs.
/// Work and traffic dimensions are summed over the complete variant. SPM
/// remains private to a Tile, so both the maximum per-Tile high-water and the
/// sum of Tile-local high-waters are retained. No bandwidth-to-time conversion
/// is performed here.
struct WholeCardInstructionProgramCost {
  llvm::SmallVector<InstructionProgramCost, 16> tileCosts;

  /// Sum and per-dimension Tile maximum of the same Tile-local work facts.
  /// Maxima expose Tile-local pressure without pretending that every maximum
  /// came from one fictitious critical Tile.
  InstructionProgramWork aggregateWork;
  InstructionProgramWork maximumTileWork;
  ScheduleComputeCost aggregateCompute;
  ScheduleComputeCost maximumTileCompute;
  ScheduleCostMetric aggregateDDRReadBytes;
  ScheduleCostMetric aggregateDDRWriteBytes;
  ScheduleCostMetric aggregateSPMMovementBytes;
  ScheduleCostMetric aggregateGatherScatterBytes;
  /// Tile-local movement pressure is compared by Tile maximum. Aggregates
  /// remain available as whole-program work audit and are not substituted for
  /// this maximum.
  ScheduleCostMetric maximumTileSPMMovementBytes;
  ScheduleCostMetric maximumTileGatherScatterBytes;
  ScheduleNoCCost aggregateNoC;
  /// Endpoint pressure derived from the actual per-Tile instruction programs.
  /// These are maxima, not sums, because endpoints are tile-local resources.
  ScheduleCostMetric maximumTileNoCTransmitBytes;
  ScheduleCostMetric maximumTileNoCReceiveBytes;
  ScheduleCostMetric maximumTileNoCTransmitMessageCount;
  ScheduleCostMetric maximumTileNoCReceiveMessageCount;
  /// Sum over final send instructions of
  /// payload bytes * static execution multiplicity * minimum topology hops.
  /// This is a whole-domain link-byte demand lower bound, not an actual route,
  /// directional link load, congestion estimate, or execution time.
  ScheduleCostMetric minimumHopLinkByteDemand;
  /// Sum over final send instructions of static execution multiplicity *
  /// minimum topology hops. Unlike the byte demand above, this preserves the
  /// number of routed message-hop stages needed by a conservative latency
  /// upper bound.
  ScheduleCostMetric minimumHopMessageDemand;
  /// Directed adjacency links in the complete typed topology graph. This is
  /// counted once for each transmission direction and is never interpreted as
  /// an aggregate-bandwidth guarantee.
  ScheduleCostMetric directedNoCLinkCount;
  /// Route-independent lower bound on the most loaded directed link under an
  /// ideal balancing of the minimum-hop link-byte demand:
  /// ceil(minimumHopLinkByteDemand / directedNoCLinkCount). Real routing,
  /// cuts, endpoint pressure and congestion can only make the peak larger.
  ScheduleCostMetric idealizedMinimumPeakLinkByteDemand;
  /// Explicitly modeled link pressure. This is kept separate from exact
  /// final-IR work and route-independent lower bounds because the current
  /// target does not expose the hardware's selected physical routes.
  ModeledNoCRouteCost modeledNoCRoute;
  /// Maximum minimum-hop distance among final send sites that may execute.
  /// This remains a typed-topology lower bound; dynamic execution multiplicity
  /// does not change a site's source/destination distance.
  ScheduleCostMetric maximumNoCHopCount;
  ScheduleCostMetric aggregateInstructionCount;
  ScheduleCostMetric aggregateEventCount;
  ScheduleCostMetric aggregateNCCJoinCount;
  ScheduleCostMetric aggregateSteadyStateNCCJoinCount;
  ScheduleCostMetric aggregateNonTerminalNCCJoinCount;
  ScheduleCostMetric aggregateNCCParticipantWaitCount;
  ScheduleCostMetric aggregateSteadyStateNCCParticipantWaitCount;
  ScheduleCostMetric aggregateNonTerminalNCCParticipantWaitCount;
  ScheduleCostMetric aggregateIntrinsicNCCDrainCount;

  ScheduleCostMetric maximumTileSPMHighWaterBytes;
  ScheduleCostMetric summedTileSPMHighWaterBytes;
  ScheduleCostMetric maximumTileDDRHighWaterBytes;
  ScheduleCostMetric summedTileDDRHighWaterBytes;
  ScheduleCostMetric aggregateCompilerOwnedSPMBufferCount;
  ScheduleCostMetric aggregateCompilerOwnedDDRBufferCount;
  ScheduleCostMetric maximumTileCompilerOwnedSPMBufferCount;
  ScheduleCostMetric maximumTileCompilerOwnedDDRBufferCount;
};

/// Analyze a lowered instruction program without mutating it. Static scf.for
/// trip counts multiply execution costs. Dynamic or unsupported control flow,
/// missing accepted offsets, and arithmetic overflow are surfaced per metric
/// instead of being guessed or saturated.
InstructionProgramCost
analyzeInstructionProgramCost(mlir::Operation *root,
                              const TargetScheduleCostPolicy &policy);

/// One explicitly identified physical-Tile instruction program.  Identity is
/// supplied by the artifact owner and is never recovered from vector order,
/// module/function names, or logical partition metadata.
struct PhysicalTileInstructionProgram {
  PhysicalTileId tileId{0};
  mlir::Operation *root = nullptr;
};

/// One query-local partition of an accepted physical-Tile instruction
/// program. `includedOperations` identifies actual operations under `root`;
/// the analyzer still walks the complete structured control flow so selected
/// instructions retain their real loop/path multiplicity. Slices are an
/// analysis input only and are never serialized into compiler IR or a target
/// artifact.
struct PhysicalTileInstructionProgramSlice {
  PhysicalTileId tileId{0};
  mlir::Operation *root = nullptr;
  llvm::ArrayRef<mlir::Operation *> includedOperations;
};

/// Recompute every explicitly identified physical-Tile cost and aggregate the
/// complete card variant. Unavailable, unsupported and overflow states
/// propagate independently for each raw metric instead of being replaced with
/// estimates.
WholeCardInstructionProgramCost analyzeWholeCardInstructionProgramCost(
    llvm::ArrayRef<PhysicalTileInstructionProgram> tilePrograms,
    const TargetScheduleCostPolicy &policy);

/// Recompute one exact operation partition of the complete physical-Tile
/// domain. Operations not listed in a Tile slice contribute no work, but the
/// enclosing accepted control flow and explicit physical identity remain the
/// source of multiplicity and NoC topology. Callers must prove that a cohort
/// of slices is disjoint and conserves the unsliced raw costs before using it
/// to claim schedule overlap.
WholeCardInstructionProgramCost analyzeWholeCardInstructionProgramCostSlice(
    llvm::ArrayRef<PhysicalTileInstructionProgramSlice> tilePrograms,
    const TargetScheduleCostPolicy &policy);

llvm::StringRef stringifyScheduleCostKnowledge(ScheduleCostKnowledge knowledge);
llvm::StringRef stringifyScheduleCostReason(ScheduleCostReason reason);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_SCHEDULECOSTANALYSIS_H
