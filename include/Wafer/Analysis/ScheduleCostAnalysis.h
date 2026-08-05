//===- ScheduleCostAnalysis.h - Instruction program cost -------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_SCHEDULECOSTANALYSIS_H
#define WAFER_ANALYSIS_SCHEDULECOSTANALYSIS_H

#include "Wafer/Support/TargetPolicy.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>

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
  MissingAcceptedDDROffset,
  InvalidAcceptedDDROffset,
  UnsupportedSPMRoot,
  UnresolvedNoCRoute,
  InvalidExecutionTopology,
  UnsupportedInstructionSemantics,
  UnsupportedComputeType,
  MissingPerformanceCalibration,
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

/// Hardware facts used to interpret the logical cost dimensions. Peak,
/// nominal, and conservative-bound fields are deliberately distinct:
/// reporting references must not silently become production bounds.
struct TargetScheduleCostPolicy {
  /// Exact same-worker NCC engine group with compiler-shipped overlap
  /// qualification. A zero mask means the current target has no qualified
  /// group. Bits use the closed InstrFamily enum values; this is an ordinal
  /// profitability fact, not a latency estimate.
  uint32_t qualifiedOverlapFamilyMask = 0;
  uint32_t qualifiedOverlapWorker = 0;
  /// Direct-DTE plus compute families known to use independent target
  /// resources when typed issue and wait delimit an overlap window.
  uint32_t qualifiedDirectDTEOverlapFamilyMask = 0;
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
  /// Point estimates used by the versioned analytical selector. These are
  /// compiler policy priors, not measured lower/upper bounds. Keeping them
  /// separate from the conservative fields below lets normal production make
  /// an Estimated decision without misreporting it as a proof.
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
  /// SPM has a documented 1024-bit internal interface but no qualified
  /// sustained bandwidth. One 128-byte beat per 1 GHz model quantum is an
  /// explicit point prior, not a hardware bound.
  uint64_t spmBytesPerSecondPerTileEstimate = 128'000'000'000ULL;
  /// Fixed issue/release priors. Blocking resource service is accounted by the
  /// DDR/NoC/compute terms and is not charged again here.
  uint64_t instructionFixedPicosecondsEstimate = 1'000ULL;
  uint64_t dteWaitedEventPicosecondsEstimate = 1'000ULL;
  uint64_t nccParticipantWaitPicosecondsEstimate = 1'000ULL;
  uint64_t f16Bf16NPULogicalOpsPerSecondPerTile = 8'000'000'000'000ULL;
  uint64_t f16Bf16VectorLogicalOpsPerSecondPerTile = 64'000'000'000ULL;
  uint64_t f32VectorLogicalOpsPerSecondPerTile = 32'000'000'000ULL;

  /// Optional conservative bounds used only by production profitability.
  /// The current profile intentionally leaves these absent: existing board
  /// evidence establishes the references above but not sustained lower rates,
  /// Direct-DTE startup/hop upper bounds, or route dilation.
  std::optional<uint64_t> cardDDRSustainedBytesPerSecondLowerBound;
  std::optional<uint64_t> directionalNoCSustainedBytesPerSecondLowerBound;
  std::optional<uint64_t> dteEndpointBytesPerSecondLowerBound;
  std::optional<uint64_t> dteMessageStartupPicosecondsUpperBound;
  /// Conservative per-hop route-fill time used with the dilated aggregate
  /// minimum-hop message demand.
  std::optional<uint64_t> noCHopPicosecondsUpperBound;
  std::optional<uint32_t> noCRouteDilationUpperBound;
  std::optional<uint64_t> f16Bf16NPULogicalOpsPerSecondPerTileLowerBound;
  std::optional<uint64_t> f16Bf16VectorLogicalOpsPerSecondPerTileLowerBound;
  std::optional<uint64_t> f32VectorLogicalOpsPerSecondPerTileLowerBound;
  std::optional<uint64_t> spmBytesPerSecondPerTileLowerBound;
  /// Maximum fixed non-service time per statically executed instruction.
  /// Resource service time is modeled separately; this bound covers issue,
  /// control and nonblocking completion bookkeeping.
  std::optional<uint64_t> instructionFixedPicosecondsUpperBound;
  /// Additional blocking-poll/release overhead after the corresponding
  /// resource service has completed.
  std::optional<uint64_t> dteWaitedEventPicosecondsUpperBound;
  std::optional<uint64_t> nccParticipantWaitPicosecondsUpperBound;

  /// Compiler safety margin, not a measured hardware rate. Estimated winners
  /// must retain at least a 20% advantage under the central point model.
  uint32_t productionBenefitMarginPermille = 200;
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
  /// traffic multiplicity is Unknown.
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
  std::array<ScheduleCostMetric, 5> collectiveTransmitBytes;

  const ScheduleCostMetric &directional(NoCDirection direction) const;
  const ScheduleCostMetric &collective(NoCCollectiveKind kind) const;
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
/// both `dteOperations` and exactly one protocol-class counter.
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
  InstructionExecutionCount collectiveDTEIssues;
  InstructionExecutionCount peerDTEIssues;
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

  /// Number of static rotating-buffer loops whose direct instruction window
  /// exactly matches a compiler-shipped target overlap capability. Higher is
  /// preferred before capacity high-water; it is an ordinal IR fact, not a
  /// cycle or time estimate.
  ScheduleCostMetric qualifiedOverlapWindowCount;

  /// Count of accepted same-block windows with one explicit bound Direct-DTE
  /// issue, an independent FP16/BF16 CT/NE instruction, and the matching exact
  /// wait. Fixed-slot qualification proves the originating rotating SPM
  /// realization separately because endpoint specialization may replace the
  /// recurrence with exact roots. This is a structural V3 witness; it does not
  /// assert temporal overlap or profitability on hardware.
  ScheduleCostMetric directDTEComputeOverlapWindowCount;

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

/// Exact all-rank aggregation of independently lowered instruction programs.
/// Work and traffic dimensions are summed over the complete variant. SPM
/// remains private to a tile, so both the maximum per-rank high-water and the
/// sum of rank-local high-waters are retained. No bandwidth-to-time conversion
/// is performed here: the current target does not establish issue
/// timing or cross-resource overlap.
struct WholeCardInstructionProgramCost {
  llvm::SmallVector<InstructionProgramCost, 16> rankCosts;

  /// Sum and per-dimension rank maximum of the same rank-local work facts.
  /// Maxima expose rank-local pressure without pretending that every maximum
  /// came from one fictitious critical rank.
  InstructionProgramWork aggregateWork;
  InstructionProgramWork maximumRankWork;
  ScheduleComputeCost aggregateCompute;
  ScheduleCostMetric aggregateDDRReadBytes;
  ScheduleCostMetric aggregateDDRWriteBytes;
  ScheduleCostMetric aggregateSPMMovementBytes;
  ScheduleCostMetric aggregateGatherScatterBytes;
  ScheduleNoCCost aggregateNoC;
  /// Endpoint pressure derived from the actual per-rank instruction programs.
  /// These are maxima, not sums, because endpoints are tile-local resources.
  ScheduleCostMetric maximumRankNoCTransmitBytes;
  ScheduleCostMetric maximumRankNoCReceiveBytes;
  ScheduleCostMetric maximumRankNoCTransmitMessageCount;
  ScheduleCostMetric maximumRankNoCReceiveMessageCount;
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

  /// Maximum rank-local structural data-dependency depth. It is retained for
  /// exact-resource-equivalent static policy tie-breaking, not summed as
  /// consumed work and not converted to time.
  ScheduleCostMetric maximumRankDataDependencyDepth;

  /// Sum of rank-local ready-priority inversions for exact-resource and
  /// dependency-depth-equivalent static policy tie-breaking.
  ScheduleCostMetric aggregateReadyOrderPriorityInversions;

  /// Sum of target-qualified rotating-buffer overlap windows across ranks.
  ScheduleCostMetric aggregateQualifiedOverlapWindowCount;

  /// Sum of explicit Direct-DTE issue/compute/exact-wait windows across ranks.
  /// A Known value is an accepted-IR structural fact, not a timing estimate.
  ScheduleCostMetric aggregateDirectDTEComputeOverlapWindowCount;

  ScheduleCostMetric maximumRankSPMHighWaterBytes;
  ScheduleCostMetric summedRankSPMHighWaterBytes;
  ScheduleCostMetric maximumRankDDRHighWaterBytes;
  ScheduleCostMetric summedRankDDRHighWaterBytes;
  ScheduleCostMetric aggregateCompilerOwnedSPMBufferCount;
  ScheduleCostMetric aggregateCompilerOwnedDDRBufferCount;
  ScheduleCostMetric maximumRankCompilerOwnedSPMBufferCount;
  ScheduleCostMetric maximumRankCompilerOwnedDDRBufferCount;
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
