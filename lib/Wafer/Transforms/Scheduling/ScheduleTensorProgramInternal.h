//===- ScheduleTensorProgramInternal.h - Task scheduling API -*- C++ -*-===//
#pragma once

#include "Scheduling/StructuredSchedulingScope.h"
#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/Internal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer::tensor_program_scheduling {

using tile_region_to_instr::AllGatherSchedule;
using tile_region_to_instr::AllReduceSchedule;
using tile_region_to_instr::ReduceScatterSchedule;
using tile_region_to_instr::TileRegionToInstrOptions;

enum class CommunicationAlternative {
  Ring,
  DirectAllGather,
  RingReduceScatter,
  TreeAllReduce,
  DirectAllGatherTreeAllReduce,
};

struct CandidateSpec {
  llvm::SmallVector<int64_t, 4> tileSizes;
  llvm::SmallVector<int64_t, 2> reductionSplitSizes;
  std::optional<TargetImplementationKind> selectedImplementationAlternative;
};

struct TileInstance {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

struct CandidateStats {
  analysis::InstructionProgramCost program;
};

enum class CandidateArtifactSource {
  RepresentativeTile,
  CompleteTileInstance,
  CompleteTraversalAPI,
  FullTraversalFallback,
};

inline bool isCompleteArtifactSource(CandidateArtifactSource source) {
  return source != CandidateArtifactSource::RepresentativeTile;
}

struct CandidateEvaluation {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  CandidateStats stats;
  std::string failureReason;
  CandidateArtifactSource artifactSource =
      CandidateArtifactSource::RepresentativeTile;
};

struct SelectedCandidate {
  std::string label;
  structured_scheduler::StructuredSchedulingScope scope;
  /// Owns a producer-mutated structured source when this selection did not
  /// originate from the baseline task. The source task handle below always
  /// points either into this module or into the caller-owned baseline module.
  mlir::OwningOpRef<mlir::ModuleOp> sourceModule;
  mlir::func::FuncOp sourceTask;
  CandidateSpec spec;
  CandidateStats stats;
  int64_t candidateCount = 0;
  int64_t completeEvaluationCount = 0;
  int64_t rejectedCount = 0;
  int64_t representativeCount = 0;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  CandidateArtifactSource artifactSource =
      CandidateArtifactSource::RepresentativeTile;
};

struct CandidateWorkItem {
  CandidateSpec spec;
};

class CandidateEvaluationExecutor;

struct SelectionConfig {
  explicit SelectionConfig(const WaferTargetPolicy &policy,
                           TargetProfileId targetProfile)
      : scheduleCostPolicy(
            analysis::getTargetScheduleCostPolicy(targetProfile)),
        preferredTileSizes(policy.tileSearch.preferredTileSizes),
        maxCandidatesPerDim(policy.tileSearch.maxCandidatesPerDim),
        maxSearchCandidates(policy.tileSearch.maxSearchCandidates),
        searchBeamWidth(policy.tileSearch.searchBeamWidth),
        spmBase(policy.memory.spmBase), spmLimit(policy.memory.spmLimit),
        spmAlignment(policy.memory.spmAlignment),
        ddrCapacityBytes(policy.memory.ddrCapacityBytes),
        ddrLargestContiguousBytes(policy.memory.ddrLargestContiguousBytes),
        ddrBandwidthLimitBytes(policy.memory.ddrBandwidthLimitBytes),
        ddrAlignmentBytes(policy.memory.ddrAlignmentBytes) {}

  int64_t logicalRank = -1;
  /// Immutable compiler-shipped target-contract interpretation of every
  /// candidate cost. Copies sent to evaluation workers preserve the exact
  /// static production contract; no live-card/profile state is consulted.
  analysis::TargetScheduleCostPolicy scheduleCostPolicy;
  llvm::SmallVector<int64_t, 8> preferredTileSizes;
  int64_t maxCandidatesPerDim = 0;
  int64_t maxSearchCandidates = 0;
  int64_t searchBeamWidth = 0;
  int64_t candidateParallelism = 1;
  /// Invocation-local execution service owned by the enclosing rank-frontier
  /// build. It affects only bounded evaluation execution; candidate semantics,
  /// ordering and accepted artifacts remain in the IR.
  CandidateEvaluationExecutor *evaluationExecutor = nullptr;
  /// Selects one compiler-private communication rewrite parameter point. The
  /// choice is materialized in an actual clone and never persisted as an IR
  /// attribute or artifact field.
  CommunicationAlternative communicationAlternative =
      CommunicationAlternative::Ring;
  /// Requests a bounded alternative in stable complete-candidate generation
  /// order. A missing ordinal rejects only the enclosing optimized rank recipe.
  unsigned taskAlternativeOrdinal = 0;
  /// Rank-frontier recipes disable implicit implementation enumeration so the
  /// reserved tuple remains a true source-interface baseline and each
  /// optimized implementation is represented by its own complete clone.
  bool allowAutomaticImplementationAlternatives = true;
  /// When present, eligible source ops are evaluated only with this
  /// implementation form; ineligible tasks retain their ordinary baseline.
  std::optional<TargetImplementationKind> forcedImplementationAlternative;
  /// When true, exact cross-space identity transfers may load/store directly
  /// between a DDR Tensor boundary and the selected Cx/NCx SPM version.
  /// Otherwise the candidate materializes the conservative Tensor staging
  /// route. The choice is reflected only by actual movement IR.
  bool useDirectMappedBoundaryTransfer = false;
  bool printCandidateSummary = false;
  int64_t spmBase = 0;
  int64_t spmLimit = 0;
  int64_t spmAlignment = 0;
  int64_t ddrCapacityBytes = 0;
  int64_t ddrLargestContiguousBytes = 0;
  int64_t ddrBandwidthLimitBytes = 0;
  int64_t ddrAlignmentBytes = 0;
};

struct CandidateCheckResult {
  CandidateSpec spec;
  CandidateStats stats;
  std::string failureReason;
  int64_t representativeCount = 0;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  /// Invocation-local transport for a fully accepted module produced in a
  /// worker-owned MLIRContext. This is imported and cleared by the owner; it
  /// is never an artifact identity, cache key, or serialized frontier.
  std::string acceptedModuleText;
  CandidateArtifactSource artifactSource =
      CandidateArtifactSource::RepresentativeTile;
};

/// Bounded rank-frontier executor. Each persistent worker owns one MLIRContext
/// for the full frontier invocation and reuses the last parsed standalone task
/// when its text is unchanged. Results are consumed in submission order.
class CandidateEvaluationExecutor {
public:
  explicit CandidateEvaluationExecutor(unsigned workerCount);
  ~CandidateEvaluationExecutor();

  CandidateEvaluationExecutor(const CandidateEvaluationExecutor &) = delete;
  CandidateEvaluationExecutor &
  operator=(const CandidateEvaluationExecutor &) = delete;

  std::future<CandidateCheckResult>
  submit(std::shared_ptr<const std::string> standaloneTaskModuleText,
         llvm::ArrayRef<int64_t> traversalShape, const CandidateSpec &candidate,
         const SelectionConfig &config);

  unsigned getWorkerCount() const;
  unsigned getWorkerConstructionCount() const;
  unsigned getContextConstructionCount() const;
  unsigned getTaskParseCount() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

struct TileSizeOptions {
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 4> traversal;
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 2> reduction;
};

std::string getNearestSymbolName(mlir::Operation *op);

void printI64List(llvm::ArrayRef<int64_t> values, llvm::raw_ostream &os);

int64_t saturatingAdd(int64_t lhs, int64_t rhs);

int64_t saturatingMul(int64_t lhs, int64_t rhs);

int64_t ceilDiv(int64_t numerator, int64_t denominator);

CandidateStats
estimateStats(mlir::ModuleOp module,
              const analysis::TargetScheduleCostPolicy &scheduleCostPolicy);

std::optional<std::string> getRankingCostFailure(const CandidateStats &stats);

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticTraversalShape(mlir::func::FuncOp task);

std::optional<int64_t> getElementByteWidth(mlir::Type type);

std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
getYieldedRootLinalgOps(mlir::func::FuncOp task);

/// Returns the Linalg compute roots used only to direct traversal-pressure and
/// target-capacity analysis. In addition to direct yielded Linalg roots, this
/// may look through a verified single-input/single-result shape-preserving
/// all-reduce to its unique direct Linalg producer. Reduction-range and
/// reduction-split legality must continue to use getYieldedRootLinalgOps so an
/// upstream SPMD contracting shard is not reinterpreted as a local K split.
std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
getTraversalComputeRootLinalgOps(mlir::func::FuncOp task);

CandidateTraversalRootCapability
getTaskTraversalRootCapability(mlir::func::FuncOp task);

std::string getStandaloneTaskModuleText(mlir::func::FuncOp task);

mlir::func::FuncOp findSingleSelectionTask(mlir::ModuleOp module);

mlir::FailureOr<llvm::SmallVector<int64_t, 2>>
getStaticRootReductionRanges(mlir::func::FuncOp task);

std::optional<std::string>
getReductionSplitLegalityFailure(mlir::func::FuncOp task);

bool failsCheapSPMBound(mlir::func::FuncOp task, const CandidateSpec &candidate,
                        int64_t spmBase, int64_t spmLimit);

/// Returns a conservative target-physical SPM root inventory only when the
/// tensor program shape and lowering contract are closed enough to direct
/// search. This is a queue-ordering hint, not a rejection proof; accepted
/// candidates still pass the complete instruction and SPM-planning gates.
std::optional<int64_t>
estimateTargetSPMWorkingSetBytes(mlir::func::FuncOp task,
                                 const CandidateSpec &candidate,
                                 int64_t spmAlignment);

/// Returns the target-physical roots that are provably live together for the
/// same closed direct/fused-transpose matmul contracts. Unlike the
/// conservative working-set inventory above, this value may be used as an
/// early rejection bound; accepted candidates still pass complete planning.
std::optional<int64_t>
estimateTargetSPMRequiredLiveBytes(mlir::func::FuncOp task,
                                   const CandidateSpec &candidate,
                                   int64_t spmAlignment);

std::optional<std::string>
getCheapTargetGeometryFailure(mlir::func::FuncOp task,
                              const CandidateSpec &candidate,
                              llvm::ArrayRef<int64_t> reductionRanges);

TileSizeOptions buildTileSizeOptions(llvm::ArrayRef<int64_t> traversalShape,
                                     llvm::ArrayRef<int64_t> reductionRanges,
                                     llvm::ArrayRef<int64_t> preferred,
                                     int64_t maxCandidatesPerDim,
                                     bool allowReductionSplits);

bool isFullFirstTile(llvm::ArrayRef<int64_t> traversalShape,
                     const TileInstance &tile);

llvm::SmallVector<TileInstance, 8>
buildRepresentativeTiles(llvm::ArrayRef<int64_t> traversalShape,
                         llvm::ArrayRef<int64_t> tileSizes);

CandidateEvaluation evaluateCompleteCandidate(
    mlir::func::FuncOp task, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config);

CandidateCheckResult evaluateCandidateOnOriginalTask(
    mlir::func::FuncOp task, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config);

CandidateCheckResult
evaluateCandidateOnStandaloneTaskText(llvm::StringRef standaloneTaskModuleText,
                                      llvm::ArrayRef<int64_t> traversalShape,
                                      const CandidateSpec &candidate,
                                      const SelectionConfig &config);

/// Imports a fully accepted worker result into the owner context, then
/// verifies the imported module and recomputes its ranking facts from that IR.
/// A failure is recorded on `result` and no partial module is retained.
mlir::LogicalResult importAcceptedCandidateModule(
    CandidateCheckResult &result, mlir::MLIRContext &ownerContext,
    const analysis::TargetScheduleCostPolicy &scheduleCostPolicy);

mlir::FailureOr<SelectedCandidate> selectCandidateForScope(
    const structured_scheduler::StructuredSchedulingScope &scope,
    mlir::func::FuncOp task, llvm::StringRef label,
    const SelectionConfig &config);

mlir::LogicalResult commitSelectedTaskCandidate(SelectedCandidate &selected,
                                                const SelectionConfig &config);

mlir::LogicalResult accumulateStaticTerminalOperations(mlir::Operation *root,
                                                       uint64_t &count);

/// Promotes closed full-buffer producer/consumer boundaries from a DDR spill
/// to an explicit sibling tile-region SPM SSA handoff.  A boundary is changed
/// only when one complete producer WDMA and every external consumer RDMA are
/// statically provable full-buffer transfers.  The caller remains responsible
/// for whole-rank resource planning and for retaining a spill fallback.
unsigned promoteFullBufferHandoffs(mlir::ModuleOp module);

/// Coalesces a movement-defined SPM allocation with its exact source storage
/// only when relation, physical encoding, alias/effect, snapshot, alignment,
/// base-preserving view provenance, ownership, and completion facts all prove
/// that no observable state is lost. Unknown or unsupported cases retain their
/// explicit movement. Production callers apply this only to optional sibling
/// clones and preserve the conservative spill fallback.
unsigned elideRedundantFullBufferTransfers(mlir::ModuleOp module);

} // namespace wafer::tensor_program_scheduling
