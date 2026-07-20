//===- ScheduleTensorProgramInternal.h - Task scheduling API -*- C++ -*-===//
#pragma once

#include "Scheduling/StructuredSchedulingScope.h"
#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
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
#include <optional>
#include <string>
#include <vector>

namespace wafer::tensor_program_scheduling {

enum class TileSearchMode { FirstLegal, MinEstimatedTime };

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
  int64_t estimatedTimePs = 0;
  int64_t candidateCount = 0;
  int64_t rejectedCount = 0;
  int64_t representativeCount = 0;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  CandidateArtifactSource artifactSource =
      CandidateArtifactSource::RepresentativeTile;
};

struct CandidateWorkItem {
  CandidateSpec spec;
};

struct SelectionConfig {
  explicit SelectionConfig(const WaferTargetPolicy &policy)
      : preferredTileSizes(policy.tileSearch.preferredTileSizes),
        maxCandidatesPerDim(policy.tileSearch.maxCandidatesPerDim),
        maxSearchCandidates(policy.tileSearch.maxSearchCandidates),
        searchBeamWidth(policy.tileSearch.searchBeamWidth),
        spmBase(policy.memory.spmBase), spmLimit(policy.memory.spmLimit),
        spmAlignment(policy.memory.spmAlignment),
        ddrCapacityBytes(policy.memory.ddrCapacityBytes),
        ddrLargestContiguousBytes(policy.memory.ddrLargestContiguousBytes),
        ddrBandwidthLimitBytes(policy.memory.ddrBandwidthLimitBytes),
        ddrAlignmentBytes(policy.memory.ddrAlignmentBytes) {}

  TileSearchMode mode = TileSearchMode::FirstLegal;
  int64_t logicalRank = -1;
  llvm::SmallVector<int64_t, 8> preferredTileSizes;
  int64_t maxCandidatesPerDim = 0;
  int64_t maxSearchCandidates = 0;
  int64_t searchBeamWidth = 0;
  int64_t candidateParallelism = 1;
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
  CandidateArtifactSource artifactSource =
      CandidateArtifactSource::RepresentativeTile;
};

struct TileSizeOptions {
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 4> traversal;
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 2> reduction;
};

std::string getNearestSymbolName(mlir::Operation *op);

void printI64List(llvm::ArrayRef<int64_t> values, llvm::raw_ostream &os);

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
parseI64List(llvm::StringRef text, llvm::StringRef optionName,
             mlir::Operation *anchor);

mlir::FailureOr<TileSearchMode> parseTileSearchMode(llvm::StringRef text,
                                                    mlir::Operation *anchor);

mlir::FailureOr<TileSearchEffort>
parseTileSearchEffort(llvm::StringRef text, mlir::Operation *anchor);

int64_t saturatingAdd(int64_t lhs, int64_t rhs);

int64_t saturatingMul(int64_t lhs, int64_t rhs);

int64_t ceilDiv(int64_t numerator, int64_t denominator);

CandidateStats estimateStats(mlir::ModuleOp module);

std::optional<std::string> getRankingCostFailure(const CandidateStats &stats);

int64_t estimateCandidateTimePs(const CandidateStats &stats);

/// Returns true only when every exact execution-cost dimension is no worse
/// than the baseline and at least one is strictly lower.  This supplies a
/// deterministic ranking fallback when an uncalibrated compute class makes
/// both coarse scalar time estimates saturate.
bool hasStrictExecutionCostDominance(const CandidateStats &candidate,
                                     const CandidateStats &baseline);

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticTraversalShape(mlir::func::FuncOp task);

std::optional<int64_t> getElementByteWidth(mlir::Type type);

std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
getYieldedRootLinalgOps(mlir::func::FuncOp task);

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

mlir::FailureOr<SelectedCandidate> selectCandidateForScope(
    const structured_scheduler::StructuredSchedulingScope &scope,
    mlir::func::FuncOp task, llvm::StringRef label,
    const SelectionConfig &config);

void printSelectedSummary(const SelectedCandidate &selected,
                          TileSearchMode mode);

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

} // namespace wafer::tensor_program_scheduling
