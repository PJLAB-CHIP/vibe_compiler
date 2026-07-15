//===- SelectGroupTileInternal.h - Tile selection private API -*- C++ -*-===//
#pragma once

#include "Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"

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

namespace wafer::group_tile_selection {

enum class TileSearchMode { FirstLegal, MinEstimatedTime };

struct CandidateSpec {
  llvm::SmallVector<int64_t, 4> tileSizes;
  llvm::SmallVector<int64_t, 2> reductionSplitSizes;
};

struct TileInstance {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

struct CandidateStats {
  int64_t computeOps = 0;
  int64_t ddrBytes = 0;
  int64_t spmBytes = 0;
  int64_t instrCount = 0;
};

enum class CandidateArtifactSource {
  RepresentativeTile,
  CompleteTileInstance,
  CompleteTraversalAPI,
  FullGroupFallback,
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
  GroupOp group;
  CandidateSpec spec;
  CandidateStats stats;
  int64_t estimatedCycles = 0;
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
        ddrAlignmentBytes(policy.memory.ddrAlignmentBytes),
        computeOpsPerCycle(policy.timing.computeOpsPerCycle),
        ddrBytesPerCycle(policy.timing.ddrBytesPerCycle),
        spmBytesPerCycle(policy.timing.spmBytesPerCycle),
        instrIssueCycles(policy.timing.instrIssueCycles),
        assumeDdrComputeOverlap(policy.timing.assumeDdrComputeOverlap) {}

  TileSearchMode mode = TileSearchMode::FirstLegal;
  int64_t logicalRank = -1;
  llvm::SmallVector<int64_t, 8> preferredTileSizes;
  int64_t maxCandidatesPerDim = 0;
  int64_t maxSearchCandidates = 0;
  int64_t searchBeamWidth = 0;
  int64_t candidateParallelism = 1;
  int64_t spmBase = 0;
  int64_t spmLimit = 0;
  int64_t spmAlignment = 0;
  int64_t ddrCapacityBytes = 0;
  int64_t ddrLargestContiguousBytes = 0;
  int64_t ddrBandwidthLimitBytes = 0;
  int64_t ddrAlignmentBytes = 0;
  int64_t computeOpsPerCycle = 0;
  int64_t ddrBytesPerCycle = 0;
  int64_t spmBytesPerCycle = 0;
  int64_t instrIssueCycles = 0;
  bool assumeDdrComputeOverlap = false;
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

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticTraversalShape(GroupOp group);

std::optional<int64_t> getElementByteWidth(mlir::Type type);

std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
getYieldedRootLinalgOps(GroupOp group);

std::string getStandaloneGroupModuleText(GroupOp group);

GroupOp findSingleSelectionGroup(mlir::ModuleOp module);

mlir::FailureOr<llvm::SmallVector<int64_t, 2>>
getStaticRootReductionRanges(GroupOp group);

bool failsCheapSPMBound(GroupOp group, const CandidateSpec &candidate,
                        int64_t spmBase, int64_t spmLimit);

std::optional<std::string>
getCheapTargetGeometryFailure(GroupOp group, const CandidateSpec &candidate,
                              llvm::ArrayRef<int64_t> reductionRanges);

TileSizeOptions buildTileSizeOptions(llvm::ArrayRef<int64_t> traversalShape,
                                     llvm::ArrayRef<int64_t> reductionRanges,
                                     llvm::ArrayRef<int64_t> preferred,
                                     int64_t maxCandidatesPerDim);

bool isFullFirstTile(llvm::ArrayRef<int64_t> traversalShape,
                     const TileInstance &tile);

llvm::SmallVector<TileInstance, 8>
buildRepresentativeTiles(llvm::ArrayRef<int64_t> traversalShape,
                         llvm::ArrayRef<int64_t> tileSizes);

CandidateEvaluation
evaluateCompleteCandidate(GroupOp group, llvm::ArrayRef<int64_t> traversalShape,
                          const CandidateSpec &candidate,
                          const SelectionConfig &config);

CandidateCheckResult evaluateCandidateOnOriginalGroup(
    GroupOp group, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config);

CandidateCheckResult
evaluateCandidateOnStandaloneText(llvm::StringRef standaloneGroupModuleText,
                                  llvm::ArrayRef<int64_t> traversalShape,
                                  const CandidateSpec &candidate,
                                  const SelectionConfig &config);

mlir::FailureOr<SelectedCandidate>
selectCandidateForGroup(GroupOp group, llvm::StringRef label,
                        const SelectionConfig &config);

void printSelectedSummary(const SelectedCandidate &selected,
                          TileSearchMode mode);

mlir::LogicalResult commitSelectedCandidate(SelectedCandidate &selected,
                                            const SelectionConfig &config);

mlir::LogicalResult accumulateStaticTerminalOperations(mlir::Operation *root,
                                                       bool skipGroupBodies,
                                                       uint64_t &count);

} // namespace wafer::group_tile_selection
