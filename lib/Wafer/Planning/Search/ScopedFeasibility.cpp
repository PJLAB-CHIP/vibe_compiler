//===- ScopedFeasibility.cpp - Pure partial-state bounds --------------===//

#include "Wafer/Planning/Search/ScopedFeasibility.h"

#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace wafer::compiler::detail {
namespace {

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

std::optional<uint64_t> getLogicalPayloadBytes(mlir::ShapedType type,
                                               llvm::ArrayRef<int64_t> shape) {
  unsigned elementBits = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type.getElementType()))
    elementBits = integer.getWidth();
  else if (auto floating =
               mlir::dyn_cast<mlir::FloatType>(type.getElementType()))
    elementBits = floating.getWidth();
  else
    return std::nullopt;
  uint64_t elements = 1;
  for (int64_t extent : shape) {
    if (extent <= 0)
      return std::nullopt;
    elements = saturatingMultiply(elements, static_cast<uint64_t>(extent));
  }
  const uint64_t bits = saturatingMultiply(elements, elementBits);
  return bits / 8 + (bits % 8 != 0);
}

const analysis::LogicalNodeTrial *
findNodeTrial(const analysis::LogicalShardTrial &trial,
              StructuredDAGNodeID node) {
  auto found = llvm::find_if(trial.nodes, [&](const auto &candidate) {
    return candidate.node == node;
  });
  return found == trial.nodes.end() ? nullptr : &*found;
}

const TemporalNodeAssignment *
findTemporal(const CardTemporalAssignment &assignment,
             StructuredDAGNodeID node) {
  auto found = llvm::find_if(assignment.nodes, [&](const auto &candidate) {
    return candidate.node == node;
  });
  return found == assignment.nodes.end() ? nullptr : &*found;
}

std::optional<analysis::StaticRectangularIndexSet>
findShard(const analysis::LogicalNodeTrial &trial, TileId tile) {
  auto found = llvm::find_if(trial.executionShards, [&](const auto &shard) {
    return shard.tile == tile;
  });
  if (found == trial.executionShards.end() || !found->executionDomain)
    return std::nullopt;
  analysis::StaticRectangularIndexSetResult rectangle =
      analysis::IndexSetResult{
          analysis::IndexRelationStatus::Exact, *found->executionDomain, {}}
          .getExactStaticRectangularDomain();
  return rectangle.isExact() ? rectangle.domain : std::nullopt;
}

struct NodeLowerBound {
  uint64_t bytes = 0;
  bool complete = true;
};

NodeLowerBound getNodeLowerBound(mlir::Operation *operation,
                                 llvm::ArrayRef<int64_t> iteratorTileSizes) {
  NodeLowerBound result;
  for (auto [operandNumber, value] :
       llvm::enumerate(operation->getOperands())) {
    auto type = mlir::dyn_cast<mlir::ShapedType>(value.getType());
    if (!type || !type.hasRank())
      continue;
    auto shape = getStructuredOperandTileShape(
        operation, static_cast<unsigned>(operandNumber), iteratorTileSizes);
    if (!shape) {
      result.complete = false;
      continue;
    }
    std::optional<uint64_t> bytes = getLogicalPayloadBytes(type, *shape);
    if (!bytes) {
      result.complete = false;
      continue;
    }
    result.bytes = std::max(result.bytes, *bytes);
  }
  for (unsigned resultNumber = 0; resultNumber < operation->getNumResults();
       ++resultNumber) {
    auto type = mlir::dyn_cast<mlir::ShapedType>(
        operation->getResult(resultNumber).getType());
    if (!type || !type.hasRank())
      continue;
    auto shape = getStructuredResultTileShape(operation, resultNumber,
                                              iteratorTileSizes);
    if (!shape) {
      result.complete = false;
      continue;
    }
    std::optional<uint64_t> bytes = getLogicalPayloadBytes(type, *shape);
    if (!bytes) {
      result.complete = false;
      continue;
    }
    result.bytes = std::max(result.bytes, *bytes);
  }
  return result;
}

} // namespace

mlir::FailureOr<ScopedFeasibilityResult> analyzeScopedFeasibility(
    const CardProgramAnalysis &program,
    const analysis::LogicalShardTrial &trial,
    const CoupledRegionDomain &coupledDomain,
    const CoupledRegionAssignment &coupledAssignment,
    const CardTemporalDomain &temporalDomain,
    const CardTemporalAssignment &temporalAssignment,
    const TargetMemoryPolicy &memory,
    llvm::ArrayRef<FeasibilityCoordinate> unresolvedCoordinates) {
  if (trial.epoch != program.epoch ||
      !coupledDomain.contains(coupledAssignment) ||
      !temporalDomain.contains(temporalAssignment) || memory.spmBase < 0 ||
      memory.spmLimit <= memory.spmBase)
    return mlir::failure();
  llvm::DenseSet<uint8_t> seenCoordinates;
  for (FeasibilityCoordinate coordinate : unresolvedCoordinates)
    if (!seenCoordinates.insert(static_cast<uint8_t>(coordinate)).second)
      return mlir::failure();

  ScopedFeasibilityResult result;
  result.kind = unresolvedCoordinates.empty()
                    ? ScopedFeasibilityKind::LowerBound
                    : ScopedFeasibilityKind::Deferred;
  result.reason = unresolvedCoordinates.empty()
                      ? ScopedFeasibilityReason::None
                      : ScopedFeasibilityReason::MissingCoordinates;
  result.requiredCoordinates.assign(unresolvedCoordinates.begin(),
                                    unresolvedCoordinates.end());
  llvm::sort(result.requiredCoordinates);
  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);

  for (const CoupledRegionGroup &group : coupledAssignment.groups) {
    for (StructuredDAGNodeID node : group.nodes) {
      const StructuredDAGNode *dagNode = program.dag.getNode(node);
      const analysis::LogicalNodeTrial *nodeTrial = findNodeTrial(trial, node);
      const TemporalNodeAssignment *temporal =
          findTemporal(temporalAssignment, node);
      std::optional<analysis::StaticRectangularIndexSet> shard =
          nodeTrial ? findShard(*nodeTrial, group.tile) : std::nullopt;
      if (!dagNode || !dagNode->operation || !temporal || !shard ||
          shard->sizes.size() != temporal->iteratorTileSizes.size())
        return mlir::failure();

      llvm::SmallVector<int64_t, 4> leafSizes;
      for (auto [extent, tile] :
           llvm::zip_equal(shard->sizes, temporal->iteratorTileSizes)) {
        if (extent <= 0 || tile <= 0)
          return mlir::failure();
        leafSizes.push_back(std::min(extent, tile));
      }
      NodeLowerBound minimum = getNodeLowerBound(dagNode->operation, leafSizes);
      if (!minimum.complete) {
        result.kind = unresolvedCoordinates.empty()
                          ? ScopedFeasibilityKind::Indeterminate
                          : ScopedFeasibilityKind::Deferred;
        result.reason = unresolvedCoordinates.empty()
                            ? ScopedFeasibilityReason::UnsupportedFootprint
                            : ScopedFeasibilityReason::MissingCoordinates;
      }
      std::optional<uint64_t> estimate =
          estimateStructuredOperationTileResidencyBytes(dagNode->operation,
                                                        leafSizes, memory);
      result.lowerBounds.push_back(
          ScopedSPMLowerBound{node, group.tile, minimum.bytes, estimate});
      if (minimum.bytes <= capacity)
        continue;

      result.kind = ScopedFeasibilityKind::ExactRejection;
      result.reason = ScopedFeasibilityReason::MinimumFootprintExceedsSPM;
      result.requiredCoordinates.clear();
      result.rejection = ScopedSPMRejection{
          ScopedFeasibilityAssignmentKey{
              node, group.tile, group.nodes, shard->offsets, shard->sizes,
              temporal->iteratorTileSizes, temporal->waveLoopOrder},
          minimum.bytes, capacity};
      return result;
    }
  }
  llvm::sort(result.lowerBounds, [](const auto &lhs, const auto &rhs) {
    return std::tuple(lhs.tile.getValue(), lhs.node) <
           std::tuple(rhs.tile.getValue(), rhs.node);
  });
  return result;
}

} // namespace wafer::compiler::detail
