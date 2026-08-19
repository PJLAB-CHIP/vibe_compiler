//===- TemporalTiling.cpp - Complete iterator wave domain -------------===//

#include "Wafer/Planning/Search/TemporalTiling.h"

#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace wafer::compiler::detail {
namespace {

llvm::SmallVector<uint32_t, 4>
getActiveIterators(llvm::ArrayRef<int64_t> extents,
                   llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<uint32_t, 4> active;
  for (auto [dimension, extent, size] : llvm::enumerate(extents, sizes))
    if (size < extent)
      active.push_back(static_cast<uint32_t>(dimension));
  return active;
}

} // namespace

mlir::FailureOr<TemporalNodeDomain>
TemporalNodeDomain::create(const StructuredDAGNode &node,
                           const StructuredDAGNodePlacement &placement,
                           std::string *failureReason) {
  if (node.id != placement.node)
    return mlir::failure();
  auto extents = deriveLocalIteratorExtents(
      node.operation, placement.iteratorPartitionFactors, failureReason);
  if (mlir::failed(extents))
    return mlir::failure();
  return TemporalNodeDomain(node.id, node.operation, std::move(*extents));
}

TemporalNodeAssignment TemporalNodeDomain::getFirstAssignment() const {
  return TemporalNodeAssignment{node, localExtents, {}};
}

mlir::FailureOr<TemporalNodeAssignment>
TemporalNodeDomain::getCapacityGuidedAssignment(
    const TargetMemoryPolicy &memory, std::string *failureReason) const {
  auto sizes = deriveStructuredOperationTemporalTileShape(operation,
                                                          localExtents, memory);
  if (mlir::failed(sizes)) {
    if (failureReason)
      *failureReason = (llvm::Twine("node ") + llvm::Twine(node) +
                        " has no capacity-guided temporal shape")
                           .str();
    return mlir::failure();
  }
  TemporalNodeAssignment result{node, *sizes,
                                getActiveIterators(localExtents, *sizes)};
  if (!contains(result)) {
    if (failureReason) {
      llvm::raw_string_ostream stream(*failureReason);
      stream << "node " << node
             << " produced an out-of-domain capacity-guided temporal shape; "
                "local_extents=[";
      llvm::interleaveComma(localExtents, stream);
      stream << "], tile_sizes=[";
      llvm::interleaveComma(*sizes, stream);
      stream << "], wave_order=[";
      llvm::interleaveComma(result.waveLoopOrder, stream);
      stream << ']';
    }
    return mlir::failure();
  }
  return result;
}

bool TemporalNodeDomain::contains(
    const TemporalNodeAssignment &assignment) const {
  if (assignment.node != node ||
      assignment.iteratorTileSizes.size() != localExtents.size())
    return false;
  for (auto [size, extent] :
       llvm::zip_equal(assignment.iteratorTileSizes, localExtents))
    if (size <= 0 || size > extent)
      return false;
  llvm::SmallVector<uint32_t, 4> active =
      getActiveIterators(localExtents, assignment.iteratorTileSizes);
  llvm::SmallVector<uint32_t, 4> order = assignment.waveLoopOrder;
  llvm::sort(order);
  return order == active;
}

mlir::FailureOr<std::optional<TemporalNodeAssignment>>
TemporalNodeDomain::getNextAssignment(
    const TemporalNodeAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  TemporalNodeAssignment next = assignment;
  if (std::next_permutation(next.waveLoopOrder.begin(),
                            next.waveLoopOrder.end()))
    return std::optional<TemporalNodeAssignment>(std::move(next));
  for (size_t reverse = 0; reverse < localExtents.size(); ++reverse) {
    const size_t dimension = localExtents.size() - reverse - 1;
    if (next.iteratorTileSizes[dimension] <= 1)
      continue;
    --next.iteratorTileSizes[dimension];
    for (size_t reset = dimension + 1; reset < localExtents.size(); ++reset)
      next.iteratorTileSizes[reset] = localExtents[reset];
    next.waveLoopOrder =
        getActiveIterators(localExtents, next.iteratorTileSizes);
    return std::optional<TemporalNodeAssignment>(std::move(next));
  }
  return std::optional<TemporalNodeAssignment>{};
}

mlir::FailureOr<CardTemporalDomain> CardTemporalDomain::create(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> placements,
    std::string *failureReason) {
  if (placements.size() != dag.getNodes().size())
    return mlir::failure();
  llvm::SmallVector<TemporalNodeDomain, 16> domains;
  for (const StructuredDAGNode &node : dag.getNodes()) {
    auto placement = llvm::find_if(placements, [&](const auto &candidate) {
      return candidate.node == node.id;
    });
    if (placement == placements.end())
      return mlir::failure();
    auto domain = TemporalNodeDomain::create(node, *placement, failureReason);
    if (mlir::failed(domain))
      return mlir::failure();
    domains.push_back(std::move(*domain));
  }
  return CardTemporalDomain(std::move(domains));
}

CardTemporalAssignment CardTemporalDomain::getFirstAssignment() const {
  CardTemporalAssignment result;
  for (const TemporalNodeDomain &domain : domains)
    result.nodes.push_back(domain.getFirstAssignment());
  return result;
}

mlir::FailureOr<CardTemporalAssignment>
CardTemporalDomain::getCapacityGuidedAssignment(
    const TargetMemoryPolicy &memory, std::string *failureReason) const {
  CardTemporalAssignment result;
  for (const TemporalNodeDomain &domain : domains) {
    auto node = domain.getCapacityGuidedAssignment(memory, failureReason);
    if (mlir::failed(node))
      return mlir::failure();
    result.nodes.push_back(std::move(*node));
  }
  return result;
}

bool CardTemporalDomain::contains(
    const CardTemporalAssignment &assignment) const {
  if (assignment.nodes.size() != domains.size())
    return false;
  for (auto [domain, node] : llvm::zip_equal(domains, assignment.nodes))
    if (!domain.contains(node))
      return false;
  return true;
}

mlir::FailureOr<std::optional<CardTemporalAssignment>>
CardTemporalDomain::getNextAssignment(
    const CardTemporalAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  CardTemporalAssignment next = assignment;
  for (size_t reverse = 0; reverse < domains.size(); ++reverse) {
    const size_t index = domains.size() - reverse - 1;
    auto advanced = domains[index].getNextAssignment(next.nodes[index]);
    if (mlir::failed(advanced))
      return mlir::failure();
    if (!*advanced)
      continue;
    next.nodes[index] = std::move(**advanced);
    for (size_t reset = index + 1; reset < domains.size(); ++reset)
      next.nodes[reset] = domains[reset].getFirstAssignment();
    return std::optional<CardTemporalAssignment>(std::move(next));
  }
  return std::optional<CardTemporalAssignment>{};
}

} // namespace wafer::compiler::detail
