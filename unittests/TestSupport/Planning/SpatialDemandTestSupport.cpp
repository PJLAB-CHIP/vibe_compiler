//===- SpatialDemandTestSupport.cpp - Closed test spatial demand -------===//

#include "TestSupport/Planning/SpatialDemandTestSupport.h"

#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::test {

mlir::FailureOr<TestSpatialDemand> buildTestSpatialDemand(
    const compiler::detail::StructuredDAGAnalysis &dag,
    llvm::ArrayRef<compiler::detail::StructuredDAGNodePlacement> placements,
    std::string *failureReason) {
  llvm::SmallVector<TileId, 16> tiles;
  for (const compiler::detail::StructuredDAGNodePlacement &placement :
       placements)
    for (TileId tile : placement.tiles)
      if (!llvm::is_contained(tiles, tile))
        tiles.push_back(tile);
  llvm::sort(tiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  compiler::detail::SpatialDomainProblemResult problem =
      compiler::detail::buildSpatialDomainProblem(dag, tiles);
  if (!problem.succeeded()) {
    if (failureReason && problem.failure)
      *failureReason = problem.failure->detail;
    return mlir::failure();
  }
  compiler::detail::SpatialPlan plan;
  for (const compiler::detail::SpatialRootDomainFacts &root :
       problem.problem->getRoots()) {
    auto selected = llvm::find_if(
        placements, [&](const compiler::detail::StructuredDAGNodePlacement &p) {
          return p.node == root.node;
        });
    if (selected == placements.end() ||
        selected->iteratorPartitionFactors.size() !=
            root.iteratorExtents.size())
      return mlir::failure();
    compiler::detail::NodeSpatialPlan node;
    node.root = root.root;
    for (auto [iterator, factor] :
         llvm::enumerate(selected->iteratorPartitionFactors)) {
      if (factor == 0)
        return mlir::failure();
      node.axes.push_back(
          {static_cast<uint32_t>(iterator),
           compiler::detail::IteratorPartitionScheme::BalancedParts,
           static_cast<int64_t>(factor)});
    }
    node.embedding = selected->tiles;
    auto groups = compiler::detail::deriveSpatialReductionGroups(
        root, node.axes, failureReason);
    if (mlir::failed(groups))
      return mlir::failure();
    for (const compiler::detail::ReductionGroupId &group : *groups) {
      if (group.resultGroup >= root.resultParallelIteratorsByGroup.size())
        return mlir::failure();
      const llvm::SmallBitVector &resultParallel =
          root.resultParallelIteratorsByGroup[group.resultGroup];
      std::optional<TileId> contributor;
      for (size_t cell = 0; cell < node.embedding.size(); ++cell) {
        size_t remainder = cell;
        llvm::SmallVector<uint32_t, 4> coordinate(
            selected->iteratorPartitionFactors.size());
        for (size_t reverse = 0; reverse < coordinate.size(); ++reverse) {
          size_t iterator = coordinate.size() - reverse - 1;
          coordinate[iterator] = static_cast<uint32_t>(
              remainder % selected->iteratorPartitionFactors[iterator]);
          remainder /= selected->iteratorPartitionFactors[iterator];
        }
        llvm::SmallVector<uint32_t, 4> parallel;
        for (int iterator = resultParallel.find_first(); iterator >= 0;
             iterator = resultParallel.find_next(iterator))
          parallel.push_back(coordinate[iterator]);
        if (parallel == group.parallelCoordinate) {
          contributor = node.embedding[cell];
          break;
        }
      }
      if (!contributor)
        return mlir::failure();
      node.reductionMerges.push_back({group, *contributor});
    }
    plan.nodes.push_back(std::move(node));
  }
  mlir::FailureOr<compiler::detail::SpatialAssignment> spatial =
      compiler::detail::closeSpatialPlanStructure(
          problem.problem->getStructuralProblem(), plan, failureReason);
  if (mlir::failed(spatial))
    return mlir::failure();
  auto session = compiler::detail::DemandPlanningSession::create(
      dag, analysis::IndexRelationLimits(), failureReason);
  if (mlir::failed(session))
    return mlir::failure();
  analysis::ExactDemandOutcome outcome = session->query(*spatial);
  session->close();
  const analysis::ExactDemandProof *proof =
      analysis::getExactDemandProof(outcome);
  if (!proof) {
    if (failureReason)
      *failureReason = std::visit(
          [](const auto &value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
              return {};
            else
              return value.detail;
          },
          outcome);
    return mlir::failure();
  }
  return TestSpatialDemand{std::move(*spatial), *proof};
}

} // namespace wafer::test
