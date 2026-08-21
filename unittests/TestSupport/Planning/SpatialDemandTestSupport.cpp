//===- SpatialDemandTestSupport.cpp - Closed test spatial demand -------===//

#include "TestSupport/Planning/SpatialDemandTestSupport.h"

#include "Wafer/Planning/Search/SpatialPlacement.h"

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
  llvm::sort(tiles,
             [](TileId lhs, TileId rhs) {
               return lhs.getValue() < rhs.getValue();
             });
  auto domain = compiler::detail::CardSpatialPlacementDomain::create(dag, tiles);
  if (mlir::failed(domain))
    return mlir::failure();
  compiler::detail::CardSpatialPlacementAssignment assignment;
  for (const compiler::detail::StructuredDAGNodePlacement &placement :
       placements) {
    compiler::detail::SpatialPlacementAssignment selected;
    selected.node = placement.node;
    selected.iteratorFactors = placement.iteratorPartitionFactors;
    selected.tiles = placement.tiles;
    assignment.nodes.push_back(std::move(selected));
  }
  mlir::FailureOr<compiler::detail::SpatialAssignment> spatial =
      domain->close(dag, assignment, failureReason);
  if (mlir::failed(spatial))
    return mlir::failure();
  auto session = compiler::detail::DemandPlanningSession::create(
      dag, analysis::IndexRelationLimits(), failureReason);
  if (mlir::failed(session))
    return mlir::failure();
  analysis::ExactDemandOutcome outcome = session->query(*spatial);
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
  session->close();
  return TestSpatialDemand{std::move(*spatial), *proof};
}

} // namespace wafer::test
