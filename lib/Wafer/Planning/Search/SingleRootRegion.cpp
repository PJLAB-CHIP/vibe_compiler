//===- SingleRootRegion.cpp - Selected placement region apply ----------===//

#include "Wafer/Planning/Search/SingleRootRegion.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {

mlir::FailureOr<CardSingleRootRegionMaterialization>
materializeCardSingleRootRegions(mlir::ModuleOp tensorProgram,
                                 const CardProgramAnalysis &program,
                                 CardId cardId,
                                 const SpatialAssignment &spatial,
                                 const analysis::ExactDemandProof &demand,
                                 std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<CardSingleRootRegionMaterialization> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  auto rootWork = RootRegionWorkAnalysis::create(program.dag, spatial, demand,
                                                 failureReason);
  if (!tensorProgram || mlir::failed(rootWork))
    return fail("single-root region requires closed spatial demand");

  llvm::SmallVector<StructuredNodeIterationShard, 32> shards;
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    const SemanticRootKey *root = rootWork->getRoot(node.id);
    if (!root)
      return fail("single-root region omitted one node execution domain");
    for (TileId tile : program.availableTileIds) {
      analysis::RootRegionWorkOutcome outcome = rootWork->query(*root, tile);
      if (std::holds_alternative<analysis::NoRootRegionWork>(outcome))
        continue;
      const analysis::RootRegionWork *work =
          analysis::getRootRegionWork(outcome);
      if (!work)
        return fail(std::visit(
            [](const auto &value) -> std::string {
              using T = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<T, analysis::RootRegionWork> ||
                            std::is_same_v<T, analysis::NoRootRegionWork>)
                return "single-root work has no materializable leaf";
              else
                return value.detail;
            },
            outcome));
      auto leaf = prepareStructuredRootLeaf(node.id, *work, failureReason);
      if (mlir::failed(leaf))
        return mlir::failure();
      if (*leaf)
        shards.push_back(std::move(**leaf));
    }
  }

  CardSingleRootRegionMaterialization result;
  if (mlir::failed(lowerStructuredNodeShardsToCardModule(
          tensorProgram, cardId, program.availableTileIds,
          program.operationNodes, shards, result.module, &result.relations,
          failureReason)))
    return mlir::failure();
  return result;
}

} // namespace wafer::compiler::detail
