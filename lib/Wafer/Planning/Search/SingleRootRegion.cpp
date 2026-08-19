//===- SingleRootRegion.cpp - Selected placement region apply ----------===//

#include "Wafer/Planning/Search/SingleRootRegion.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {

mlir::FailureOr<CardSingleRootRegionMaterialization>
materializeCardSingleRootRegions(mlir::ModuleOp tensorProgram,
                                 const CardProgramAnalysis &program,
                                 CardId cardId,
                                 const analysis::LogicalShardTrial &trial,
                                 std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<CardSingleRootRegionMaterialization> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  if (!tensorProgram || trial.epoch != program.epoch)
    return fail("single-root region trial belongs to another IR epoch");
  if (trial.nodes.size() != program.dag.getNodes().size())
    return fail("single-root region trial does not cover every DAG node");

  llvm::SmallVector<StructuredNodeIterationShard, 32> shards;
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    auto nodeTrial = llvm::find_if(
        trial.nodes, [&](const analysis::LogicalNodeTrial &candidate) {
          return candidate.node == node.id;
        });
    if (nodeTrial == trial.nodes.end() || nodeTrial->executionShards.empty())
      return fail("single-root region trial omitted one node execution domain");
    const bool partialReduction = llvm::any_of(
        nodeTrial->bindings, [](const analysis::LogicalTileBinding &binding) {
          return binding.role ==
                 analysis::TileRole::PartialReductionContribution;
        });
    if (partialReduction != nodeTrial->reductionMergeTile.has_value())
      return fail("single-root reduction merge ownership is inconsistent");
    for (const analysis::LogicalExecutionShard &execution :
         nodeTrial->executionShards) {
      if (!execution.executionDomain)
        return fail("single-root region shard has no exact execution domain");
      analysis::StaticRectangularIndexSetResult rectangle =
          analysis::IndexSetResult{analysis::IndexRelationStatus::Exact,
                                   *execution.executionDomain,
                                   {}}
              .getExactStaticRectangularDomain();
      if (!rectangle.isExact() || !rectangle.domain)
        return fail("single-root execution domain is not one exact rectangle");
      shards.push_back(StructuredNodeIterationShard{
          node.id, execution.tile, std::move(rectangle.domain->offsets),
          std::move(rectangle.domain->sizes),
          partialReduction
              ? StructuredNodeIterationShardRole::PartialReductionContribution
              : StructuredNodeIterationShardRole::Complete,
          nodeTrial->reductionMergeTile});
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
