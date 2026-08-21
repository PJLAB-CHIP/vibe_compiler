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
  mlir::FailureOr<StructuredDemandView> view = StructuredDemandView::create(
      program.dag, spatial, demand, failureReason);
  if (!tensorProgram || mlir::failed(view))
    return fail("single-root region requires closed spatial demand");

  llvm::SmallVector<StructuredNodeIterationShard, 32> shards;
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    const NodeExecutionPartition *partition = view->getNode(node.id);
    const SemanticRootKey *root = view->getRoot(node.id);
    if (!partition || !root || partition->shards.empty())
      return fail("single-root region omitted one node execution domain");
    for (const ExecutionShard &execution : partition->shards) {
      StructuredNodeIterationShard shard;
      shard.structuredNodeId = node.id;
      shard.tile = execution.tile;
      for (const IteratorInterval &interval : execution.iterationDomain) {
        shard.offsets.push_back(interval.offset);
        shard.sizes.push_back(interval.size);
      }
      for (const analysis::ReductionMergeRequirement &merge :
           demand.reductionMerges) {
        if (merge.group.root != *root ||
            !llvm::any_of(merge.contributions,
                          [&](const analysis::ReductionContribution &entry) {
                            return entry.shard == execution.shard;
                          }))
          continue;
        shard.reductionGroups.push_back({merge.group, merge.mergeTile});
      }
      shards.push_back(std::move(shard));
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
