//===- StructuredDemandView.cpp - DAG lookup over exact demand ---------===//

#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace wafer::compiler::detail {

mlir::FailureOr<StructuredDemandView> StructuredDemandView::create(
    const StructuredDAGAnalysis &dag, const SpatialAssignment &assignment,
    const analysis::ExactDemandProof &proof, std::string *failureReason) {
  mlir::FailureOr<SemanticRootAnalysis> roots =
      SemanticRootAnalysis::create(dag, failureReason);
  if (mlir::failed(roots))
    return mlir::failure();
  if (assignment.nodes.size() != roots->getRoots().size()) {
    if (failureReason)
      *failureReason = "spatial assignment does not cover semantic roots";
    return mlir::failure();
  }
  for (auto [binding, node] :
       llvm::zip_equal(roots->getRoots(), assignment.nodes)) {
    if (binding.key != node.root) {
      if (failureReason)
        *failureReason = "spatial assignment roots do not use semantic order";
      return mlir::failure();
    }
  }
  auto findBinding =
      [&](const SemanticRootKey &key) -> const SemanticRootBinding * {
    auto binding = llvm::lower_bound(
        roots->getRoots(), key,
        [](const SemanticRootBinding &candidate, const SemanticRootKey &root) {
          return candidate.key < root;
        });
    return binding == roots->getRoots().end() || binding->key != key
               ? nullptr
               : &*binding;
  };
  for (const analysis::FinalResultOwner &owner : proof.finalOwners) {
    const SemanticRootBinding *binding = findBinding(owner.root);
    if (!binding || owner.result >= binding->operation->getNumResults()) {
      if (failureReason)
        *failureReason = "exact-demand final owner names an unknown result";
      return mlir::failure();
    }
  }
  for (const analysis::DependencyDemand &dependency : proof.dependencyDemands) {
    const SemanticRootBinding *binding = findBinding(dependency.consumer);
    if (!binding || binding->operation != dependency.consumerOperation ||
        dependency.consumerOperand >=
            dependency.consumerOperation->getNumOperands()) {
      if (failureReason)
        *failureReason =
            "exact-demand dependency names an unknown consumer operand";
      return mlir::failure();
    }
  }
  for (const analysis::ReductionMergeRequirement &merge :
       proof.reductionMerges) {
    if (!findBinding(merge.group.root)) {
      if (failureReason)
        *failureReason = "exact-demand merge names an unknown root";
      return mlir::failure();
    }
  }
  return StructuredDemandView(dag, assignment, proof, std::move(*roots));
}

const SemanticRootKey *
StructuredDemandView::getRoot(StructuredDAGNodeID node) const {
  const StructuredDAGNode *entry = dag.getNode(node);
  const SemanticRootBinding *binding =
      entry ? roots.find(entry->operation) : nullptr;
  return binding ? &binding->key : nullptr;
}

const NodeExecutionPartition *
StructuredDemandView::getNode(StructuredDAGNodeID node) const {
  const SemanticRootKey *root = getRoot(node);
  if (!root)
    return nullptr;
  auto entry = llvm::lower_bound(
      assignment.nodes, *root,
      [](const NodeExecutionPartition &candidate, const SemanticRootKey &key) {
        return candidate.root < key;
      });
  return entry == assignment.nodes.end() || entry->root != *root ? nullptr
                                                                 : &*entry;
}

const ExecutionShard *StructuredDemandView::getShard(StructuredDAGNodeID node,
                                                     TileId tile) const {
  const NodeExecutionPartition *partition = getNode(node);
  if (!partition)
    return nullptr;
  auto shard = llvm::find_if(partition->shards, [tile](const auto &candidate) {
    return candidate.tile == tile;
  });
  return shard == partition->shards.end() ? nullptr : &*shard;
}

const analysis::DependencyDemand *
StructuredDemandView::getDependency(StructuredDAGNodeID consumer,
                                    uint32_t operand) const {
  const SemanticRootKey *root = getRoot(consumer);
  if (!root)
    return nullptr;
  auto demand = llvm::find_if(
      proof.dependencyDemands, [&](const analysis::DependencyDemand &entry) {
        return entry.consumer == *root && entry.consumerOperand == operand;
      });
  return demand == proof.dependencyDemands.end() ? nullptr : &*demand;
}

llvm::SmallVector<const analysis::FinalResultOwner *, 4>
StructuredDemandView::getFinalOwners(StructuredDAGNodeID node,
                                     uint32_t result) const {
  llvm::SmallVector<const analysis::FinalResultOwner *, 4> owners;
  const SemanticRootKey *root = getRoot(node);
  if (!root)
    return owners;
  for (const analysis::FinalResultOwner &owner : proof.finalOwners)
    if (owner.root == *root && owner.result == result)
      owners.push_back(&owner);
  return owners;
}

bool StructuredDemandView::hasSpatialReduction(StructuredDAGNodeID node) const {
  const SemanticRootKey *root = getRoot(node);
  return root &&
         llvm::any_of(proof.reductionMerges,
                      [&](const analysis::ReductionMergeRequirement &merge) {
                        return merge.group.root == *root;
                      });
}

} // namespace wafer::compiler::detail
