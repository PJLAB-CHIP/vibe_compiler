//===- StructuredDemandView.h - DAG lookup over exact demand ---*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_STRUCTUREDDEMANDVIEW_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_STRUCTUREDDEMANDVIEW_H

#include "Wafer/Planning/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Analysis/Linalg/StructuredDAGAnalysis.h"
#include "Wafer/Analysis/Linalg/SemanticRootAnalysis.h"

#include "mlir/Support/LogicalResult.h"

#include <string>

namespace wafer::compiler::detail {

/// Non-owning query view that maps current DAG node IDs to the stable semantic
/// identities used by SpatialAssignment and ExactDemandProof. It derives no
/// new demand, ownership, placement, or physical action.
class StructuredDemandView {
public:
  static mlir::FailureOr<StructuredDemandView>
  create(const StructuredDAGAnalysis &dag, const SpatialAssignment &assignment,
         const analysis::ExactDemandProof &proof,
         std::string *failureReason = nullptr);

  const SemanticRootKey *getRoot(StructuredDAGNodeID node) const;
  const SemanticRootBinding *getRootBinding(const SemanticRootKey &root) const;
  const SemanticValueBinding *getValueBinding(mlir::Value value) const;
  const NodeExecutionPartition *getNode(StructuredDAGNodeID node) const;
  const ExecutionShard *getShard(StructuredDAGNodeID node, TileId tile) const;
  const analysis::DependencyDemand *getDependency(StructuredDAGNodeID consumer,
                                                  uint32_t operand) const;
  llvm::SmallVector<const analysis::FinalResultOwner *, 4>
  getFinalOwners(StructuredDAGNodeID node, uint32_t result) const;
  bool hasSpatialReduction(StructuredDAGNodeID node) const;

private:
  StructuredDemandView(const StructuredDAGAnalysis &dag,
                       const SpatialAssignment &assignment,
                       const analysis::ExactDemandProof &proof,
                       SemanticRootAnalysis roots)
      : dag(dag), assignment(assignment), proof(proof),
        roots(std::move(roots)) {}

  const StructuredDAGAnalysis &dag;
  const SpatialAssignment &assignment;
  const analysis::ExactDemandProof &proof;
  SemanticRootAnalysis roots;
};

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_STRUCTUREDDEMANDVIEW_H
