//===- RootRegionWorkAnalysis.h - Single-root work query -----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_ROOTREGIONWORKANALYSIS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_ROOTREGIONWORKANALYSIS_H

#include "Wafer/Analysis/PhysicalDataflow/RootRegionWork.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <utility>

namespace wafer::compiler::detail {

/// Read-only query over one closed assignment and its exact-demand proof. The
/// query borrows current IR and produces no IR, physical choice, or cache.
class RootRegionWorkAnalysis {
public:
  static mlir::FailureOr<RootRegionWorkAnalysis>
  create(const StructuredDAGAnalysis &dag, const SpatialAssignment &assignment,
         const analysis::ExactDemandProof &proof,
         std::string *failureReason = nullptr);

  analysis::RootRegionWorkOutcome
  query(const SemanticRootKey &root, TileId tile,
        const analysis::IndexRelationLimits &limits =
            analysis::IndexRelationLimits()) const;

  const SemanticRootKey *getRoot(StructuredDAGNodeID node) const {
    return view.getRoot(node);
  }

private:
  RootRegionWorkAnalysis(const StructuredDAGAnalysis &dag,
                         const SpatialAssignment &assignment,
                         const analysis::ExactDemandProof &proof,
                         StructuredDemandView view)
      : dag(dag), assignment(assignment), proof(proof), view(std::move(view)) {}

  const StructuredDAGAnalysis &dag;
  const SpatialAssignment &assignment;
  const analysis::ExactDemandProof &proof;
  StructuredDemandView view;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_ROOTREGIONWORKANALYSIS_H
