//===- SelectedRegionMaterialization.h - RegionPlan apply ----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SELECTEDREGIONMATERIALIZATION_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SELECTEDREGIONMATERIALIZATION_H

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"
#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/RepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <vector>

namespace wafer::compiler::detail {

struct SelectedRegionExecutionNode {
  RegionExecutionId execution;
  uint32_t structuredNodeId = 0;
  SemanticRootKey root;
};

struct SelectedRegionMaterializationSource {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> operationNodes;
  std::vector<SelectedRegionExecutionNode> executionNodes;
};

/// Clones one candidate-owned TensorProgram and explicitly materializes every
/// ReplicaExecutionId as its own pure structured producer clone. Required
/// executions retain their source node identity. No source IR is modified.
mlir::FailureOr<SelectedRegionMaterializationSource>
prepareSelectedRegionMaterializationSource(
    mlir::ModuleOp source, const CardProgramAnalysis &program,
    const RegionPlan &regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    std::string *failureReason = nullptr);

/// Converts one already-selected RegionPlan into the narrow construction
/// descriptors consumed by the common TileRegion materializer. This query
/// does not mutate IR, choose another region/binding, or fill later physical
/// axes. A returned descriptor remains valid only for the supplied current-IR
/// epoch and caller-owned candidate transaction.
mlir::FailureOr<std::vector<StructuredNodeShardGroup>>
prepareSelectedRegionGroups(
    const CardProgramAnalysis &program, const RegionPlan &regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const TemporalPlan &temporal,
    llvm::ArrayRef<SelectedRegionExecutionNode> executionNodes = {},
    std::string *failureReason = nullptr);

/// Adds the selected per-node operand/result layouts consumed by the common
/// TileRegion emitter. Every shaped payload/result must resolve through the
/// typed RepresentationPlan; this function does not create IR or choose a
/// fallback layout.
mlir::LogicalResult applySelectedRegionRepresentations(
    const RegionPlan &regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const RepresentationPlan &representations,
    llvm::ArrayRef<SelectedRegionExecutionNode> executionNodes,
    std::vector<StructuredNodeShardGroup> &groups,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SELECTEDREGIONMATERIALIZATION_H
