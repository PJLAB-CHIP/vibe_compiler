//===- ExecutionStructurePlan.cpp - Typed cyclic structure ------------===//

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructurePlan.h"

namespace wafer::compiler::detail {

const PipelineScopeId &
getPipelineScope(const ExecutionStructureChoice &choice) {
  return std::visit(
      [](const auto &structure) -> const PipelineScopeId & {
        return structure.scope;
      },
      choice);
}

} // namespace wafer::compiler::detail
