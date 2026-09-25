//===- CurrentTensorPipeline.h - Unit-test pipeline composition -*- C++ -*-===//
#ifndef WAFER_TESTSUPPORT_TRANSFORMS_CURRENTTENSORPIPELINE_H
#define WAFER_TESTSUPPORT_TRANSFORMS_CURRENTTENSORPIPELINE_H
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/TensorPreparation.h"

namespace wafer::test {
inline compiler::detail::LayoutOptimizationResult prepareTensorsAndLayout(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    const compiler::detail::ExternalBufferLayout &external = {}) {
  auto prepared =
      compiler::detail::prepareCurrentTensorInput(module, relations);
  if (!prepared.succeeded()) {
    compiler::detail::LayoutOptimizationResult result;
    result.status =
        prepared.status ==
                compiler::detail::TensorPreparationStatus::Unsupported
            ? compiler::detail::ExactPBQPStatus::NoSolution
            : compiler::detail::ExactPBQPStatus::BrokenContract;
    result.detail = std::move(prepared.detail);
    return result;
  }
  return compiler::detail::prepareCurrentLayoutInput(module, relations,
                                                     external);
}
inline compiler::detail::LayoutOptimizationResult prepareTensorsAndBufferize(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    uint64_t workLimit = UINT64_C(1048576),
    const compiler::detail::ExternalBufferLayout &external = {}) {
  auto prepared =
      compiler::detail::prepareCurrentTensorInput(module, relations);
  if (!prepared.succeeded()) {
    compiler::detail::LayoutOptimizationResult result;
    result.status =
        prepared.status ==
                compiler::detail::TensorPreparationStatus::Unsupported
            ? compiler::detail::ExactPBQPStatus::NoSolution
            : compiler::detail::ExactPBQPStatus::BrokenContract;
    result.detail = std::move(prepared.detail);
    return result;
  }
  return compiler::detail::resolveCurrentLayoutsAndBufferize(
      module, relations, workLimit, external);
}
} // namespace wafer::test
#endif
