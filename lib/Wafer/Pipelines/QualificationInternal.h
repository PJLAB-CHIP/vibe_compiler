//===- QualificationInternal.h - Internal optimization pipeline configurations
//-*- C++ -*-===//

#ifndef WAFER_PIPELINES_QUALIFICATIONINTERNAL_H
#define WAFER_PIPELINES_QUALIFICATIONINTERNAL_H

#include "Wafer/Support/OptimizationQualification.h"

#include <cstdint>
#include <string>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace wafer::qualification_internal {

/// Qualification-only builders for the exact production pipeline bodies.
/// They are deliberately private to the compiler and have no CLI/pass option.
/// Production calls the same bodies with the active AllOn configuration.
bool buildStablehloToLinalgPipeline(
    mlir::OpPassManager &pm, const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration,
    std::string *diagnostic = nullptr,
    EquivalentInputVariantV1 inputVariant = EquivalentInputVariantV1::Original,
    uint32_t requiredTensorNormalizationRepetitions = 1);

bool buildScheduleTensorProgramToSelectedInstrPipeline(
    mlir::OpPassManager &pm, int64_t logicalRank,
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration,
    std::string *diagnostic = nullptr);

bool buildFinalizeScheduledTensorProgramPipeline(
    mlir::OpPassManager &pm, const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration,
    std::string *diagnostic = nullptr, uint64_t invocationOrdinal = 0);

} // namespace wafer::qualification_internal

#endif // WAFER_PIPELINES_QUALIFICATIONINTERNAL_H
