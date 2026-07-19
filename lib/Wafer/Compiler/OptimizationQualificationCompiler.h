//===- OptimizationQualificationCompiler.h - Internal compile seam -*- C++
//-*-===//

#ifndef WAFER_COMPILER_OPTIMIZATIONQUALIFICATIONCOMPILER_H
#define WAFER_COMPILER_OPTIMIZATIONQUALIFICATIONCOMPILER_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Support/OptimizationQualification.h"
#include "Wafer/Support/OptimizationQualificationEvidence.h"

namespace wafer::compiler::qualification_internal {

struct QualificationCompilationProductV1 {
  ExecutableBundle executableBundle;
  TargetLLVMModuleBundle targetLLVMModuleBundle;
  ExactStaticVectorEvidenceV1 staticMetrics;
};

/// Runs the exact production source-to-package transaction with one validated
/// fixed/cleanup optimization configuration. This seam is linked only by the
/// qualification driver; it is not a user compiler option or a second pipeline.
mlir::FailureOr<QualificationCompilationProductV1>
compileProgramWithOptimizationConfiguration(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain,
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &optimizationConfiguration,
    EquivalentInputVariantV1 inputVariant,
    uint32_t requiredTensorNormalizationRepetitions,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::qualification_internal

#endif // WAFER_COMPILER_OPTIMIZATIONQUALIFICATIONCOMPILER_H
