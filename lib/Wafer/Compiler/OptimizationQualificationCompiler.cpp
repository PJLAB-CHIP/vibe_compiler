//===- OptimizationQualificationCompiler.cpp - Internal compile seam ---===//

#include "OptimizationQualificationCompiler.h"

#include "CompilationInternal.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <limits>

namespace wafer::compiler::qualification_internal {
namespace {

static std::optional<uint64_t>
computeDDRHighWaterBytes(const ExecutableBundle &bundle) {
  uint64_t highWater = 0;
  for (const RankExecutable &rank : bundle.getRankExecutables()) {
    bool valid = true;
    rank.getModule().walk([&](mlir::memref::AllocOp allocation) {
      if (!valid || !isWaferDDRMemRefType(allocation.getType()))
        return;
      auto offset =
          allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
      std::optional<WaferPhysicalTensorInfo> info =
          computeWaferPhysicalTensorInfo(allocation.getType());
      if (!offset || offset.getOffset() < 0 || !info ||
          info->physicalBytes < 0 ||
          static_cast<uint64_t>(info->physicalBytes) >
              std::numeric_limits<uint64_t>::max() -
                  static_cast<uint64_t>(offset.getOffset())) {
        valid = false;
        return;
      }
      highWater =
          std::max(highWater, static_cast<uint64_t>(offset.getOffset()) +
                                  static_cast<uint64_t>(info->physicalBytes));
    });
    if (!valid)
      return std::nullopt;
  }
  return highWater;
}

static ExactStaticVectorEvidenceV1
computeStaticMetrics(const ExecutableBundle &bundle) {
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (const RankExecutable &rank : bundle.getRankExecutables())
    roots.push_back(rank.getModule().getOperation());
  analysis::WholeCardInstructionProgramCost cost =
      analysis::analyzeWholeCardInstructionProgramCost(
          roots, analysis::getTargetScheduleCostPolicy(
                     bundle.getExecutionConfig().getTargetProfileId()));
  auto known = [](const analysis::ScheduleCostMetric &metric)
      -> std::optional<uint64_t> {
    return metric.isKnown() ? std::optional<uint64_t>(metric.value)
                            : std::nullopt;
  };
  RegistryRefV1 registry = getCurrentStaticMetricRegistryRefV1();
  ExactStaticVectorEvidenceV1 result;
  result.registrySchema = registry.registrySchema;
  result.registryDigest = registry.registryDigest;
  result.orderedMetrics = {
      {1, computeDDRHighWaterBytes(bundle)},
      {2, known(cost.maximumRankSPMHighWaterBytes)},
      {3, known(cost.aggregateSPMMovementBytes)},
      {4, known(cost.aggregateInstructionCount)},
  };
  return result;
}

} // namespace

mlir::FailureOr<QualificationCompilationProductV1>
compileProgramWithOptimizationConfiguration(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain,
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &optimizationConfiguration,
    EquivalentInputVariantV1 inputVariant, llvm::raw_ostream &diagnostics) {
  std::string policyDiagnostic;
  if (!validateOptimizationQualificationProposal(proposal, &policyDiagnostic) ||
      !validateOptimizationConfiguration(proposal, optimizationConfiguration,
                                         &policyDiagnostic)) {
    detail::reject(diagnostics,
                   "invalid optimization qualification configuration: " +
                       policyDiagnostic);
    return mlir::failure();
  }
  if (inputVariant != EquivalentInputVariantV1::Original &&
      inputVariant != EquivalentInputVariantV1::Metamorphic) {
    detail::reject(diagnostics, "unknown equivalent input variant");
    return mlir::failure();
  }
  detail::CompilationOptimizationPolicyV1 policy{
      proposal, optimizationConfiguration, inputVariant};
  std::optional<ExecutableBundle> retained;
  std::optional<TargetLLVMModuleBundle> retainedTargetLLVM;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, std::nullopt, std::nullopt,
          std::nullopt, &retained, &retainedTargetLLVM, policy)) ||
      !retained || !retainedTargetLLVM)
    return mlir::failure();
  ExactStaticVectorEvidenceV1 staticMetrics = computeStaticMetrics(*retained);
  return QualificationCompilationProductV1{std::move(*retained),
                                           std::move(*retainedTargetLLVM),
                                           std::move(staticMetrics)};
}

} // namespace wafer::compiler::qualification_internal
