//===- OptimizationQualificationExecution.h - Isolated run result -*- C++
//-*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONEXECUTION_H
#define WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONEXECUTION_H

#include "Wafer/Support/OptimizationQualificationEvidence.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer {

/// Canonical result produced by exactly one isolated qualification compiler
/// process.  The parent runner accepts no stdout-derived evidence: it parses
/// this record, revalidates every invocation terminal, and then aggregates the
/// completed run transaction.
struct OptimizationQualificationExecutionResultV1 {
  uint16_t schemaVersion = 1;
  AdoptionDigest qualificationRunDigest{};
  AdoptionDigest inputSnapshotDigest{};
  QualificationCaseKeyV1 caseKey;
  OptimizationConfigurationV1 configuration;
  uint32_t requiredTensorNormalizationRepetitions = 1;
  AdoptionDigest outputArtifactDigest{};
  ExactStaticVectorEvidenceV1 staticMetrics;
  std::optional<AdoptionDigest> targetModelEvidenceDigest;
  std::vector<InvocationTelemetryV1> invocationTerminals;
};

std::vector<uint8_t> encodeOptimizationQualificationExecutionResultV1(
    const OptimizationQualificationExecutionResultV1 &value);
AdoptionDigest digestOptimizationQualificationExecutionResultV1(
    const OptimizationQualificationExecutionResultV1 &value);
bool decodeCanonicalOptimizationQualificationExecutionResultV1(
    const std::vector<uint8_t> &bytes,
    OptimizationQualificationExecutionResultV1 &value,
    std::string *diagnostic = nullptr);
bool validateOptimizationQualificationExecutionResultV1(
    const OptimizationQualificationExecutionResultV1 &value,
    std::string *diagnostic = nullptr);

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONEXECUTION_H
