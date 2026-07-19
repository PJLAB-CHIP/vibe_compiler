//===- StructuredOptimization.h - Structured normalization seam -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_STRUCTUREDOPTIMIZATION_H
#define WAFER_TRANSFORMS_STRUCTUREDOPTIMIZATION_H

#include "Wafer/Support/OptimizationMechanism.h"

#include <functional>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
class Pass;
class PassInstrumentation;
class ModuleOp;
} // namespace mlir

namespace wafer {

using OptimizationPassFactory =
    std::function<std::unique_ptr<mlir::Pass>()>;

/// Wraps one owner pass in the validating optimization invocation gateway.
/// The wrapper is intentionally not registered as a user-visible pass.
std::unique_ptr<mlir::Pass> createOptimizationInvocationPass(
    MechanismKey key, OptimizationCutPoint cutPoint,
    OptimizationPassFactory factory, uint64_t invocationOrdinal = 0);

enum class TensorNormalizationStatus : uint8_t {
  Success,
  UnsupportedSemantic,
  InvalidIR,
  ResourceExhausted,
  InternalInvariant,
};

enum class TensorNormalizationFamily : uint8_t {
  ProgramEnvelope,
  StructuredComputeDPS,
  DPSInitReduction,
  ScalarRegion,
  TensorRelation,
  StructuredControl,
  LogicalCollective,
};

struct TensorNormalizationDiagnostic {
  TensorNormalizationFamily family =
      TensorNormalizationFamily::ProgramEnvelope;
  std::string reason;
  std::string canonicalOperationPath;
};

struct TensorNormalizationWorkSummary {
  uint64_t initialOperationCount = 0;
  uint64_t initialEdgeCount = 0;
  uint64_t initialDimensionCount = 0;
  uint64_t fuelLimit = 0;
  uint64_t fuelConsumed = 0;
  uint64_t operationVisits = 0;
  uint64_t rewriteAttempts = 0;
  uint64_t committedRewrites = 0;
  uint64_t newOperations = 0;
  uint64_t relationNodes = 0;
};

struct TensorNormalizationOutcome {
  TensorNormalizationStatus status =
      TensorNormalizationStatus::InternalInvariant;
  bool changed = false;
  std::optional<TensorNormalizationDiagnostic> diagnostic;
  TensorNormalizationWorkSummary work;
};

enum class SelectedPayloadNormalizationStatus : uint8_t {
  Success,
  UnsupportedSemantic,
  InvalidIR,
  ResourceExhausted,
  InternalInvariant,
};

enum class SelectedPayloadNormalizationFamily : uint8_t {
  ProgramEnvelope,
  MemRefRelation,
  AllocationRootLayout,
  EffectCoverage,
  Completion,
  Traversal,
};

struct SelectedPayloadNormalizationDiagnostic {
  SelectedPayloadNormalizationFamily family =
      SelectedPayloadNormalizationFamily::ProgramEnvelope;
  std::string reason;
  std::string canonicalOperationPath;
};

struct SelectedPayloadNormalizationWorkSummary {
  uint64_t initialOperationCount = 0;
  uint64_t initialEdgeCount = 0;
  uint64_t initialRegionCount = 0;
  uint64_t initialViewRootNodes = 0;
  uint64_t initialEffectEntries = 0;
  uint64_t fuelLimit = 0;
  uint64_t fuelConsumed = 0;
  uint64_t operationVisits = 0;
  uint64_t regionVisits = 0;
  uint64_t rewriteAttempts = 0;
  uint64_t committedRewrites = 0;
  uint64_t newOperations = 0;
  uint64_t proofNodes = 0;
};

struct SelectedPayloadNormalizationOutcome {
  SelectedPayloadNormalizationStatus status =
      SelectedPayloadNormalizationStatus::InternalInvariant;
  bool changed = false;
  std::optional<SelectedPayloadNormalizationDiagnostic> diagnostic;
  SelectedPayloadNormalizationWorkSummary work;
};

/// Read-only description of a gateway wrapper.  This is used by pipeline
/// audits and tests to inspect the wrapped owner without making that owner a
/// second user-visible pipeline entry.
struct OptimizationInvocationDescription {
  MechanismKey key;
  OptimizationCutPoint cutPoint;
  std::string ownerTextualPipeline;
};

std::optional<OptimizationInvocationDescription>
describeOptimizationInvocationPass(const mlir::Pass &pass);

/// Debug-only pass-manager instrumentation used by wafer-opt.  It maps raw
/// registered upstream pass-family execution back to the canonical mechanism
/// registry.  Nested owner passes already running inside an explicit gateway
/// are suppressed to preserve exactly-once telemetry.
std::unique_ptr<mlir::PassInstrumentation>
createDebugOptimizationInvocationInstrumentation();

/// Clones the complete rank-local module, performs bounded deterministic
/// required rewrites, verifies the postcondition, and commits the clone only
/// on Success.  Every non-success leaves the source module untouched.
TensorNormalizationOutcome
normalizeRequiredTensorModule(mlir::ModuleOp module);

/// Pure-read postcondition check for the pre-bufferization structured tensor
/// normal form.  It never repairs input IR.
TensorNormalizationOutcome
verifyRequiredTensorNormalForm(mlir::ModuleOp module);

std::unique_ptr<mlir::Pass> createRequiredTensorNormalizationPass();

SelectedPayloadNormalizationOutcome
normalizeSelectedPhysicalPayload(mlir::ModuleOp module);
SelectedPayloadNormalizationOutcome
verifySelectedPhysicalPayloadNormalForm(mlir::ModuleOp module);
std::unique_ptr<mlir::Pass> createSelectedPayloadNormalizationPass();

} // namespace wafer

#endif // WAFER_TRANSFORMS_STRUCTUREDOPTIMIZATION_H
