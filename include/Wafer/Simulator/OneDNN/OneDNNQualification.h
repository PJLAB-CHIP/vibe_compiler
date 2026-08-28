//===- OneDNNQualification.h - Auditable onednn backend qualification -*- C++
//-*-===//

#ifndef WAFER_TARGET_ONEDNNQUALIFICATION_H
#define WAFER_TARGET_ONEDNNQUALIFICATION_H

#include "Wafer/Simulator/OneDNN/OneDNNTensorNumeric.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace wafer {

/// Exact deterministic GEMM corpus identity used by the offline three-stage
/// qualification producer. Shapes and layouts remain example/domain fields;
/// runtime qualification still binds the operation and payload digests.
class OneDNNQualificationSpec {
public:
  OneDNNQualificationSpec() = delete;

  static llvm::Expected<OneDNNQualificationSpec>
  create(LogicalFormat format, uint64_t m, uint64_t k, uint64_t n,
         uint64_t batchCount, PhysicalTensorLayout lhsLayout,
         PhysicalTensorLayout rhsLayout, PhysicalTensorLayout destinationLayout,
         uint64_t seed);

  /// Creates a held-out row whose exact target-physical payload is supplied
  /// by an offline source adapter. The bytes are embedded in the canonical
  /// spec, so validation never reopens source paths or regenerates payload.
  static llvm::Expected<OneDNNQualificationSpec> createWithPhysicalPayload(
      LogicalFormat format, uint64_t m, uint64_t k, uint64_t n,
      uint64_t batchCount, PhysicalTensorLayout lhsLayout,
      PhysicalTensorLayout rhsLayout, PhysicalTensorLayout destinationLayout,
      uint64_t seed, std::vector<uint8_t> lhsPhysical,
      std::vector<uint8_t> rhsPhysical,
      std::vector<uint8_t> destinationTemplatePhysical);

  LogicalFormat getFormat() const { return format; }
  uint64_t getM() const { return m; }
  uint64_t getK() const { return k; }
  uint64_t getN() const { return n; }
  uint64_t getBatchCount() const { return batchCount; }
  PhysicalTensorLayout getLHSLayout() const { return lhsLayout; }
  PhysicalTensorLayout getRHSLayout() const { return rhsLayout; }
  PhysicalTensorLayout getDestinationLayout() const {
    return destinationLayout;
  }
  uint64_t getSeed() const { return seed; }
  bool hasExplicitPhysicalPayload() const { return explicitPhysicalPayload; }
  llvm::ArrayRef<uint8_t> getLHSPhysicalPayload() const { return lhsPhysical; }
  llvm::ArrayRef<uint8_t> getRHSPhysicalPayload() const { return rhsPhysical; }
  llvm::ArrayRef<uint8_t> getDestinationTemplatePhysicalPayload() const {
    return destinationTemplatePhysical;
  }
  llvm::StringRef getDigest() const { return digest; }

private:
  OneDNNQualificationSpec(
      LogicalFormat format, uint64_t m, uint64_t k, uint64_t n,
      uint64_t batchCount, PhysicalTensorLayout lhsLayout,
      PhysicalTensorLayout rhsLayout, PhysicalTensorLayout destinationLayout,
      uint64_t seed, bool explicitPhysicalPayload,
      std::vector<uint8_t> lhsPhysical, std::vector<uint8_t> rhsPhysical,
      std::vector<uint8_t> destinationTemplatePhysical, std::string digest)
      : format(format), m(m), k(k), n(n), batchCount(batchCount),
        lhsLayout(lhsLayout), rhsLayout(rhsLayout),
        destinationLayout(destinationLayout), seed(seed),
        explicitPhysicalPayload(explicitPhysicalPayload),
        lhsPhysical(std::move(lhsPhysical)),
        rhsPhysical(std::move(rhsPhysical)),
        destinationTemplatePhysical(std::move(destinationTemplatePhysical)),
        digest(std::move(digest)) {}

  LogicalFormat format;
  uint64_t m;
  uint64_t k;
  uint64_t n;
  uint64_t batchCount;
  PhysicalTensorLayout lhsLayout;
  PhysicalTensorLayout rhsLayout;
  PhysicalTensorLayout destinationLayout;
  uint64_t seed;
  bool explicitPhysicalPayload;
  std::vector<uint8_t> lhsPhysical;
  std::vector<uint8_t> rhsPhysical;
  std::vector<uint8_t> destinationTemplatePhysical;
  std::string digest;
};

/// Deterministically materialized operation and physical payload for one
/// qualification row. This is also the only supported way for tests and
/// runtime callers to reconstruct the exact held-out row named by a record.
class OneDNNQualificationCase {
public:
  OneDNNQualificationCase() = delete;

  const OneDNNQualificationSpec &getSpec() const { return spec; }
  const FormalGemmOperation &getOperation() const { return operation; }
  llvm::ArrayRef<OneDNNTensorStorage> getInputs() const { return inputs; }
  const OneDNNTensorStorage &getDestinationTemplate() const {
    return destinationTemplate;
  }

private:
  friend llvm::Expected<OneDNNQualificationCase>
  materializeOneDNNQualificationCase(OneDNNQualificationSpec spec,
                                     OneDNNNumericWorkBudget onednnBudget);

  OneDNNQualificationCase(OneDNNQualificationSpec spec,
                          FormalGemmOperation operation,
                          std::vector<OneDNNTensorStorage> inputs,
                          OneDNNTensorStorage destinationTemplate)
      : spec(std::move(spec)), operation(std::move(operation)),
        inputs(std::move(inputs)),
        destinationTemplate(std::move(destinationTemplate)) {}

  OneDNNQualificationSpec spec;
  FormalGemmOperation operation;
  std::vector<OneDNNTensorStorage> inputs;
  OneDNNTensorStorage destinationTemplate;
};

llvm::Expected<OneDNNQualificationCase>
materializeOneDNNQualificationCase(OneDNNQualificationSpec spec,
                                   OneDNNNumericWorkBudget onednnBudget);

struct OneDNNQualificationTolerance {
  double maximumAbsoluteError = 0.0;
  double maximumRelativeError = 0.0;
};

/// Fully parsed and canonical validation record. It produces a qualified
/// execution only for the exact environment, operation, payload and
/// destination template recorded by the validation run.
class VerifiedOneDNNQualificationRecord {
public:
  VerifiedOneDNNQualificationRecord() = delete;

  llvm::StringRef getRecordDigest() const { return recordDigest; }
  llvm::StringRef getPolicyDigest() const { return policyDigest; }
  llvm::StringRef getAdapterDigest() const { return adapterDigest; }
  llvm::StringRef getSpecDigest() const { return specDigest; }
  llvm::StringRef getEnvironmentDigest() const { return environmentDigest; }
  llvm::StringRef getProblemDigest() const { return problemDigest; }
  llvm::StringRef getImplementation() const { return implementation; }
  llvm::StringRef getResolvedDescriptorDigest() const {
    return resolvedDescriptorDigest;
  }
  OneDNNQualificationKind getKind() const { return kind; }

  llvm::Expected<QualifiedOneDNNExecution>
  qualifyExecution(const OneDNNExecutionEnvironment &environment,
                   const FormalGemmOperation &operation,
                   llvm::ArrayRef<OneDNNTensorStorage> inputs,
                   const OneDNNTensorStorage &destinationTemplate) const;

private:
  friend llvm::Expected<VerifiedOneDNNQualificationRecord>
  loadVerifiedOneDNNQualificationRecord(llvm::StringRef path);

  VerifiedOneDNNQualificationRecord(
      std::string recordDigest, std::string policyDigest,
      std::string adapterDigest, std::string specDigest,
      std::string environmentDigest, std::string problemDigest,
      std::string inputPayloadDigest, std::string destinationTemplateDigest,
      std::string expectedBackendOutputDigest, std::string implementation,
      std::string resolvedDescriptorDigest, OneDNNQualificationKind kind,
      FormalNumericExceptionFlags formalFlags)
      : recordDigest(std::move(recordDigest)),
        policyDigest(std::move(policyDigest)),
        adapterDigest(std::move(adapterDigest)),
        specDigest(std::move(specDigest)),
        environmentDigest(std::move(environmentDigest)),
        problemDigest(std::move(problemDigest)),
        inputPayloadDigest(std::move(inputPayloadDigest)),
        destinationTemplateDigest(std::move(destinationTemplateDigest)),
        expectedBackendOutputDigest(std::move(expectedBackendOutputDigest)),
        implementation(std::move(implementation)),
        resolvedDescriptorDigest(std::move(resolvedDescriptorDigest)),
        kind(kind), formalFlags(formalFlags) {}

  std::string recordDigest;
  std::string policyDigest;
  std::string adapterDigest;
  std::string specDigest;
  std::string environmentDigest;
  std::string problemDigest;
  std::string inputPayloadDigest;
  std::string destinationTemplateDigest;
  std::string expectedBackendOutputDigest;
  std::string implementation;
  std::string resolvedDescriptorDigest;
  OneDNNQualificationKind kind;
  FormalNumericExceptionFlags formalFlags;
};

/// Reads a canonical qualification spec. Unknown, missing, duplicate or
/// noncanonical fields fail before a operation or tensor is constructed.
llvm::Error writeOneDNNQualificationSpec(const OneDNNQualificationSpec &spec,
                                         llvm::StringRef path);
llvm::Expected<OneDNNQualificationSpec>
loadOneDNNQualificationSpec(llvm::StringRef path);

/// Offline producer stages. Every output file is created atomically and is
/// never replaced.
/// Freeze reads calibration and pre-registers a disjoint held-out spec without
/// executing it; validate is the first stage allowed to execute held-out data.
llvm::Error
calibrateOneDNNBackend(const OneDNNExecutionEnvironment &environment,
                       llvm::StringRef specPath, llvm::StringRef outputPath,
                       FormalNumericWorkBudget formalBudget,
                       OneDNNNumericWorkBudget onednnBudget);
llvm::Error freezeOneDNNBackendPolicy(llvm::StringRef calibrationPath,
                                      llvm::StringRef heldOutSpecPath,
                                      llvm::StringRef outputPath,
                                      OneDNNQualificationTolerance tolerance);
llvm::Error validateOneDNNBackend(const OneDNNExecutionEnvironment &environment,
                                  llvm::StringRef policyPath,
                                  llvm::StringRef outputPath,
                                  FormalNumericWorkBudget formalBudget,
                                  OneDNNNumericWorkBudget onednnBudget);

llvm::Expected<VerifiedOneDNNQualificationRecord>
loadVerifiedOneDNNQualificationRecord(llvm::StringRef path);

} // namespace wafer

#endif // WAFER_TARGET_ONEDNNQUALIFICATION_H
