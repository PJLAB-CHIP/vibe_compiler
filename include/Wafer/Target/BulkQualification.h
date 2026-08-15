//===- BulkQualification.h - Auditable bulk backend qualification -*- C++
//-*-===//

#ifndef WAFER_TARGET_BULKQUALIFICATION_H
#define WAFER_TARGET_BULKQUALIFICATION_H

#include "Wafer/Target/BulkTensorNumeric.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace wafer {

/// Exact deterministic GEMM corpus identity used by the offline three-stage
/// qualification producer. Shapes and layouts remain example/domain fields;
/// runtime qualification still binds the resolved command and payload digests.
class BulkQualificationSpec {
public:
  BulkQualificationSpec() = delete;

  static llvm::Expected<BulkQualificationSpec>
  create(LogicalFormat format, uint64_t m, uint64_t k, uint64_t n,
         uint64_t batchCount, PhysicalTensorLayout lhsLayout,
         PhysicalTensorLayout rhsLayout, PhysicalTensorLayout destinationLayout,
         uint64_t seed);

  /// Creates a held-out row whose exact target-physical payload is supplied
  /// by an offline source adapter. The bytes are embedded in the canonical
  /// spec, so validation never reopens source paths or regenerates payload.
  static llvm::Expected<BulkQualificationSpec> createWithPhysicalPayload(
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
  BulkQualificationSpec(
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

/// Deterministically materialized command and physical payload for one
/// qualification row. This is also the only supported way for tests and
/// runtime callers to reconstruct the exact held-out row named by a record.
class BulkQualificationCase {
public:
  BulkQualificationCase() = delete;

  const BulkQualificationSpec &getSpec() const { return spec; }
  const ResolvedNumericCommand &getCommand() const { return command; }
  llvm::ArrayRef<BulkTensorStorage> getInputs() const { return inputs; }
  const BulkTensorStorage &getDestinationTemplate() const {
    return destinationTemplate;
  }

private:
  friend llvm::Expected<BulkQualificationCase>
  materializeBulkQualificationCase(BulkQualificationSpec spec,
                                   BulkNumericWorkBudget bulkBudget);

  BulkQualificationCase(BulkQualificationSpec spec,
                        ResolvedNumericCommand command,
                        std::vector<BulkTensorStorage> inputs,
                        BulkTensorStorage destinationTemplate)
      : spec(std::move(spec)), command(std::move(command)),
        inputs(std::move(inputs)),
        destinationTemplate(std::move(destinationTemplate)) {}

  BulkQualificationSpec spec;
  ResolvedNumericCommand command;
  std::vector<BulkTensorStorage> inputs;
  BulkTensorStorage destinationTemplate;
};

llvm::Expected<BulkQualificationCase>
materializeBulkQualificationCase(BulkQualificationSpec spec,
                                 BulkNumericWorkBudget bulkBudget);

struct BulkQualificationTolerance {
  double maximumAbsoluteError = 0.0;
  double maximumRelativeError = 0.0;
};

/// Fully parsed and canonical validation record. It produces a qualified
/// execution only for the exact environment, resolved command, payload and
/// destination template recorded by the validation run.
class VerifiedBulkQualificationRecord {
public:
  VerifiedBulkQualificationRecord() = delete;

  llvm::StringRef getRecordDigest() const { return recordDigest; }
  llvm::StringRef getPolicyDigest() const { return policyDigest; }
  llvm::StringRef getAdapterDigest() const { return adapterDigest; }
  llvm::StringRef getSemanticProfileDigest() const {
    return semanticProfileDigest;
  }
  llvm::StringRef getSpecDigest() const { return specDigest; }
  llvm::StringRef getEnvironmentDigest() const { return environmentDigest; }
  llvm::StringRef getResolutionDigest() const { return resolutionDigest; }
  llvm::StringRef getImplementation() const { return implementation; }
  llvm::StringRef getResolvedDescriptorDigest() const {
    return resolvedDescriptorDigest;
  }
  BulkQualificationKind getKind() const { return kind; }

  llvm::Expected<QualifiedBulkExecution>
  qualifyExecution(const BulkExecutionEnvironment &environment,
                   const ResolvedNumericCommand &command,
                   llvm::ArrayRef<BulkTensorStorage> inputs,
                   const BulkTensorStorage &destinationTemplate) const;

private:
  friend llvm::Expected<VerifiedBulkQualificationRecord>
  loadVerifiedBulkQualificationRecord(llvm::StringRef path);

  VerifiedBulkQualificationRecord(
      std::string recordDigest, std::string policyDigest,
      std::string adapterDigest, std::string semanticProfileDigest,
      std::string specDigest, std::string environmentDigest,
      std::string resolutionDigest, std::string inputPayloadDigest,
      std::string destinationTemplateDigest,
      std::string expectedBackendOutputDigest, std::string implementation,
      std::string resolvedDescriptorDigest, BulkQualificationKind kind,
      FormalNumericExceptionFlags formalFlags)
      : recordDigest(std::move(recordDigest)),
        policyDigest(std::move(policyDigest)),
        adapterDigest(std::move(adapterDigest)),
        semanticProfileDigest(std::move(semanticProfileDigest)),
        specDigest(std::move(specDigest)),
        environmentDigest(std::move(environmentDigest)),
        resolutionDigest(std::move(resolutionDigest)),
        inputPayloadDigest(std::move(inputPayloadDigest)),
        destinationTemplateDigest(std::move(destinationTemplateDigest)),
        expectedBackendOutputDigest(std::move(expectedBackendOutputDigest)),
        implementation(std::move(implementation)),
        resolvedDescriptorDigest(std::move(resolvedDescriptorDigest)),
        kind(kind), formalFlags(formalFlags) {}

  std::string recordDigest;
  std::string policyDigest;
  std::string adapterDigest;
  std::string semanticProfileDigest;
  std::string specDigest;
  std::string environmentDigest;
  std::string resolutionDigest;
  std::string inputPayloadDigest;
  std::string destinationTemplateDigest;
  std::string expectedBackendOutputDigest;
  std::string implementation;
  std::string resolvedDescriptorDigest;
  BulkQualificationKind kind;
  FormalNumericExceptionFlags formalFlags;
};

/// Reads a canonical qualification spec. Unknown, missing, duplicate or
/// noncanonical fields fail before a command or tensor is constructed.
llvm::Error writeBulkQualificationSpec(const BulkQualificationSpec &spec,
                                       llvm::StringRef path);
llvm::Expected<BulkQualificationSpec>
loadBulkQualificationSpec(llvm::StringRef path);

/// Offline producer stages. Every output file is created atomically and is
/// never replaced.
/// Freeze reads calibration and pre-registers a disjoint held-out spec without
/// executing it; validate is the first stage allowed to execute held-out data.
llvm::Error calibrateBulkBackend(const BulkExecutionEnvironment &environment,
                                 llvm::StringRef specPath,
                                 llvm::StringRef outputPath,
                                 FormalNumericWorkBudget formalBudget,
                                 BulkNumericWorkBudget bulkBudget);
llvm::Error freezeBulkBackendPolicy(llvm::StringRef calibrationPath,
                                    llvm::StringRef heldOutSpecPath,
                                    llvm::StringRef outputPath,
                                    BulkQualificationTolerance tolerance);
llvm::Error validateBulkBackend(const BulkExecutionEnvironment &environment,
                                llvm::StringRef policyPath,
                                llvm::StringRef outputPath,
                                FormalNumericWorkBudget formalBudget,
                                BulkNumericWorkBudget bulkBudget);

llvm::Expected<VerifiedBulkQualificationRecord>
loadVerifiedBulkQualificationRecord(llvm::StringRef path);

} // namespace wafer

#endif // WAFER_TARGET_BULKQUALIFICATION_H
