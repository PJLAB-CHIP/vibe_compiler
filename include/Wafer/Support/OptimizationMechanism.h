//===- OptimizationMechanism.h - Typed optimization audit seam -*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONMECHANISM_H
#define WAFER_SUPPORT_OPTIMIZATIONMECHANISM_H

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer {

using OptimizationDigest = std::array<uint8_t, 32>;

struct InvocationPreparationV1;
struct InvocationTelemetryV1;
struct OptimizationInvocationScopeContextV1;

/// Stable semantic identity shared by invocation sites, audit specs and
/// qualification telemetry.  The integer is append-only and is deliberately
/// independent of pass names, registration order and workload details.
struct MechanismKey {
  uint32_t semanticId = 0;

  friend bool operator==(MechanismKey lhs, MechanismKey rhs) {
    return lhs.semanticId == rhs.semanticId;
  }
  friend bool operator!=(MechanismKey lhs, MechanismKey rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(MechanismKey lhs, MechanismKey rhs) {
    return lhs.semanticId < rhs.semanticId;
  }
};

enum class OptimizationCutPoint : uint32_t {
  FrontendProgramImport = 0,
  PreSPMDStableHLOModule = 1,
  SPMDPartitionTransaction = 2,
  PostSPMDStableHLOModule = 3,
  StructuredTensorModule = 4,
  SelectedPhysicalPayloadModule = 5,
  FinalInstructionModule = 6,
  TargetLLVMModule = 7,
  DevicePublicationTransaction = 8,
};

enum class InvocationOutcome : uint8_t {
  Applied,
  NoChange,
  NotApplicable,
  Unsupported,
  ResourceExhausted,
  Invalid,
  Cancelled,
};

enum class InvocationEvidenceKind : uint8_t {
  Rewrite,
  BackendAction,
  InvocationOnly,
};

struct MechanismDescriptor {
  MechanismKey key;
  OptimizationCutPoint cutPoint;
  InvocationEvidenceKind evidenceKind;
  const char *displayLabel;
};

/// Canonical live terminal emitted exactly once for one gateway invocation.
/// The optional action fields are populated from the actual launched action;
/// they are never reconstructed from the audit registry.
struct OptimizationInvocationTelemetry {
  MechanismKey key;
  OptimizationCutPoint cutPoint;
  InvocationOutcome outcome = InvocationOutcome::Invalid;
  uint64_t rewriteCount = 0;
  uint64_t successfulBackendActionCount = 0;
  uint64_t workUnits = 0;
  uint64_t invocationOrdinal = 0;
  OptimizationDigest inputSnapshotDigest{};
  OptimizationDigest observedToolDigest{};
  OptimizationDigest observedOutputDigest{};
  std::string actionExecutor;
  std::vector<std::string> actionArgv;
};

class OptimizationInvocationRecorder {
public:
  virtual ~OptimizationInvocationRecorder() = default;
  virtual bool beginInvocation(const InvocationPreparationV1 &preparation,
                               std::string *diagnostic = nullptr) = 0;
  virtual bool recordInvocationTerminal(
      const InvocationTelemetryV1 &telemetry,
      std::string *diagnostic = nullptr) = 0;
};

/// Installs a process-wide qualification recorder.  Production has no
/// recorder and
/// therefore retains no side table or artifact payload, while every invocation
/// still passes through the same validating gateway.  Only one scoped
/// recorder may be active at a time; nested or concurrent installation fails
/// closed.
class ScopedOptimizationInvocationRecorder {
public:
  explicit ScopedOptimizationInvocationRecorder(
      std::shared_ptr<OptimizationInvocationRecorder> recorder,
      const OptimizationInvocationScopeContextV1 &scope);
  ~ScopedOptimizationInvocationRecorder();

  ScopedOptimizationInvocationRecorder(
      const ScopedOptimizationInvocationRecorder &) = delete;
  ScopedOptimizationInvocationRecorder &
  operator=(const ScopedOptimizationInvocationRecorder &) = delete;

  bool installed() const { return installed_; }

private:
  bool installed_ = false;
};

/// Opaque begin token. A token is created before the owner executes and can
/// commit exactly one terminal afterward. It is move-only so ownership cannot
/// accidentally fork across parallel completion paths.
class OptimizationInvocationTokenV1 {
public:
  OptimizationInvocationTokenV1() = default;
  OptimizationInvocationTokenV1(OptimizationInvocationTokenV1 &&) = default;
  OptimizationInvocationTokenV1 &
  operator=(OptimizationInvocationTokenV1 &&) = default;
  OptimizationInvocationTokenV1(const OptimizationInvocationTokenV1 &) =
      delete;
  OptimizationInvocationTokenV1 &
  operator=(const OptimizationInvocationTokenV1 &) = delete;

  bool begun() const { return state_ != nullptr; }

private:
  std::shared_ptr<void> state_;
  friend bool beginOptimizationInvocationV1(
      MechanismKey, OptimizationCutPoint, uint64_t,
      const OptimizationDigest &, OptimizationInvocationTokenV1 &,
      std::string *);
  friend bool commitOptimizationInvocationV1(
      OptimizationInvocationTokenV1 &, const OptimizationInvocationTelemetry &,
      std::string *);
};

/// Establishes identity and input evidence before executing an owner.
bool beginOptimizationInvocationV1(
    MechanismKey key, OptimizationCutPoint cutPoint,
    uint64_t invocationOrdinal, const OptimizationDigest &inputSnapshotDigest,
    OptimizationInvocationTokenV1 &token,
    std::string *diagnostic = nullptr);

/// Commits the one terminal for a begun invocation. The result key/cut/
/// ordinal/input digest must match the begin token exactly.
bool commitOptimizationInvocationV1(
    OptimizationInvocationTokenV1 &token,
    const OptimizationInvocationTelemetry &telemetry,
    std::string *diagnostic = nullptr);

std::optional<MechanismDescriptor>
lookupMechanismDescriptor(MechanismKey key);
std::vector<MechanismDescriptor> getAllMechanismDescriptors();

namespace mechanism {

inline constexpr MechanismKey FrontendProgramImport{1};
inline constexpr MechanismKey SpmdPartition{2};
inline constexpr MechanismKey StablehloCollectiveNormalization{3};
inline constexpr MechanismKey StablehloStructuredLegalization{4};
inline constexpr MechanismKey RequiredTensorNormalization{5};
inline constexpr MechanismKey StablehloCleanup{6};
inline constexpr MechanismKey StructuredTensorCleanup{7};
inline constexpr MechanismKey CandidateCommitCleanup{8};
inline constexpr MechanismKey PreBufferizationCleanup{9};
inline constexpr MechanismKey FunctionBoundaryBufferization{10};
inline constexpr MechanismKey PostBufferizationCleanup{11};
inline constexpr MechanismKey PostMemoryPlanningCleanup{12};
inline constexpr MechanismKey StructuredTilingInterface{13};
inline constexpr MechanismKey TiledShapeConstruction{14};
inline constexpr MechanismKey ProducerSliceFusion{15};
inline constexpr MechanismKey TileDataflowMaterialization{16};
inline constexpr MechanismKey InstructionLowering{17};
inline constexpr MechanismKey TargetLLVMConversion{18};
inline constexpr MechanismKey DeviceObjectCompilation{19};
inline constexpr MechanismKey DeviceRuntimeCompilation{20};
inline constexpr MechanismKey DeviceGarbageCollectionLink{21};
inline constexpr MechanismKey ScalarCommonSubexpressionElimination{22};
inline constexpr MechanismKey SparseConditionalConstantPropagation{23};
inline constexpr MechanismKey LinalgTransformFamily{24};
inline constexpr MechanismKey TensorTransformFamily{25};
inline constexpr MechanismKey ScfTransformFamily{26};
inline constexpr MechanismKey BufferizationTransformFamily{27};
inline constexpr MechanismKey ArithTransformFamily{28};
inline constexpr MechanismKey SelectedPayloadNormalization{29};
inline constexpr MechanismKey PostLegalizationCanonicalization{30};
inline constexpr MechanismKey StructuredTensorCanonicalization{31};

} // namespace mechanism
} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONMECHANISM_H
