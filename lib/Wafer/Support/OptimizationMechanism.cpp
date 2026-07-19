//===- OptimizationMechanism.cpp - Typed optimization audit seam --------===//

#include "Wafer/Support/OptimizationMechanism.h"
#include "Wafer/Support/OptimizationAdoption.h"
#include "Wafer/Support/OptimizationInvocation.h"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <utility>

namespace wafer {
namespace {

using EK = InvocationEvidenceKind;
using CP = OptimizationCutPoint;

constexpr std::array<MechanismDescriptor, 31> descriptors = {{
    {mechanism::FrontendProgramImport, CP::FrontendProgramImport,
     EK::InvocationOnly, "frontend program import"},
    {mechanism::SpmdPartition, CP::SPMDPartitionTransaction,
     EK::BackendAction, "SPMD partition transaction"},
    {mechanism::StablehloCollectiveNormalization,
     CP::PostSPMDStableHLOModule, EK::Rewrite,
     "logical collective normalization"},
    {mechanism::StablehloStructuredLegalization,
     CP::PostSPMDStableHLOModule, EK::Rewrite,
     "structured tensor legalization"},
    {mechanism::RequiredTensorNormalization, CP::StructuredTensorModule,
     EK::Rewrite, "required tensor normalization"},
    {mechanism::StablehloCleanup, CP::StructuredTensorModule, EK::Rewrite,
     "post-legalization cleanup"},
    {mechanism::StructuredTensorCleanup, CP::StructuredTensorModule,
     EK::Rewrite, "structured tensor cleanup"},
    {mechanism::CandidateCommitCleanup, CP::SelectedPhysicalPayloadModule,
     EK::Rewrite, "candidate commit cleanup"},
    {mechanism::PreBufferizationCleanup,
     CP::SelectedPhysicalPayloadModule, EK::Rewrite,
     "pre-bufferization cleanup"},
    {mechanism::FunctionBoundaryBufferization,
     CP::SelectedPhysicalPayloadModule, EK::Rewrite,
     "function boundary bufferization"},
    {mechanism::PostBufferizationCleanup,
     CP::SelectedPhysicalPayloadModule, EK::Rewrite,
     "post-bufferization cleanup"},
    {mechanism::PostMemoryPlanningCleanup, CP::FinalInstructionModule,
     EK::Rewrite, "post-memory-planning cleanup"},
    {mechanism::StructuredTilingInterface, CP::StructuredTensorModule,
     EK::Rewrite, "structured tiling interface"},
    {mechanism::TiledShapeConstruction, CP::StructuredTensorModule,
     EK::Rewrite, "tiled shape construction"},
    {mechanism::ProducerSliceFusion, CP::StructuredTensorModule, EK::Rewrite,
     "producer slice fusion"},
    {mechanism::TileDataflowMaterialization, CP::StructuredTensorModule,
     EK::Rewrite, "tile dataflow materialization"},
    {mechanism::InstructionLowering, CP::SelectedPhysicalPayloadModule,
     EK::Rewrite, "instruction lowering"},
    {mechanism::TargetLLVMConversion, CP::FinalInstructionModule,
     EK::Rewrite, "target LLVM conversion"},
    {mechanism::DeviceObjectCompilation, CP::DevicePublicationTransaction,
     EK::BackendAction, "device object compilation"},
    {mechanism::DeviceRuntimeCompilation, CP::DevicePublicationTransaction,
     EK::BackendAction, "device runtime compilation"},
    {mechanism::DeviceGarbageCollectionLink,
     CP::DevicePublicationTransaction, EK::BackendAction,
     "device garbage-collecting link"},
    {mechanism::ScalarCommonSubexpressionElimination,
     CP::StructuredTensorModule, EK::Rewrite,
     "scalar and shape common-subexpression elimination"},
    {mechanism::SparseConditionalConstantPropagation,
     CP::StructuredTensorModule, EK::Rewrite,
     "sparse conditional constant propagation"},
    {mechanism::LinalgTransformFamily, CP::StructuredTensorModule,
     EK::Rewrite, "linalg transform family"},
    {mechanism::TensorTransformFamily, CP::StructuredTensorModule,
     EK::Rewrite, "tensor transform family"},
    {mechanism::ScfTransformFamily, CP::StructuredTensorModule,
     EK::Rewrite, "structured control-flow transform family"},
    {mechanism::BufferizationTransformFamily,
     CP::SelectedPhysicalPayloadModule, EK::Rewrite,
     "bufferization transform family"},
    {mechanism::ArithTransformFamily, CP::StructuredTensorModule,
     EK::Rewrite, "arithmetic transform family"},
    {mechanism::SelectedPayloadNormalization,
     CP::SelectedPhysicalPayloadModule, EK::Rewrite,
     "selected payload normalization"},
    {mechanism::PostLegalizationCanonicalization,
     CP::StructuredTensorModule, EK::Rewrite,
     "required post-legalization canonicalization"},
    {mechanism::StructuredTensorCanonicalization,
     CP::StructuredTensorModule, EK::Rewrite,
     "required structured tensor canonicalization"},
}};

std::mutex recorderMutex;
std::shared_ptr<OptimizationInvocationRecorder> activeRecorder;
std::optional<OptimizationInvocationScopeContextV1> activeScope;

static bool validateEvidenceShape(
    const MechanismDescriptor &descriptor,
    const OptimizationInvocationTelemetry &telemetry,
    std::string *diagnostic) {
  auto fail = [&](const char *message) {
    if (diagnostic)
      *diagnostic = message;
    return false;
  };

  switch (descriptor.evidenceKind) {
  case EK::Rewrite:
    if (telemetry.successfulBackendActionCount != 0 ||
        !telemetry.actionExecutor.empty() || !telemetry.actionArgv.empty())
      return fail("rewrite invocation carried backend-action evidence");
    break;
  case EK::BackendAction:
    if (telemetry.rewriteCount != 0)
      return fail("backend action invocation carried rewrite evidence");
    if (telemetry.outcome == InvocationOutcome::Applied &&
        telemetry.successfulBackendActionCount == 0)
      return fail("applied backend action has no successful action");
    break;
  case EK::InvocationOnly:
    if (telemetry.rewriteCount != 0 ||
        telemetry.successfulBackendActionCount != 0 ||
        !telemetry.actionExecutor.empty() || !telemetry.actionArgv.empty() ||
        telemetry.outcome == InvocationOutcome::Applied)
      return fail("invocation-only evidence carried an action or rewrite");
    break;
  }
  return true;
}

} // namespace

ScopedOptimizationInvocationRecorder::ScopedOptimizationInvocationRecorder(
    std::shared_ptr<OptimizationInvocationRecorder> recorder,
    const OptimizationInvocationScopeContextV1 &scope) {
  if (!recorder)
    return;
  if (std::all_of(scope.scopeDigest.begin(), scope.scopeDigest.end(),
                  [](uint8_t byte) { return byte == 0; }))
    return;
  if ((scope.scopeKind == InvocationScopeKindV1::QualificationRun) !=
      scope.qualificationCase.has_value())
    return;
  if (scope.invocationOrdinalBase %
          kOptimizationInvocationLocalOrdinalLimit !=
      0 ||
      scope.invocationOrdinalBase >
          std::numeric_limits<uint64_t>::max() -
              (kOptimizationInvocationLocalOrdinalLimit - 1))
    return;
  std::lock_guard<std::mutex> lock(recorderMutex);
  if (activeRecorder)
    return;
  activeRecorder = std::move(recorder);
  activeScope = scope;
  installed_ = true;
}

ScopedOptimizationInvocationRecorder::~ScopedOptimizationInvocationRecorder() {
  if (!installed_)
    return;
  std::lock_guard<std::mutex> lock(recorderMutex);
  activeRecorder.reset();
  activeScope.reset();
}

std::optional<MechanismDescriptor>
lookupMechanismDescriptor(MechanismKey key) {
  auto it = std::lower_bound(
      descriptors.begin(), descriptors.end(), key,
      [](const MechanismDescriptor &descriptor, MechanismKey query) {
        return descriptor.key < query;
      });
  if (it == descriptors.end() || it->key != key)
    return std::nullopt;
  return *it;
}

std::vector<MechanismDescriptor> getAllMechanismDescriptors() {
  return {descriptors.begin(), descriptors.end()};
}

namespace {

struct InvocationTokenState {
  MechanismDescriptor descriptor;
  AdoptionSpec spec;
  MechanismKey key;
  OptimizationCutPoint cutPoint;
  uint64_t invocationOrdinal = 0;
  OptimizationDigest inputSnapshotDigest{};
  std::shared_ptr<OptimizationInvocationRecorder> recorder;
  std::optional<InvocationPreparationV1> preparation;
  bool committed = false;
};

static bool validateInvocationRegistryBinding(
    MechanismKey key, OptimizationCutPoint cutPoint,
    MechanismDescriptor &descriptor, AdoptionSpec &spec,
    std::string *diagnostic) {
  auto foundDescriptor = lookupMechanismDescriptor(key);
  if (!foundDescriptor) {
    if (diagnostic)
      *diagnostic = "unknown or zero semantic mechanism id";
    return false;
  }
  descriptor = *foundDescriptor;
  if (descriptor.cutPoint != cutPoint) {
    if (diagnostic)
      *diagnostic = "mechanism invoked at the wrong optimization cut point";
    return false;
  }
  std::optional<AdoptionSpec> foundSpec = lookupAdoptionSpec(key);
  if (!foundSpec) {
    if (diagnostic)
      *diagnostic = "live mechanism has no adoption spec";
    return false;
  }
  spec = *foundSpec;
  if (spec.cutPoint != cutPoint ||
      spec.evidenceKind != descriptor.evidenceKind) {
    if (diagnostic)
      *diagnostic = "live mechanism disagrees with its canonical adoption spec";
    return false;
  }
  return true;
}

static InvocationTelemetryV1 buildTerminal(
    const InvocationTokenState &state,
    const OptimizationInvocationTelemetry &telemetry) {
  const InvocationPreparationV1 &preparation = *state.preparation;
  InvocationTelemetryV1 terminal;
  terminal.identity = preparation.identity;
  terminal.qualificationCase = preparation.qualificationCase;
  terminal.specDigest = preparation.specDigest;
  terminal.inputSnapshotDigest = preparation.inputSnapshotDigest;
  terminal.outcome = telemetry.outcome;
  if (telemetry.outcome == InvocationOutcome::Unsupported)
    terminal.terminalReason = ClosedReasonV1{
        getGlobalClosedReasonRefV1(
            GlobalClosedReasonV1::UnsupportedSemantic),
        std::nullopt};
  else if (telemetry.outcome == InvocationOutcome::ResourceExhausted)
    terminal.terminalReason = ClosedReasonV1{
        getGlobalClosedReasonRefV1(GlobalClosedReasonV1::ResourceExhausted),
        std::nullopt};
  else if (telemetry.outcome == InvocationOutcome::Cancelled)
    terminal.terminalReason = ClosedReasonV1{
        getGlobalClosedReasonRefV1(GlobalClosedReasonV1::Cancelled),
        std::nullopt};
  else if (telemetry.outcome == InvocationOutcome::Invalid)
    terminal.terminalReason = ClosedReasonV1{
        getGlobalClosedReasonRefV1(
            GlobalClosedReasonV1::InvalidOwnerTerminal),
        std::nullopt};
  terminal.rewriteCount = telemetry.rewriteCount;
  terminal.workSummary.workPolicyDigest =
      digestAdoptionWorkPolicyV1(state.spec.workPolicyKind);
  terminal.workSummary.orderedCounters = {{1, telemetry.workUnits}};
  if (!telemetry.actionArgv.empty()) {
    BackendActionEvidenceV1 action;
    action.invocationId = digestInvocationIdentityV1(terminal.identity);
    action.argv = telemetry.actionArgv;
    action.observedToolDigest = telemetry.observedToolDigest;
    action.observedOutputDigest = telemetry.observedOutputDigest;
    action.terminalStatus = telemetry.successfulBackendActionCount != 0
                                ? BackendActionStatusV1::Success
                                : BackendActionStatusV1::Failed;
    terminal.backendActions.push_back(std::move(action));
  }
  return terminal;
}

} // namespace

bool beginOptimizationInvocationV1(
    MechanismKey key, OptimizationCutPoint cutPoint,
    uint64_t invocationOrdinal, const OptimizationDigest &inputSnapshotDigest,
    OptimizationInvocationTokenV1 &token, std::string *diagnostic) {
  if (token.begun()) {
    if (diagnostic)
      *diagnostic = "optimization invocation token was already begun";
    return false;
  }
  if (std::all_of(inputSnapshotDigest.begin(), inputSnapshotDigest.end(),
                  [](uint8_t byte) { return byte == 0; })) {
    if (diagnostic)
      *diagnostic = "optimization invocation input snapshot digest is zero";
    return false;
  }
  auto state = std::make_shared<InvocationTokenState>();
  if (!validateInvocationRegistryBinding(key, cutPoint, state->descriptor,
                                         state->spec, diagnostic))
    return false;
  state->key = key;
  state->cutPoint = cutPoint;
  state->invocationOrdinal = invocationOrdinal;
  state->inputSnapshotDigest = inputSnapshotDigest;

  std::optional<OptimizationInvocationScopeContextV1> scope;
  {
    std::lock_guard<std::mutex> lock(recorderMutex);
    state->recorder = activeRecorder;
    scope = activeScope;
  }
  if (state->recorder) {
    if (invocationOrdinal >= kOptimizationInvocationLocalOrdinalLimit) {
      if (diagnostic)
        *diagnostic = "owner-local invocation ordinal exceeds its namespace";
      return false;
    }
    if (!scope) {
      if (diagnostic)
        *diagnostic = "invocation recorder has no scope context";
      return false;
    }
    std::optional<RegistryRefV1> site =
        lookupOptimizationInvocationSiteV1(key);
    if (!site) {
      if (diagnostic)
        *diagnostic = "live mechanism has no invocation-site registry row";
      return false;
    }
    InvocationPreparationV1 preparation;
    preparation.identity.scopeKind = scope->scopeKind;
    preparation.identity.scopeDigest = scope->scopeDigest;
    preparation.identity.mechanismKey = key;
    preparation.identity.invocationSite = *site;
    preparation.identity.cutPoint = cutPoint;
    preparation.identity.invocationOrdinal =
        scope->invocationOrdinalBase + invocationOrdinal;
    preparation.qualificationCase = scope->qualificationCase;
    preparation.specDigest = digestAdoptionSpecV1(state->spec);
    preparation.inputSnapshotDigest = inputSnapshotDigest;
    if (!state->recorder->beginInvocation(preparation, diagnostic))
      return false;
    state->preparation = preparation;
  }
  token.state_ = std::move(state);
  return true;
}

bool commitOptimizationInvocationV1(
    OptimizationInvocationTokenV1 &token,
    const OptimizationInvocationTelemetry &telemetry,
    std::string *diagnostic) {
  if (!token.begun()) {
    if (diagnostic)
      *diagnostic = "optimization invocation terminal has no begin token";
    return false;
  }
  auto state = std::static_pointer_cast<InvocationTokenState>(token.state_);
  if (state->committed) {
    if (diagnostic)
      *diagnostic = "optimization invocation token already committed";
    return false;
  }
  if (telemetry.key != state->key || telemetry.cutPoint != state->cutPoint ||
      telemetry.invocationOrdinal != state->invocationOrdinal ||
      telemetry.inputSnapshotDigest != state->inputSnapshotDigest) {
    if (diagnostic)
      *diagnostic = "optimization invocation terminal does not match begin";
    return false;
  }
  if (!validateEvidenceShape(state->descriptor, telemetry, diagnostic))
    return false;
  if (state->recorder) {
    InvocationTelemetryV1 terminal = buildTerminal(*state, telemetry);
    if (!state->recorder->recordInvocationTerminal(terminal, diagnostic))
      return false;
  }
  state->committed = true;
  return true;
}

} // namespace wafer
