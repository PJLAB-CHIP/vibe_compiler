//===- SystemCTargetModel.cpp - SystemC functional-event model ----------===//

#include "Wafer/Model/SystemC/SystemCTargetModel.h"
#include "Wafer/Model/TestSupport/Testing.h"

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"

#include "../Core/TargetModelTileCommandTracker.h"
#include "SystemCBridge.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer::model {
namespace {

llvm::Error systemCError(SystemCTargetModelErrorCode code,
                         const llvm::Twine &detail) {
  return llvm::make_error<SystemCTargetModelError>(code, detail.str());
}

struct InvocationFailure {
  SystemCTargetModelErrorCode code =
      SystemCTargetModelErrorCode::InvocationFailure;
  std::string stage;
  std::optional<int64_t> launchSlot;
  std::optional<uint64_t> issueOrdinal;
  std::string detail;

  std::string str() const {
    std::string result = "stage=" + stage;
    if (launchSlot)
      result += " launch_slot=" + std::to_string(*launchSlot);
    if (issueOrdinal)
      result += " issue=" + std::to_string(*issueOrdinal);
    result += ": " + detail;
    return result;
  }
};

class SystemCTargetModel final : public compiler::TargetCommandSink {
public:
  SystemCTargetModel(compiler::TargetCallExecutable executable,
                     InvocationMemoryRegistry memory,
                     TargetModelKernelBudget budget,
                     TargetModelExecutionPolicy policy,
                     std::optional<int64_t> completionFailureLaunchSlot)
      : executable(std::move(executable)), memory(std::move(memory)),
        budget(budget), policy(policy),
        completionFailureLaunchSlot(completionFailureLaunchSlot),
        tileStates(this->memory.getAddressPlan().getLaunchSlots().size()),
        dteStates(tileStates.size(), DTEState::NotBegun),
        tileCompletionEvents(tileStates.size(), nullptr) {
    for (detail::SystemCEvent *&event : tileCompletionEvents) {
      event = detail::createSystemCEvent();
      if (!event) {
        initializationDiagnostic = detail::getSystemCBridgeDiagnostic();
        return;
      }
    }
    runner = detail::createSystemCRunner(tileStates.size(), tileEntry, this);
    if (!runner)
      initializationDiagnostic = detail::getSystemCBridgeDiagnostic();
  }

  ~SystemCTargetModel() override {
    detail::destroySystemCRunner(runner);
    for (DTEEndpoint &endpoint : endpoints) {
      detail::destroySystemCEvent(endpoint.completionEvent);
      detail::destroySystemCEvent(endpoint.readinessEvent);
    }
    for (detail::SystemCEvent *event : tileCompletionEvents)
      detail::destroySystemCEvent(event);
  }

  llvm::Error start() {
    if (!initializationDiagnostic.empty())
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          initializationDiagnostic);
    return executable.begin(*this);
  }

  llvm::Expected<TargetModelResult> finish() {
    bool needsWakeDrain = false;
    const std::string bridgeFailure = detail::getSystemCBridgeDiagnostic();
    if (!bridgeFailure.empty() && !failure) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "systemc-bridge", std::nullopt, std::nullopt, bridgeFailure);
      needsWakeDrain = true;
    }
    if (!failure && completedLaunchSlots.size() != tileStates.size()) {
      std::string detail = describeNoProgress();
      latchFailure(SystemCTargetModelErrorCode::NoProgress, "no-progress",
                   std::nullopt, std::nullopt, detail);
      needsWakeDrain = true;
    }
    if (failure) {
      executable.abort(failure->str());
      if (needsWakeDrain)
        detail::startSystemCSimulation();
      return systemCError(failure->code, failure->str());
    }

    llvm::Expected<compiler::TargetCallExecutionResult> frontend =
        executable.finish();
    if (!frontend)
      return systemCError(SystemCTargetModelErrorCode::InvocationFailure,
                          llvm::toString(frontend.takeError()));
    if (!completedResult)
      return systemCError(
          SystemCTargetModelErrorCode::ResultInvariantViolation,
          "target-call execution finished without one model result");
    if (completedResult->completedTileCount != frontend->completedTileCount ||
        completedResult->issuedCommandCount != frontend->issuedCommandCount)
      return systemCError(
          SystemCTargetModelErrorCode::ResultInvariantViolation,
          "model result disagrees with target-call frontend counters");
    return std::move(*completedResult);
  }

  llvm::Error
  begin(const compiler::TargetCallInvocationDescriptor &invocation) override {
    if (begun)
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "sink begin was called more than once");
    if (invocation.targetIdentity !=
            memory.getAddressPlan().getTargetIdentity() ||
        invocation.tiles.size() != tileStates.size())
      return systemCError(
          SystemCTargetModelErrorCode::InvalidLifecycle,
          "sink invocation differs from the closed address plan");
    tileBindings.reserve(invocation.tiles.size());
    for (size_t index = 0; index < invocation.tiles.size(); ++index) {
      const compiler::TargetCallTileDescriptor &tile = invocation.tiles[index];
      if (tile.launchSlotId.getValue() != static_cast<int64_t>(index))
        return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                            "sink launch-slot order is not canonical");
      tileBindings.push_back({tile.cardId, tile.tileId, tile.launchSlotId});
    }
    begun = true;
    return llvm::Error::success();
  }

  llvm::Expected<uint64_t>
  issue(const compiler::TargetCommand &command) override {
    if (!begun || failure)
      return currentFailureOrLifecycle("command arrived before sink begin");
    const int64_t launchSlot = command.launchSlotId.getValue();
    if (launchSlot < 0 ||
        static_cast<uint64_t>(launchSlot) >= tileStates.size())
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "command launch slot is outside the invocation");
    const TileBinding &binding = tileBindings[static_cast<size_t>(launchSlot)];
    if (command.cardId != binding.cardId || command.tileId != binding.tileId)
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "command Tile binding disagrees with "
                          "its launch slot");
    TileState &tile = tileStates[static_cast<size_t>(launchSlot)];
    if (!tile.completion.beginIssue(command.issueOrdinal)) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "issue-order", launchSlot, command.issueOrdinal,
                   "Tile issue ordinal is not contiguous");
      return currentFailureOrLifecycle("issue order failure");
    }
    ++issuedCommandCount;

    // Every issue crosses at least one delta before field/address/numeric
    // effects can become visible. The calling JIT stack remains suspended.
    detail::waitSystemCDelta();
    if (bridgeFailed(command, "issue-delta") || failure)
      return currentFailureOrLifecycle("issue delta failed");

    llvm::Expected<TargetModelCommandEffect> effect =
        executeTargetModelCommand(command, memory, budget, policy);
    if (!effect) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "command-kernel", launchSlot, command.issueOrdinal,
                   llvm::toString(effect.takeError()));
      return currentFailureOrLifecycle("command kernel failed");
    }
    switch (effect->controlAction) {
    case TargetModelControlAction::None: {
      const TargetModelNumericBackend numericBackend = effect->numericBackend;
      TargetModelOneDNNDispatchEvidence onednnEvidence = effect->onednnEvidence;
      TargetModelManagedReferenceEvidence managedReferenceEvidence =
          effect->managedReferenceEvidence;
      std::optional<PendingNCCMemoryEffect> pendingMemoryEffect;
      if (command.nccIssueDomain &&
          command.nccIssueDomain->completionBehavior ==
              TargetNCCCompletionBehavior::OrderedAsynchronousIssue)
        pendingMemoryEffect = summarizePendingNCCMemoryEffect(*effect);
      if (llvm::Error error = applyTargetModelCommandEffect(
              memory, numericContext, std::move(*effect))) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "command-effect", launchSlot, command.issueOrdinal,
                     llvm::toString(std::move(error)));
        return currentFailureOrLifecycle("applying command effect failed");
      }
      if (numericBackend == TargetModelNumericBackend::Formal)
        ++formalNumericOperationCount;
      if (numericBackend == TargetModelNumericBackend::ManagedReference) {
        ++managedReferenceNumericOperationCount;
        managedReferenceScalarEvaluationCount +=
            managedReferenceEvidence.scalarEvaluations;
        if (!llvm::is_contained(managedReferenceTensorEnvironmentDigests,
                                managedReferenceEvidence.environmentDigest))
          managedReferenceTensorEnvironmentDigests.push_back(
              std::move(managedReferenceEvidence.environmentDigest));
        if (!llvm::is_contained(managedReferenceTensorImplementations,
                                managedReferenceEvidence.implementation))
          managedReferenceTensorImplementations.push_back(
              std::move(managedReferenceEvidence.implementation));
      }
      if (numericBackend == TargetModelNumericBackend::OneDNN) {
        ++onednnNumericOperationCount;
        onednnMatmulInvocationCount += onednnEvidence.matmulInvocations;
        onednnReorderInvocationCount += onednnEvidence.reorderInvocations;
        onednnFormalFusedMultiplyAddCount +=
            onednnEvidence.formalFusedMultiplyAdds;
        if (onednnEvidence.evidenceKind ==
            TargetModelOneDNNEvidenceKind::ExactQualificationRecord)
          onednnQualificationRecordDigests.push_back(
              std::move(onednnEvidence.evidenceDigest));
        else if (onednnEvidence.evidenceKind ==
                 TargetModelOneDNNEvidenceKind::ManagedReferenceEnvironment) {
          if (!llvm::is_contained(onednnManagedReferenceEnvironmentDigests,
                                  onednnEvidence.evidenceDigest))
            onednnManagedReferenceEnvironmentDigests.push_back(
                std::move(onednnEvidence.evidenceDigest));
        }
      }
      if (command.nccIssueDomain) {
        const compiler::TargetNCCIssueDomain &domain = *command.nccIssueDomain;
        if (domain.completionBehavior ==
            TargetNCCCompletionBehavior::SynchronousWriteback) {
          completeNCCParticipantPending(
              launchSlot, uint32_t{1} << static_cast<uint32_t>(domain.worker));
          if (failure)
            return currentFailureOrLifecycle(
                "synchronous NCC writeback failed");
          markOrdinalComplete(launchSlot, command.issueOrdinal);
          tryMatchReadyEndpoints();
        } else if (!tile.completion.addNCCPending(domain.worker,
                                                  command.issueOrdinal)) {
          latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                       "ncc-issue-domain", launchSlot, command.issueOrdinal,
                       "NCC issue worker is invalid or already pending");
          return currentFailureOrLifecycle("NCC issue-domain failure");
        } else if (!pendingMemoryEffect ||
                   !pendingNCCMemoryEffects
                        .try_emplace(
                            std::make_pair(launchSlot, command.issueOrdinal),
                            std::move(*pendingMemoryEffect))
                        .second) {
          latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                       "ncc-issue-domain", launchSlot, command.issueOrdinal,
                       "NCC pending memory effect is missing or duplicate");
          return currentFailureOrLifecycle("NCC issue-domain failure");
        }
      } else {
        markOrdinalComplete(launchSlot, command.issueOrdinal);
      }
      return UINT64_C(0);
    }
    case TargetModelControlAction::NCCJoin:
      return processNCCJoin(command);
    case TargetModelControlAction::DirectDTEBegin:
      return processDTEBegin(command);
    case TargetModelControlAction::DirectDTESendPrepare:
      return processDTESendPrepare(command);
    case TargetModelControlAction::DirectDTESendIssue:
      return processDTESendIssue(command);
    case TargetModelControlAction::DirectDTEReceive:
      return processDTEReceive(command);
    case TargetModelControlAction::DirectDTEWait:
      return processDTEWait(command);
    case TargetModelControlAction::DirectDTEFinish:
      return processDTEFinish(command);
    }
    llvm_unreachable("unknown target model control action");
  }

  llvm::Error completeTile(CardId cardId, TileId tileId,
                           LaunchSlotId launchSlotId) override {
    if (failure)
      return systemCError(failure->code, failure->str());
    const int64_t launchSlot = launchSlotId.getValue();
    if (launchSlot < 0 ||
        static_cast<uint64_t>(launchSlot) >= tileStates.size())
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "completed Tile is outside the invocation");
    const TileBinding &binding = tileBindings[static_cast<size_t>(launchSlot)];
    if (cardId != binding.cardId || tileId != binding.tileId ||
        launchSlotId != binding.launchSlotId)
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "completed Tile binding disagrees with its launch "
                          "slot");
    TileState &tile = tileStates[static_cast<size_t>(launchSlot)];
    if (tile.completed || tile.completion.getNextCompletedOrdinal() !=
                              tile.completion.getNextIssuedOrdinal())
      return systemCError(
          SystemCTargetModelErrorCode::InvocationFailure,
          "Tile completion is duplicate or precedes issued effects");
    if (dteStates[static_cast<size_t>(launchSlot)] == DTEState::Active)
      return systemCError(SystemCTargetModelErrorCode::InvocationFailure,
                          "Tile completed with active Direct DTE");
    if (completionFailureLaunchSlot &&
        launchSlot == *completionFailureLaunchSlot) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "injected-Tile-completion-failure", launchSlot, std::nullopt,
                   "test-only Tile completion failure injection");
      return systemCError(failure->code, failure->str());
    }
    tile.completed = true;
    completedLaunchSlots.insert(launchSlot);
    return llvm::Error::success();
  }

  llvm::Error completeInvocation() override {
    if (failure || completedLaunchSlots.size() != tileStates.size())
      return systemCError(SystemCTargetModelErrorCode::ResultInvariantViolation,
                          "invocation has a failure or incomplete Tile");
    for (const DTEEndpoint &endpoint : endpoints)
      if (!endpoint.complete || !endpoint.released)
        return systemCError(
            SystemCTargetModelErrorCode::ResultInvariantViolation,
            "invocation has an incomplete or unreleased Direct DTE endpoint");
    if (completedResult)
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "model result was completed twice");
    std::vector<TargetModelOutput> outputs;
    std::vector<TargetModelResourceId> outputResources;
    for (const TargetModelPlannedSlot &slot :
         memory.getAddressPlan().getSlots()) {
      if (slot.kind != compiler::TileEntryArgumentKind::ExternalOutput)
        continue;
      if (llvm::is_contained(outputResources, slot.resource))
        continue;
      llvm::Expected<std::vector<uint8_t>> bytes =
          memory.readSlotSnapshot(slot.launchSlot, slot.slotOrdinal);
      if (!bytes)
        return bytes.takeError();
      const TileBinding &binding =
          tileBindings[static_cast<size_t>(slot.launchSlot)];
      outputResources.push_back(slot.resource);
      outputs.push_back({slot.resource, binding.cardId, binding.tileId,
                         binding.launchSlotId, slot.slotOrdinal,
                         slot.resourceIndex, std::move(*bytes)});
    }
    completedResult.emplace(
        TargetModelResult{memory.getAddressPlan().getTargetIdentity(),
                          static_cast<int64_t>(completedLaunchSlots.size()),
                          issuedCommandCount,
                          detail::getSystemCThreadProcessCount(runner),
                          detail::getSystemCDeltaCount(),
                          numericContext.getAggregateFlags(),
                          formalNumericOperationCount,
                          managedReferenceNumericOperationCount,
                          managedReferenceScalarEvaluationCount,
                          onednnNumericOperationCount,
                          onednnMatmulInvocationCount,
                          onednnReorderInvocationCount,
                          onednnFormalFusedMultiplyAddCount,
                          std::move(onednnQualificationRecordDigests),
                          std::move(onednnManagedReferenceEnvironmentDigests),
                          std::move(managedReferenceTensorEnvironmentDigests),
                          std::move(managedReferenceTensorImplementations),
                          detail::getSystemCVersion(),
                          "untimed-delta-worker-aware-ncc",
                          std::move(outputs)});
    return llvm::Error::success();
  }

  void abort(llvm::StringRef diagnostic) override {
    latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                 "frontend-abort", std::nullopt, std::nullopt, diagnostic);
    completedResult.reset();
  }

private:
  struct TileBinding {
    CardId cardId;
    TileId tileId;
    LaunchSlotId launchSlotId;
  };

  struct TileState {
    detail::TargetModelTileCommandTracker completion;
    bool completed = false;
  };

  enum class DTEState : uint8_t { NotBegun, Active, Finished };
  enum class EndpointKind : uint8_t { Send, Receive };

  struct DTEEndpoint {
    uint64_t event = 0;
    EndpointKind kind = EndpointKind::Send;
    int64_t ownerLaunchSlot = -1;
    /// The endpoint-producing command whose asynchronous effect completes
    /// when the transfer becomes visible. A prepared send has no effect
    /// ordinal until its explicit issue command executes.
    std::optional<uint64_t> effectOrdinal;
    std::optional<target::TargetDirectDTESendCommand> send;
    std::optional<target::TargetDirectDTEReceiveCommand> receive;
    detail::SystemCEvent *completionEvent = nullptr;
    detail::SystemCEvent *readinessEvent = nullptr;
    bool peerReady = false;
    bool matched = false;
    bool complete = false;
    bool released = false;
  };

  struct PreparedDTESend {
    uint64_t event = 0;
    int64_t ownerLaunchSlot = -1;
    uint64_t prepareOrdinal = 0;
    target::TargetDirectDTESendCommand send;
    bool issued = false;
    bool released = false;
  };

  struct PendingMemoryInterval {
    int64_t launchSlot = -1;
    TargetModelAddressSpace addressSpace = TargetModelAddressSpace::TileSPM;
    uint64_t begin = 0;
    uint64_t end = 0;
  };

  struct PendingNCCMemoryEffect {
    llvm::SmallVector<PendingMemoryInterval, 2> reads;
    llvm::SmallVector<PendingMemoryInterval, 2> writes;
    bool hasUnknownRead = false;
    bool hasUnknownWrite = false;
  };

  static bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
      return false;
    result = lhs + rhs;
    return true;
  }

  static bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
      return false;
    result = lhs * rhs;
    return true;
  }

  static std::optional<uint64_t>
  getConservativeSpan(const TargetModelStridedByteLayout &layout) {
    if (layout.innerBytes == 0)
      return std::nullopt;
    uint64_t span = layout.innerBytes;
    for (size_t dimension = 0; dimension < layout.iterations.size();
         ++dimension) {
      if (layout.iterations[dimension] == 0)
        return std::nullopt;
      uint64_t dimensionSpan = 0;
      if (!checkedMultiply(layout.iterations[dimension] - 1,
                           layout.strides[dimension], dimensionSpan) ||
          !checkedAdd(span, dimensionSpan, span))
        return std::nullopt;
    }
    return span;
  }

  static bool appendMemoryInterval(
      llvm::SmallVectorImpl<PendingMemoryInterval> &intervals,
      int64_t launchSlot, TargetModelAddressSpace addressSpace,
      uint64_t address, uint64_t byteCount,
      const std::optional<TargetModelStridedByteLayout> &layout =
          std::nullopt) {
    if (layout) {
      std::optional<uint64_t> span = getConservativeSpan(*layout);
      if (!span)
        return false;
      byteCount = *span;
    }
    uint64_t end = 0;
    if (byteCount == 0 || !checkedAdd(address, byteCount, end))
      return false;
    intervals.push_back({launchSlot, addressSpace, address, end});
    return true;
  }

  static PendingNCCMemoryEffect
  summarizePendingNCCMemoryEffect(const TargetModelCommandEffect &effect) {
    PendingNCCMemoryEffect summary;
    for (const TargetModelByteRead &read : effect.pendingReads)
      if (!appendMemoryInterval(summary.reads, read.launchSlot,
                                read.addressSpace, read.address, read.byteCount,
                                read.stridedLayout))
        summary.hasUnknownRead = true;
    for (const TargetModelByteWrite &write : effect.pendingWrites)
      if (!appendMemoryInterval(summary.writes, write.launchSlot,
                                write.addressSpace, write.address,
                                write.bytes.size(), write.stridedLayout))
        summary.hasUnknownWrite = true;
    return summary;
  }

  static bool memoryIntervalOverlaps(const PendingMemoryInterval &pending,
                                     int64_t launchSlot,
                                     TargetModelAddressSpace addressSpace,
                                     uint64_t begin, uint64_t end) {
    return pending.launchSlot == launchSlot &&
           pending.addressSpace == addressSpace && pending.begin < end &&
           begin < pending.end;
  }

  static void tileEntry(void *owner, int64_t launchSlot) {
    static_cast<SystemCTargetModel *>(owner)->tileProcess(launchSlot);
  }

  void tileProcess(int64_t launchSlot) {
    if (!begun) {
      latchFailure(SystemCTargetModelErrorCode::InvalidLifecycle,
                   "Tile-process", launchSlot, std::nullopt,
                   "Tile process started before frontend begin");
      return;
    }
    if (llvm::Error error = executable.executeTile(LaunchSlotId(launchSlot))) {
      if (!failure)
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "Tile-execute", launchSlot, std::nullopt,
                     llvm::toString(std::move(error)));
      else
        llvm::consumeError(std::move(error));
    }
  }

  llvm::Expected<uint64_t>
  processNCCJoin(const compiler::TargetCommand &command) {
    const auto &join = std::get<target::TargetNCCJoinCommand>(command.payload);
    completeNCCParticipantPending(command.launchSlotId.getValue(),
                                  join.participantMask);
    if (failure)
      return currentFailureOrLifecycle("NCC join failed");
    markOrdinalComplete(command.launchSlotId.getValue(), command.issueOrdinal);
    tryMatchReadyEndpoints();
    return UINT64_C(0);
  }

  llvm::Expected<uint64_t>
  processDTEBegin(const compiler::TargetCommand &command) {
    const int64_t launchSlot = command.launchSlotId.getValue();
    DTEState &state = dteStates[static_cast<size_t>(launchSlot)];
    if (state != DTEState::NotBegun) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-begin",
                   launchSlot, command.issueOrdinal,
                   "Direct DTE begin is duplicate or follows finish");
      return currentFailureOrLifecycle("Direct DTE begin failed");
    }
    const auto &begin =
        std::get<target::TargetDirectDTEBeginCommand>(command.payload);
    llvm::Expected<TargetModelResolvedRange> status =
        memory.getAddressPlan().resolve(
            launchSlot, TargetModelAddressSpace::CardDDR,
            TargetModelAccess::ReadWrite, begin.statusAddress,
            WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES,
            WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES);
    if (!status || !status->slotOrdinal) {
      const std::string diagnostic = status
                                         ? "status address has no ABI slot"
                                         : llvm::toString(status.takeError());
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-begin",
                   launchSlot, command.issueOrdinal, diagnostic);
      return currentFailureOrLifecycle("Direct DTE begin failed");
    }
    const TargetModelPlannedSlot *statusSlot = nullptr;
    for (const TargetModelPlannedSlot &slot :
         memory.getAddressPlan().getSlots())
      if (slot.launchSlot == status->launchSlot &&
          slot.slotOrdinal == *status->slotOrdinal)
        statusSlot = &slot;
    if (!statusSlot ||
        statusSlot->kind != compiler::TileEntryArgumentKind::TransportStatus) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-begin",
                   launchSlot, command.issueOrdinal,
                   "Direct DTE status address is not the transport slot");
      return currentFailureOrLifecycle("Direct DTE begin failed");
    }
    state = DTEState::Active;
    markOrdinalComplete(launchSlot, command.issueOrdinal);
    return UINT64_C(0);
  }

  llvm::Expected<uint64_t>
  processDTESendPrepare(const compiler::TargetCommand &command) {
    const int64_t launchSlot = command.launchSlotId.getValue();
    if (dteStates[static_cast<size_t>(launchSlot)] != DTEState::Active) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "dte-send-prepare", launchSlot, command.issueOrdinal,
                   "Direct DTE send prepare occurred outside begin/finish");
      return currentFailureOrLifecycle("Direct DTE send prepare failed");
    }
    const auto &send =
        std::get<target::TargetDirectDTESendCommand>(command.payload);
    if (!validateCommandTile(command, send.localTile, "dte-send-prepare"))
      return currentFailureOrLifecycle("Direct DTE send prepare failed");
    for (const PreparedDTESend &prepared : preparedSends)
      if (prepared.ownerLaunchSlot == launchSlot && !prepared.released) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-send-prepare", launchSlot, command.issueOrdinal,
                     "Direct DTE sender already has a live prepared event");
        return currentFailureOrLifecycle("Direct DTE send prepare failed");
      }
    llvm::Expected<uint64_t> event = allocateEvent();
    if (!event)
      return event.takeError();
    preparedSends.push_back(
        {*event, launchSlot, command.issueOrdinal, send, false, false});
    markOrdinalComplete(launchSlot, command.issueOrdinal);
    return *event;
  }

  llvm::Expected<uint64_t>
  processDTESendIssue(const compiler::TargetCommand &command) {
    const int64_t launchSlot = command.launchSlotId.getValue();
    if (dteStates[static_cast<size_t>(launchSlot)] != DTEState::Active) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "dte-send-issue", launchSlot, command.issueOrdinal,
                   "Direct DTE send issue occurred outside begin/finish");
      return currentFailureOrLifecycle("Direct DTE send issue failed");
    }
    const auto &issue =
        std::get<target::TargetDirectDTESendIssueCommand>(command.payload);
    PreparedDTESend *prepared = findPreparedSend(issue.event);
    if (!prepared || prepared->ownerLaunchSlot != launchSlot ||
        prepared->issued || prepared->released) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "dte-send-issue", launchSlot, command.issueOrdinal,
                   "Direct DTE send issue names an unknown, foreign, or "
                   "already-issued prepared event");
      return currentFailureOrLifecycle("Direct DTE send issue failed");
    }
    llvm::Expected<DTEEndpoint *> endpoint = issuePreparedDTESend(
        *prepared, command.issueOrdinal, command, "dte-send-issue");
    if (!endpoint)
      return endpoint.takeError();
    return UINT64_C(0);
  }

  llvm::Expected<DTEEndpoint *> issuePreparedDTESend(
      PreparedDTESend &prepared, std::optional<uint64_t> effectOrdinal,
      const compiler::TargetCommand &command, llvm::StringRef stage) {
    detail::SystemCEvent *completion = detail::createSystemCEvent();
    if (!completion) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, stage,
                   command.launchSlotId.getValue(), command.issueOrdinal,
                   detail::getSystemCBridgeDiagnostic());
      return currentFailureOrLifecycle("Direct DTE send event creation failed");
    }
    detail::SystemCEvent *readiness = detail::createSystemCEvent();
    if (!readiness) {
      detail::destroySystemCEvent(completion);
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, stage,
                   command.launchSlotId.getValue(), command.issueOrdinal,
                   detail::getSystemCBridgeDiagnostic());
      return currentFailureOrLifecycle(
          "Direct DTE send readiness event creation failed");
    }
    endpoints.push_back({prepared.event, EndpointKind::Send,
                         prepared.ownerLaunchSlot, effectOrdinal, prepared.send,
                         std::nullopt, completion, readiness, false, false,
                         false, false});
    prepared.issued = true;
    DTEEndpoint &endpoint = endpoints.back();
    tryMatchEndpoint(endpoints.size() - 1);
    while (!endpoint.peerReady && !failure) {
      detail::waitSystemCEvent(endpoint.readinessEvent);
      if (bridgeFailed(command, stage))
        break;
    }
    if (failure)
      return currentFailureOrLifecycle("Direct DTE match failed");
    return &endpoint;
  }

  llvm::Expected<uint64_t>
  processDTEReceive(const compiler::TargetCommand &command) {
    const int64_t launchSlot = command.launchSlotId.getValue();
    if (dteStates[static_cast<size_t>(launchSlot)] != DTEState::Active) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "dte-receive", launchSlot, command.issueOrdinal,
                   "Direct DTE receive occurred outside begin/finish");
      return currentFailureOrLifecycle("Direct DTE receive failed");
    }
    const auto &receive =
        std::get<target::TargetDirectDTEReceiveCommand>(command.payload);
    if (!validateCommandTile(command, receive.localTile, "dte-receive"))
      return currentFailureOrLifecycle("Direct DTE receive failed");
    llvm::Expected<uint64_t> event = allocateEvent();
    if (!event)
      return event.takeError();
    detail::SystemCEvent *completion = detail::createSystemCEvent();
    if (!completion) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "dte-receive", launchSlot, command.issueOrdinal,
                   detail::getSystemCBridgeDiagnostic());
      return currentFailureOrLifecycle("Direct DTE event creation failed");
    }
    endpoints.push_back({*event, EndpointKind::Receive, launchSlot,
                         command.issueOrdinal, std::nullopt, receive,
                         completion, nullptr, true, false, false, false});
    tryMatchEndpoint(endpoints.size() - 1);
    if (failure)
      return currentFailureOrLifecycle("Direct DTE match failed");
    return *event;
  }

  llvm::Expected<uint64_t>
  processDTEWait(const compiler::TargetCommand &command) {
    const auto &wait =
        std::get<target::TargetDirectDTEWaitCommand>(command.payload);
    PreparedDTESend *prepared = findPreparedSend(wait.event);
    const int64_t launchSlot = command.launchSlotId.getValue();
    if (prepared && prepared->ownerLaunchSlot == launchSlot &&
        !prepared->issued) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-wait",
                   launchSlot, command.issueOrdinal,
                   "Direct DTE wait names a prepared send that was not issued");
      return currentFailureOrLifecycle("Direct DTE wait failed");
    }
    DTEEndpoint *endpoint = findEndpoint(wait.event);
    if (!endpoint || endpoint->ownerLaunchSlot != launchSlot ||
        endpoint->released) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-wait",
                   launchSlot, command.issueOrdinal,
                   "Direct DTE wait names an unknown, foreign, or already "
                   "released event");
      return currentFailureOrLifecycle("Direct DTE wait failed");
    }
    while (!endpoint->complete && !failure) {
      detail::waitSystemCEvent(endpoint->completionEvent);
      if (bridgeFailed(command, "dte-wait"))
        break;
    }
    if (failure)
      return currentFailureOrLifecycle("Direct DTE wait failed");
    endpoint->released = true;
    if (prepared)
      prepared->released = true;
    markOrdinalComplete(launchSlot, command.issueOrdinal);
    return UINT64_C(0);
  }

  llvm::Expected<uint64_t>
  processDTEFinish(const compiler::TargetCommand &command) {
    const int64_t launchSlot = command.launchSlotId.getValue();
    DTEState &state = dteStates[static_cast<size_t>(launchSlot)];
    if (state != DTEState::Active) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-finish",
                   launchSlot, command.issueOrdinal,
                   "Direct DTE finish occurred outside active state");
      return currentFailureOrLifecycle("Direct DTE finish failed");
    }
    for (const PreparedDTESend &prepared : preparedSends)
      if (prepared.ownerLaunchSlot == launchSlot && !prepared.released) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-finish", launchSlot, command.issueOrdinal,
                     "Direct DTE finish precedes exact send wait/release");
        return currentFailureOrLifecycle("Direct DTE finish failed");
      }
    for (const DTEEndpoint &endpoint : endpoints)
      if (endpoint.ownerLaunchSlot == launchSlot && !endpoint.released) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-finish", launchSlot, command.issueOrdinal,
                     "Direct DTE finish precedes exact endpoint wait/release");
        return currentFailureOrLifecycle("Direct DTE finish failed");
      }
    state = DTEState::Finished;
    markOrdinalComplete(launchSlot, command.issueOrdinal);
    return UINT64_C(0);
  }

  llvm::Expected<uint64_t> allocateEvent() {
    if (nextEvent == std::numeric_limits<uint64_t>::max()) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "event-allocation", std::nullopt, std::nullopt,
                   "opaque Direct DTE event domain is exhausted");
      return currentFailureOrLifecycle("event allocation failed");
    }
    return nextEvent++;
  }

  DTEEndpoint *findEndpoint(uint64_t event) {
    for (DTEEndpoint &endpoint : endpoints)
      if (endpoint.event == event)
        return &endpoint;
    return nullptr;
  }

  PreparedDTESend *findPreparedSend(uint64_t event) {
    for (PreparedDTESend &prepared : preparedSends)
      if (prepared.event == event)
        return &prepared;
    return nullptr;
  }

  bool validateCommandTile(const compiler::TargetCommand &command,
                           uint32_t payloadTileId, llvm::StringRef stage) {
    const int64_t tileId = command.tileId.getValue();
    if (tileId < 0 ||
        static_cast<uint64_t>(tileId) > std::numeric_limits<uint32_t>::max() ||
        payloadTileId != static_cast<uint32_t>(tileId)) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, stage,
                   command.launchSlotId.getValue(), command.issueOrdinal,
                   "Direct DTE payload disagrees with the explicit physical "
                   "Tile binding");
      return false;
    }
    return true;
  }

  bool sameEndpointIdentity(const DTEEndpoint &lhs,
                            const DTEEndpoint &rhs) const {
    if (lhs.kind != rhs.kind)
      return false;
    if (lhs.kind == EndpointKind::Send) {
      const auto &left = *lhs.send;
      const auto &right = *rhs.send;
      return left.localTile == right.localTile &&
             left.remoteTile == right.remoteTile &&
             left.remoteFSM == right.remoteFSM;
    }
    const auto &left = *lhs.receive;
    const auto &right = *rhs.receive;
    return left.localTile == right.localTile &&
           left.remoteTile == right.remoteTile &&
           left.localFSM == right.localFSM;
  }

  bool hasPendingNCCConflict(int64_t launchSlot, uint64_t address,
                             uint64_t byteCount, bool dteWrites) const {
    uint64_t end = 0;
    if (byteCount == 0 || !checkedAdd(address, byteCount, end))
      return true;
    for (const auto &[key, effect] : pendingNCCMemoryEffects) {
      if (key.first != launchSlot)
        continue;
      if (effect.hasUnknownWrite ||
          llvm::any_of(effect.writes, [&](const PendingMemoryInterval &range) {
            return memoryIntervalOverlaps(range, launchSlot,
                                          TargetModelAddressSpace::TileSPM,
                                          address, end);
          }))
        return true;
      if (dteWrites &&
          (effect.hasUnknownRead ||
           llvm::any_of(effect.reads, [&](const PendingMemoryInterval &range) {
             return memoryIntervalOverlaps(range, launchSlot,
                                           TargetModelAddressSpace::TileSPM,
                                           address, end);
           })))
        return true;
    }
    return false;
  }

  void tryMatchEndpoint(size_t endpointIndex) {
    if (failure || endpointIndex >= endpoints.size())
      return;
    DTEEndpoint &endpoint = endpoints[endpointIndex];
    for (size_t otherIndex = 0; otherIndex < endpoints.size(); ++otherIndex) {
      if (otherIndex == endpointIndex)
        continue;
      DTEEndpoint &other = endpoints[otherIndex];
      if (other.kind == endpoint.kind) {
        if (!other.complete && sameEndpointIdentity(endpoint, other)) {
          latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                       "dte-match", endpoint.ownerLaunchSlot,
                       endpoint.effectOrdinal,
                       "duplicate live Direct DTE endpoint binding");
          return;
        }
        continue;
      }
      if (other.matched)
        continue;
      DTEEndpoint &sendEndpoint =
          endpoint.kind == EndpointKind::Send ? endpoint : other;
      DTEEndpoint &receiveEndpoint =
          endpoint.kind == EndpointKind::Receive ? endpoint : other;
      const auto &send = *sendEndpoint.send;
      const auto &receive = *receiveEndpoint.receive;
      const bool keyMatches = send.localTile == receive.remoteTile &&
                              send.remoteTile == receive.localTile &&
                              send.remoteFSM == receive.localFSM;
      if (!keyMatches)
        continue;
      if (send.byteCount != receive.byteCount ||
          send.remoteDestination != receive.destination) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-match", sendEndpoint.ownerLaunchSlot,
                     sendEndpoint.effectOrdinal,
                     "matched endpoints disagree on bytes or destination");
        return;
      }
      // A receive prepare establishes the receiver endpoint; a send endpoint
      // only exists after the matching explicit send issue. NCC writes become
      // visible in the shared functional memory at issue time, but Direct DTE
      // remains an external observer. The ordering proof must therefore hold
      // before peer-ready is signaled and before send_async could be
      // submitted. A participant join after issue cannot retroactively order
      // the transfer.
      if (hasPendingNCCConflict(sendEndpoint.ownerLaunchSlot, send.source,
                                send.byteCount, /*dteWrites=*/false)) {
        latchFailure(
            SystemCTargetModelErrorCode::InvocationFailure, "dte-issue-order",
            sendEndpoint.ownerLaunchSlot, sendEndpoint.effectOrdinal,
            "Direct DTE issue overlaps a pending NCC source write; matching "
            "participant join must precede the Direct DTE issue");
        return;
      }
      if (hasPendingNCCConflict(receiveEndpoint.ownerLaunchSlot,
                                receive.destination, receive.byteCount,
                                /*dteWrites=*/true)) {
        latchFailure(
            SystemCTargetModelErrorCode::InvocationFailure, "dte-issue-order",
            receiveEndpoint.ownerLaunchSlot, receiveEndpoint.effectOrdinal,
            "Direct DTE issue overlaps a pending NCC destination read/write; "
            "matching participant join must precede the Direct DTE issue");
        return;
      }
      // The real CRT's send-issue call does not return until the matching
      // receiver prepare has made its endpoint available. Signal readiness
      // only after the issue-time ordering checks have succeeded.
      if (!sendEndpoint.peerReady) {
        sendEndpoint.peerReady = true;
        detail::notifySystemCEvent(sendEndpoint.readinessEvent);
      }
      llvm::Expected<std::vector<uint8_t>> source = memory.readSnapshot(
          sendEndpoint.ownerLaunchSlot, TargetModelAddressSpace::TileSPM,
          send.source, send.byteCount, 1);
      if (!source) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-source-read", sendEndpoint.ownerLaunchSlot,
                     sendEndpoint.effectOrdinal,
                     llvm::toString(source.takeError()));
        return;
      }
      if (llvm::Error error = memory.applyAtomically({TargetModelByteWrite{
              receiveEndpoint.ownerLaunchSlot, TargetModelAddressSpace::TileSPM,
              receive.destination, 1, std::move(*source)}})) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-visibility", receiveEndpoint.ownerLaunchSlot,
                     receiveEndpoint.effectOrdinal,
                     llvm::toString(std::move(error)));
        return;
      }
      sendEndpoint.matched = sendEndpoint.complete = true;
      receiveEndpoint.matched = receiveEndpoint.complete = true;
      if (sendEndpoint.effectOrdinal)
        markOrdinalComplete(sendEndpoint.ownerLaunchSlot,
                            *sendEndpoint.effectOrdinal);
      if (receiveEndpoint.effectOrdinal)
        markOrdinalComplete(receiveEndpoint.ownerLaunchSlot,
                            *receiveEndpoint.effectOrdinal);
      detail::notifySystemCEvent(sendEndpoint.completionEvent);
      detail::notifySystemCEvent(receiveEndpoint.completionEvent);
      return;
    }
  }

  void tryMatchReadyEndpoints() {
    for (size_t endpointIndex = 0; endpointIndex < endpoints.size();
         ++endpointIndex)
      if (!endpoints[endpointIndex].matched)
        tryMatchEndpoint(endpointIndex);
  }

  void completeNCCParticipantPending(int64_t launchSlot,
                                     uint32_t participantMask) {
    TileState &tile = tileStates[static_cast<size_t>(launchSlot)];
    for (uint64_t ordinal :
         tile.completion.takeNCCParticipantPending(participantMask)) {
      if (pendingNCCMemoryEffects.erase(std::make_pair(launchSlot, ordinal)) !=
          1) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "ncc-completion", launchSlot, ordinal,
                     "NCC pending memory effect is missing");
        return;
      }
      markOrdinalComplete(launchSlot, ordinal);
    }
  }

  std::string describeNoProgress() const {
    std::string detail = "simulation starved before all Tiles completed";
    size_t unresolved = 0;
    for (const PreparedDTESend &prepared : preparedSends)
      if (!prepared.issued && !prepared.released)
        ++unresolved;
    for (const DTEEndpoint &endpoint : endpoints)
      if (!endpoint.complete)
        ++unresolved;
    if (unresolved == 0)
      return detail;
    detail += " with unresolved Direct DTE/event state:";
    for (const PreparedDTESend &prepared : preparedSends)
      if (!prepared.issued && !prepared.released)
        detail += " prepared-send(launch_slot=" +
                  std::to_string(prepared.ownerLaunchSlot) + ")";
    for (const DTEEndpoint &endpoint : endpoints) {
      if (endpoint.complete)
        continue;
      detail += endpoint.kind == EndpointKind::Send ? " send" : " receive";
      detail += "(launch_slot=" + std::to_string(endpoint.ownerLaunchSlot);
      if (endpoint.kind == EndpointKind::Send) {
        const auto &send = *endpoint.send;
        detail += ",local_tile=" + std::to_string(send.localTile) +
                  ",remote_tile=" + std::to_string(send.remoteTile) +
                  ",fsm=" + std::to_string(send.remoteFSM);
      } else {
        const auto &receive = *endpoint.receive;
        detail += ",local_tile=" + std::to_string(receive.localTile) +
                  ",remote_tile=" + std::to_string(receive.remoteTile) +
                  ",fsm=" + std::to_string(receive.localFSM);
      }
      detail += ")";
    }
    return detail;
  }

  void markOrdinalComplete(int64_t launchSlot, uint64_t issueOrdinal) {
    if (failure)
      return;
    TileState &tile = tileStates[static_cast<size_t>(launchSlot)];
    if (!tile.completion.markComplete(issueOrdinal)) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "completion-order", launchSlot, issueOrdinal,
                   "command completion was recorded twice");
      return;
    }
    detail::notifySystemCEvent(
        tileCompletionEvents[static_cast<size_t>(launchSlot)]);
  }

  bool bridgeFailed(const compiler::TargetCommand &command,
                    llvm::StringRef stage) {
    const std::string diagnostic = detail::getSystemCBridgeDiagnostic();
    if (diagnostic.empty())
      return false;
    latchFailure(SystemCTargetModelErrorCode::InvocationFailure, stage,
                 command.launchSlotId.getValue(), command.issueOrdinal,
                 diagnostic);
    return true;
  }

  llvm::Error currentFailureOrLifecycle(llvm::StringRef fallback) const {
    if (failure)
      return systemCError(failure->code, failure->str());
    return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                        fallback);
  }

  void latchFailure(SystemCTargetModelErrorCode code, llvm::StringRef stage,
                    std::optional<int64_t> launchSlot,
                    std::optional<uint64_t> issueOrdinal,
                    llvm::StringRef detailText) {
    if (failure)
      return;
    failure = InvocationFailure{code, stage.str(), launchSlot, issueOrdinal,
                                detailText.str()};
    for (detail::SystemCEvent *event : tileCompletionEvents)
      detail::notifySystemCEvent(event);
    for (DTEEndpoint &endpoint : endpoints)
      detail::notifySystemCEvent(endpoint.completionEvent);
    for (DTEEndpoint &endpoint : endpoints)
      detail::notifySystemCEvent(endpoint.readinessEvent);
  }

  compiler::TargetCallExecutable executable;
  InvocationMemoryRegistry memory;
  TargetModelKernelBudget budget;
  TargetModelExecutionPolicy policy;
  std::optional<int64_t> completionFailureLaunchSlot;
  FormalNumericExecutionContext numericContext;
  std::vector<TileBinding> tileBindings;
  std::vector<TileState> tileStates;
  std::vector<DTEState> dteStates;
  std::vector<detail::SystemCEvent *> tileCompletionEvents;
  detail::SystemCRunner *runner = nullptr;
  // Endpoint pointers remain live across SystemC wait(). A deque preserves
  // those references while other Tile processes append their endpoints.
  std::deque<PreparedDTESend> preparedSends;
  std::deque<DTEEndpoint> endpoints;
  std::map<std::pair<int64_t, uint64_t>, PendingNCCMemoryEffect>
      pendingNCCMemoryEffects;
  std::set<int64_t> completedLaunchSlots;
  std::optional<InvocationFailure> failure;
  std::optional<TargetModelResult> completedResult;
  std::string initializationDiagnostic;
  uint64_t nextEvent = 1;
  uint64_t issuedCommandCount = 0;
  uint64_t formalNumericOperationCount = 0;
  uint64_t managedReferenceNumericOperationCount = 0;
  uint64_t managedReferenceScalarEvaluationCount = 0;
  uint64_t onednnNumericOperationCount = 0;
  uint64_t onednnMatmulInvocationCount = 0;
  uint64_t onednnReorderInvocationCount = 0;
  uint64_t onednnFormalFusedMultiplyAddCount = 0;
  std::vector<std::string> onednnQualificationRecordDigests;
  std::vector<std::string> onednnManagedReferenceEnvironmentDigests;
  std::vector<std::string> managedReferenceTensorEnvironmentDigests;
  std::vector<std::string> managedReferenceTensorImplementations;
  bool begun = false;
};

} // namespace

llvm::StringRef
stringifySystemCTargetModelErrorCode(SystemCTargetModelErrorCode code) {
  switch (code) {
  case SystemCTargetModelErrorCode::InvalidLifecycle:
    return "invalid-lifecycle";
  case SystemCTargetModelErrorCode::InvocationFailure:
    return "invocation-failure";
  case SystemCTargetModelErrorCode::NoProgress:
    return "no-progress";
  case SystemCTargetModelErrorCode::ResultInvariantViolation:
    return "result-invariant-violation";
  }
  llvm_unreachable("unknown SystemC target model error code");
}

char SystemCTargetModelError::ID;

void SystemCTargetModelError::log(llvm::raw_ostream &stream) const {
  stream << "SystemC target model "
         << stringifySystemCTargetModelErrorCode(code) << ": " << detail;
}

std::error_code SystemCTargetModelError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

static llvm::Expected<TargetModelResult> executeSystemCTargetModelImpl(
    compiler::TargetCallExecutable executable,
    llvm::ArrayRef<TargetModelInputBinding> inputBindings,
    TargetModelKernelBudget budget, TargetModelExecutionPolicy policy,
    std::optional<int64_t> completionFailureLaunchSlot) {
  if (!detail::isSystemCInitialElaboration())
    return systemCError(
        SystemCTargetModelErrorCode::InvalidLifecycle,
        "SystemC model invocation must begin during initial elaboration");
  llvm::Expected<InvocationAddressPlan> plan = InvocationAddressPlan::create(
      executable.getInvocationDescriptor(), inputBindings);
  if (!plan)
    return plan.takeError();
  if (completionFailureLaunchSlot &&
      !llvm::is_contained(plan->getLaunchSlots(), *completionFailureLaunchSlot))
    return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                        "completion failure launch slot is outside the "
                        "invocation");
  llvm::Expected<InvocationMemoryRegistry> memory =
      InvocationMemoryRegistry::create(std::move(*plan));
  if (!memory)
    return memory.takeError();

  SystemCTargetModel model(std::move(executable), std::move(*memory), budget,
                           policy, completionFailureLaunchSlot);
  if (llvm::Error error = model.start())
    return std::move(error);
  detail::startSystemCSimulation();
  return model.finish();
}

llvm::Expected<TargetModelResult>
executeSystemCTargetModel(compiler::TargetCallExecutable executable,
                          llvm::ArrayRef<TargetModelInputBinding> inputBindings,
                          TargetModelKernelBudget budget,
                          TargetModelExecutionPolicy policy) {
  return executeSystemCTargetModelImpl(std::move(executable), inputBindings,
                                       budget, policy, std::nullopt);
}

namespace testing {

llvm::Expected<TargetModelResult>
executeSystemCTargetModelWithTileCompletionFailure(
    compiler::TargetCallExecutable executable,
    llvm::ArrayRef<TargetModelInputBinding> inputBindings,
    TargetModelKernelBudget budget, TargetModelExecutionPolicy policy,
    int64_t failureLaunchSlot) {
  return executeSystemCTargetModelImpl(std::move(executable), inputBindings,
                                       budget, policy, failureLaunchSlot);
}

} // namespace testing

} // namespace wafer::model
