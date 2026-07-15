//===- SystemCTargetModel.cpp - SystemC functional-event model ----------===//

#include "Wafer/Model/SystemCTargetModel.h"

#include "SystemCBridge.h"

#include "llvm/ADT/STLExtras.h"
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
  std::optional<int64_t> logicalRank;
  std::optional<uint64_t> issueOrdinal;
  std::string detail;

  std::string str() const {
    std::string result = "stage=" + stage;
    if (logicalRank)
      result += " rank=" + std::to_string(*logicalRank);
    if (issueOrdinal)
      result += " issue=" + std::to_string(*issueOrdinal);
    result += ": " + detail;
    return result;
  }
};

class SystemCTargetModel final : public compiler::TargetTransactionSink {
public:
  SystemCTargetModel(compiler::TargetCallExecutable executable,
                     InvocationMemoryRegistry memory,
                     TargetModelKernelBudget budget,
                     TargetModelExecutionPolicy policy)
      : executable(std::move(executable)), memory(std::move(memory)),
        budget(budget), policy(policy),
        rankStates(this->memory.getAddressPlan().getLogicalRanks().size()),
        dteStates(rankStates.size(), DTEState::NotBegun),
        rankPhysicalTiles(rankStates.size()),
        rankCompletionEvents(rankStates.size(), nullptr) {
    for (detail::SystemCEvent *&event : rankCompletionEvents) {
      event = detail::createSystemCEvent();
      if (!event) {
        initializationDiagnostic = detail::getSystemCBridgeDiagnostic();
        return;
      }
    }
    runner = detail::createSystemCRunner(rankStates.size(), rankEntry, this);
    if (!runner)
      initializationDiagnostic = detail::getSystemCBridgeDiagnostic();
  }

  ~SystemCTargetModel() override {
    detail::destroySystemCRunner(runner);
    for (DTEEndpoint &endpoint : endpoints)
      detail::destroySystemCEvent(endpoint.completionEvent);
    for (detail::SystemCEvent *event : rankCompletionEvents)
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
    if (!failure && terminalRanks.size() != rankStates.size()) {
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
        executable.commit();
    if (!frontend)
      return systemCError(SystemCTargetModelErrorCode::InvocationFailure,
                          llvm::toString(frontend.takeError()));
    if (!publishedResult)
      return systemCError(
          SystemCTargetModelErrorCode::ResultInvariantViolation,
          "frontend commit returned without one published model result");
    if (publishedResult->completedRankCount != frontend->completedRankCount ||
        publishedResult->issuedTransactionCount !=
            frontend->issuedTransactionCount)
      return systemCError(
          SystemCTargetModelErrorCode::ResultInvariantViolation,
          "model result disagrees with target-call frontend counters");
    return std::move(*publishedResult);
  }

  llvm::Error
  begin(const compiler::TargetCallInvocationDescriptor &invocation) override {
    if (begun)
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "sink begin was called more than once");
    if (invocation.targetProfile !=
            memory.getAddressPlan().getTargetProfile() ||
        invocation.ranks.size() != rankStates.size())
      return systemCError(
          SystemCTargetModelErrorCode::InvalidLifecycle,
          "sink invocation differs from the closed address plan");
    begun = true;
    return llvm::Error::success();
  }

  llvm::Expected<uint64_t>
  issue(const compiler::TargetTransaction &transaction) override {
    if (!begun || failure)
      return currentFailureOrLifecycle("transaction arrived before sink begin");
    if (transaction.logicalRank < 0 ||
        static_cast<uint64_t>(transaction.logicalRank) >= rankStates.size())
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "transaction rank is outside the invocation");
    RankState &rank = rankStates[static_cast<size_t>(transaction.logicalRank)];
    if (transaction.issueOrdinal != rank.nextIssuedOrdinal) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "issue-order", transaction.logicalRank,
                   transaction.issueOrdinal,
                   "rank issue ordinal is not contiguous");
      return currentFailureOrLifecycle("issue order failure");
    }
    ++rank.nextIssuedOrdinal;
    ++issuedTransactionCount;

    // Every issue crosses at least one delta before field/address/numeric
    // effects can become visible. The calling JIT stack remains suspended.
    detail::waitSystemCDelta();
    if (bridgeFailed(transaction, "issue-delta") || failure)
      return currentFailureOrLifecycle("issue delta failed");

    llvm::Expected<TargetModelCommandEffect> effect =
        executeTargetModelCommand(transaction, memory, budget, policy);
    if (!effect) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "command-kernel", transaction.logicalRank,
                   transaction.issueOrdinal,
                   llvm::toString(effect.takeError()));
      return currentFailureOrLifecycle("command kernel failed");
    }
    switch (effect->controlAction) {
    case TargetModelControlAction::None: {
      const TargetModelNumericBackend numericBackend = effect->numericBackend;
      TargetModelBulkDispatchEvidence bulkEvidence = effect->bulkEvidence;
      if (llvm::Error error = commitTargetModelCommandEffect(
              memory, numericContext, std::move(*effect))) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "command-commit", transaction.logicalRank,
                     transaction.issueOrdinal,
                     llvm::toString(std::move(error)));
        return currentFailureOrLifecycle("command commit failed");
      }
      if (numericBackend == TargetModelNumericBackend::Formal)
        ++formalNumericCommandCount;
      if (numericBackend == TargetModelNumericBackend::Bulk) {
        ++bulkNumericCommandCount;
        bulkMatmulInvocationCount += bulkEvidence.matmulInvocations;
        bulkReorderInvocationCount += bulkEvidence.reorderInvocations;
        bulkFormalFusedMultiplyAddCount += bulkEvidence.formalFusedMultiplyAdds;
        bulkAdmissionRecordDigests.push_back(
            std::move(bulkEvidence.admissionRecordDigest));
      }
      markOrdinalComplete(transaction.logicalRank, transaction.issueOrdinal);
      return UINT64_C(0);
    }
    case TargetModelControlAction::LocalFence:
      return processFence(transaction);
    case TargetModelControlAction::DirectDTEBegin:
      return processDTEBegin(transaction);
    case TargetModelControlAction::DirectDTESend:
      return processDTESend(transaction);
    case TargetModelControlAction::DirectDTEReceive:
      return processDTEReceive(transaction);
    case TargetModelControlAction::DirectDTEWait:
      return processDTEWait(transaction);
    case TargetModelControlAction::DirectDTEFinish:
      return processDTEFinish(transaction);
    }
    llvm_unreachable("unknown target model control action");
  }

  llvm::Error terminal(int64_t logicalRank) override {
    if (failure)
      return systemCError(failure->code, failure->str());
    if (logicalRank < 0 ||
        static_cast<uint64_t>(logicalRank) >= rankStates.size())
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "terminal rank is outside the invocation");
    RankState &rank = rankStates[static_cast<size_t>(logicalRank)];
    if (rank.terminal || rank.nextCompletedOrdinal != rank.nextIssuedOrdinal)
      return systemCError(
          SystemCTargetModelErrorCode::InvocationFailure,
          "rank terminal is duplicate or precedes issued effects");
    if (dteStates[static_cast<size_t>(logicalRank)] == DTEState::Active)
      return systemCError(SystemCTargetModelErrorCode::InvocationFailure,
                          "rank reached terminal with active Direct DTE");
    rank.terminal = true;
    terminalRanks.insert(logicalRank);
    return llvm::Error::success();
  }

  llvm::Error prepareCommit() override {
    if (failure || terminalRanks.size() != rankStates.size())
      return systemCError(SystemCTargetModelErrorCode::ResultInvariantViolation,
                          "prepareCommit saw a failure or nonterminal rank");
    for (const DTEEndpoint &endpoint : endpoints)
      if (!endpoint.complete)
        return systemCError(
            SystemCTargetModelErrorCode::ResultInvariantViolation,
            "prepareCommit saw an incomplete Direct DTE endpoint");
    if (stagedResult || publishedResult)
      return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                          "model result was prepared or committed twice");
    std::vector<TargetModelOutput> outputs;
    for (const TargetModelPlannedSlot &slot :
         memory.getAddressPlan().getSlots()) {
      if (slot.role != compiler::KernelABISlotRole::Output)
        continue;
      llvm::Expected<std::vector<uint8_t>> bytes =
          memory.readSlotSnapshot(slot.logicalRank, slot.slotOrdinal);
      if (!bytes)
        return bytes.takeError();
      outputs.push_back({slot.logicalRank, slot.slotOrdinal, slot.resourceIndex,
                         std::move(*bytes)});
    }
    stagedResult.emplace(TargetModelResult{
        memory.getAddressPlan().getTargetProfile(),
        ModelProfileId::formalDeterministicV1(),
        static_cast<int64_t>(terminalRanks.size()), issuedTransactionCount,
        detail::getSystemCThreadProcessCount(runner),
        detail::getSystemCDeltaCount(), numericContext.getAggregateFlags(),
        formalNumericCommandCount, bulkNumericCommandCount,
        bulkMatmulInvocationCount, bulkReorderInvocationCount,
        bulkFormalFusedMultiplyAddCount, std::move(bulkAdmissionRecordDigests),
        detail::getSystemCVersion(), "untimed-delta-single-issue-domain-v1",
        std::move(outputs)});
    return llvm::Error::success();
  }

  void commit() override {
    assert(stagedResult && !publishedResult &&
           "prepareCommit must stage exactly one result");
    publishedResult = std::move(stagedResult);
  }

  void abort(llvm::StringRef diagnostic) override {
    latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                 "frontend-abort", std::nullopt, std::nullopt, diagnostic);
    stagedResult.reset();
    publishedResult.reset();
  }

private:
  struct RankState {
    uint64_t nextIssuedOrdinal = 0;
    uint64_t nextCompletedOrdinal = 0;
    std::set<uint64_t> completedOutOfOrder;
    bool terminal = false;
  };

  enum class DTEState : uint8_t { NotBegun, Active, Finished };
  enum class EndpointKind : uint8_t { Send, Receive };

  struct DTEEndpoint {
    uint64_t event = 0;
    EndpointKind kind = EndpointKind::Send;
    int64_t ownerRank = -1;
    uint64_t issueOrdinal = 0;
    std::optional<compiler::TargetDirectDTESendTransaction> send;
    std::optional<compiler::TargetDirectDTEReceiveTransaction> receive;
    detail::SystemCEvent *completionEvent = nullptr;
    bool matched = false;
    bool complete = false;
  };

  static void rankEntry(void *owner, int64_t logicalRank) {
    static_cast<SystemCTargetModel *>(owner)->rankProcess(logicalRank);
  }

  void rankProcess(int64_t logicalRank) {
    if (!begun) {
      latchFailure(SystemCTargetModelErrorCode::InvalidLifecycle,
                   "rank-process", logicalRank, std::nullopt,
                   "rank process started before frontend begin");
      return;
    }
    if (llvm::Error error = executable.executeRank(logicalRank)) {
      if (!failure)
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "rank-execute", logicalRank, std::nullopt,
                     llvm::toString(std::move(error)));
      else
        llvm::consumeError(std::move(error));
    }
  }

  llvm::Expected<uint64_t>
  processFence(const compiler::TargetTransaction &transaction) {
    RankState &rank = rankStates[static_cast<size_t>(transaction.logicalRank)];
    while (rank.nextCompletedOrdinal < transaction.issueOrdinal && !failure) {
      detail::waitSystemCEvent(
          rankCompletionEvents[static_cast<size_t>(transaction.logicalRank)]);
      if (bridgeFailed(transaction, "fence-wait"))
        break;
    }
    if (failure)
      return currentFailureOrLifecycle("local fence failed");
    markOrdinalComplete(transaction.logicalRank, transaction.issueOrdinal);
    return UINT64_C(0);
  }

  llvm::Expected<uint64_t>
  processDTEBegin(const compiler::TargetTransaction &transaction) {
    const int64_t rank = transaction.logicalRank;
    DTEState &state = dteStates[static_cast<size_t>(rank)];
    if (state != DTEState::NotBegun) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-begin",
                   rank, transaction.issueOrdinal,
                   "Direct DTE begin is duplicate or follows finish");
      return currentFailureOrLifecycle("Direct DTE begin failed");
    }
    const auto &begin = std::get<compiler::TargetDirectDTEBeginTransaction>(
        transaction.payload);
    llvm::Expected<TargetModelResolvedRange> status =
        memory.getAddressPlan().resolve(rank, TargetModelAddressSpace::CardDDR,
                                        TargetModelAccess::ReadWrite,
                                        begin.statusAddress, 4, 4);
    if (!status || !status->slotOrdinal) {
      const std::string diagnostic = status
                                         ? "status address has no ABI slot"
                                         : llvm::toString(status.takeError());
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-begin",
                   rank, transaction.issueOrdinal, diagnostic);
      return currentFailureOrLifecycle("Direct DTE begin failed");
    }
    const TargetModelPlannedSlot *statusSlot = nullptr;
    for (const TargetModelPlannedSlot &slot :
         memory.getAddressPlan().getSlots())
      if (slot.logicalRank == status->logicalRank &&
          slot.slotOrdinal == *status->slotOrdinal)
        statusSlot = &slot;
    if (!statusSlot ||
        statusSlot->role != compiler::KernelABISlotRole::TransportStatus) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-begin",
                   rank, transaction.issueOrdinal,
                   "Direct DTE status address is not the transport slot");
      return currentFailureOrLifecycle("Direct DTE begin failed");
    }
    state = DTEState::Active;
    markOrdinalComplete(rank, transaction.issueOrdinal);
    return UINT64_C(0);
  }

  llvm::Expected<uint64_t>
  processDTESend(const compiler::TargetTransaction &transaction) {
    const int64_t rank = transaction.logicalRank;
    if (dteStates[static_cast<size_t>(rank)] != DTEState::Active) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-send",
                   rank, transaction.issueOrdinal,
                   "Direct DTE send occurred outside begin/finish");
      return currentFailureOrLifecycle("Direct DTE send failed");
    }
    const auto &send =
        std::get<compiler::TargetDirectDTESendTransaction>(transaction.payload);
    if (!bindRankTile(rank, send.localTile, "dte-send",
                      transaction.issueOrdinal))
      return currentFailureOrLifecycle("Direct DTE send failed");
    llvm::Expected<uint64_t> event = allocateEvent();
    if (!event)
      return event.takeError();
    detail::SystemCEvent *completion = detail::createSystemCEvent();
    if (!completion) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-send",
                   rank, transaction.issueOrdinal,
                   detail::getSystemCBridgeDiagnostic());
      return currentFailureOrLifecycle("Direct DTE event creation failed");
    }
    endpoints.push_back({*event, EndpointKind::Send, rank,
                         transaction.issueOrdinal, send, std::nullopt,
                         completion, false, false});
    tryMatchEndpoint(endpoints.size() - 1);
    if (failure)
      return currentFailureOrLifecycle("Direct DTE match failed");
    return *event;
  }

  llvm::Expected<uint64_t>
  processDTEReceive(const compiler::TargetTransaction &transaction) {
    const int64_t rank = transaction.logicalRank;
    if (dteStates[static_cast<size_t>(rank)] != DTEState::Active) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "dte-receive", rank, transaction.issueOrdinal,
                   "Direct DTE receive occurred outside begin/finish");
      return currentFailureOrLifecycle("Direct DTE receive failed");
    }
    const auto &receive = std::get<compiler::TargetDirectDTEReceiveTransaction>(
        transaction.payload);
    if (!bindRankTile(rank, receive.localTile, "dte-receive",
                      transaction.issueOrdinal))
      return currentFailureOrLifecycle("Direct DTE receive failed");
    llvm::Expected<uint64_t> event = allocateEvent();
    if (!event)
      return event.takeError();
    detail::SystemCEvent *completion = detail::createSystemCEvent();
    if (!completion) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "dte-receive", rank, transaction.issueOrdinal,
                   detail::getSystemCBridgeDiagnostic());
      return currentFailureOrLifecycle("Direct DTE event creation failed");
    }
    endpoints.push_back({*event, EndpointKind::Receive, rank,
                         transaction.issueOrdinal, std::nullopt, receive,
                         completion, false, false});
    tryMatchEndpoint(endpoints.size() - 1);
    if (failure)
      return currentFailureOrLifecycle("Direct DTE match failed");
    return *event;
  }

  llvm::Expected<uint64_t>
  processDTEWait(const compiler::TargetTransaction &transaction) {
    const auto &wait =
        std::get<compiler::TargetDirectDTEWaitTransaction>(transaction.payload);
    DTEEndpoint *endpoint = findEndpoint(wait.event);
    if (!endpoint || endpoint->ownerRank != transaction.logicalRank) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-wait",
                   transaction.logicalRank, transaction.issueOrdinal,
                   "Direct DTE wait names an unknown or foreign event");
      return currentFailureOrLifecycle("Direct DTE wait failed");
    }
    while (!endpoint->complete && !failure) {
      detail::waitSystemCEvent(endpoint->completionEvent);
      if (bridgeFailed(transaction, "dte-wait"))
        break;
    }
    if (failure)
      return currentFailureOrLifecycle("Direct DTE wait failed");
    markOrdinalComplete(transaction.logicalRank, transaction.issueOrdinal);
    return UINT64_C(0);
  }

  llvm::Expected<uint64_t>
  processDTEFinish(const compiler::TargetTransaction &transaction) {
    const int64_t rank = transaction.logicalRank;
    DTEState &state = dteStates[static_cast<size_t>(rank)];
    if (state != DTEState::Active) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, "dte-finish",
                   rank, transaction.issueOrdinal,
                   "Direct DTE finish occurred outside active state");
      return currentFailureOrLifecycle("Direct DTE finish failed");
    }
    for (const DTEEndpoint &endpoint : endpoints)
      if (endpoint.ownerRank == rank && !endpoint.complete) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-finish", rank, transaction.issueOrdinal,
                     "Direct DTE finish precedes endpoint completion");
        return currentFailureOrLifecycle("Direct DTE finish failed");
      }
    state = DTEState::Finished;
    markOrdinalComplete(rank, transaction.issueOrdinal);
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

  bool bindRankTile(int64_t logicalRank, uint32_t physicalTile,
                    llvm::StringRef stage, uint64_t issueOrdinal) {
    std::optional<uint32_t> &rankTile =
        rankPhysicalTiles[static_cast<size_t>(logicalRank)];
    if (rankTile && *rankTile != physicalTile) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, stage,
                   logicalRank, issueOrdinal,
                   "rank changed its Direct DTE physical tile identity");
      return false;
    }
    auto [owner, inserted] = tileOwners.emplace(physicalTile, logicalRank);
    if (!inserted && owner->second != logicalRank) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure, stage,
                   logicalRank, issueOrdinal,
                   "Direct DTE physical tile is owned by another rank");
      return false;
    }
    rankTile = physicalTile;
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
                       "dte-match", endpoint.ownerRank, endpoint.issueOrdinal,
                       "duplicate live Direct DTE endpoint identity");
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
                     "dte-match", sendEndpoint.ownerRank,
                     sendEndpoint.issueOrdinal,
                     "matched endpoints disagree on bytes or destination");
        return;
      }
      llvm::Expected<std::vector<uint8_t>> source = memory.readSnapshot(
          sendEndpoint.ownerRank, TargetModelAddressSpace::RankSPM, send.source,
          send.byteCount, 1);
      if (!source) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-source-read", sendEndpoint.ownerRank,
                     sendEndpoint.issueOrdinal,
                     llvm::toString(source.takeError()));
        return;
      }
      if (llvm::Error error = memory.applyAtomically({TargetModelByteWrite{
              receiveEndpoint.ownerRank, TargetModelAddressSpace::RankSPM,
              receive.destination, 1, std::move(*source)}})) {
        latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                     "dte-visibility", receiveEndpoint.ownerRank,
                     receiveEndpoint.issueOrdinal,
                     llvm::toString(std::move(error)));
        return;
      }
      sendEndpoint.matched = sendEndpoint.complete = true;
      receiveEndpoint.matched = receiveEndpoint.complete = true;
      markOrdinalComplete(sendEndpoint.ownerRank, sendEndpoint.issueOrdinal);
      markOrdinalComplete(receiveEndpoint.ownerRank,
                          receiveEndpoint.issueOrdinal);
      detail::notifySystemCEvent(sendEndpoint.completionEvent);
      detail::notifySystemCEvent(receiveEndpoint.completionEvent);
      return;
    }
  }

  std::string describeNoProgress() const {
    std::string detail = "simulation starved before all ranks were terminal";
    size_t unresolved = 0;
    for (const DTEEndpoint &endpoint : endpoints)
      if (!endpoint.complete)
        ++unresolved;
    if (unresolved == 0)
      return detail;
    detail += " with unresolved Direct DTE/event state:";
    for (const DTEEndpoint &endpoint : endpoints) {
      if (endpoint.complete)
        continue;
      detail += endpoint.kind == EndpointKind::Send ? " send" : " receive";
      detail += "(rank=" + std::to_string(endpoint.ownerRank);
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

  void markOrdinalComplete(int64_t logicalRank, uint64_t issueOrdinal) {
    if (failure)
      return;
    RankState &rank = rankStates[static_cast<size_t>(logicalRank)];
    if (issueOrdinal < rank.nextCompletedOrdinal ||
        !rank.completedOutOfOrder.insert(issueOrdinal).second) {
      latchFailure(SystemCTargetModelErrorCode::InvocationFailure,
                   "completion-order", logicalRank, issueOrdinal,
                   "transaction completion was published twice");
      return;
    }
    while (rank.completedOutOfOrder.erase(rank.nextCompletedOrdinal))
      ++rank.nextCompletedOrdinal;
    detail::notifySystemCEvent(
        rankCompletionEvents[static_cast<size_t>(logicalRank)]);
  }

  bool bridgeFailed(const compiler::TargetTransaction &transaction,
                    llvm::StringRef stage) {
    const std::string diagnostic = detail::getSystemCBridgeDiagnostic();
    if (diagnostic.empty())
      return false;
    latchFailure(SystemCTargetModelErrorCode::InvocationFailure, stage,
                 transaction.logicalRank, transaction.issueOrdinal, diagnostic);
    return true;
  }

  llvm::Error currentFailureOrLifecycle(llvm::StringRef fallback) const {
    if (failure)
      return systemCError(failure->code, failure->str());
    return systemCError(SystemCTargetModelErrorCode::InvalidLifecycle,
                        fallback);
  }

  void latchFailure(SystemCTargetModelErrorCode code, llvm::StringRef stage,
                    std::optional<int64_t> logicalRank,
                    std::optional<uint64_t> issueOrdinal,
                    llvm::StringRef detailText) {
    if (failure)
      return;
    failure = InvocationFailure{code, stage.str(), logicalRank, issueOrdinal,
                                detailText.str()};
    for (detail::SystemCEvent *event : rankCompletionEvents)
      detail::notifySystemCEvent(event);
    for (DTEEndpoint &endpoint : endpoints)
      detail::notifySystemCEvent(endpoint.completionEvent);
  }

  compiler::TargetCallExecutable executable;
  InvocationMemoryRegistry memory;
  TargetModelKernelBudget budget;
  TargetModelExecutionPolicy policy;
  FormalNumericExecutionContext numericContext;
  std::vector<RankState> rankStates;
  std::vector<DTEState> dteStates;
  std::vector<std::optional<uint32_t>> rankPhysicalTiles;
  std::map<uint32_t, int64_t> tileOwners;
  std::vector<detail::SystemCEvent *> rankCompletionEvents;
  detail::SystemCRunner *runner = nullptr;
  // Endpoint pointers remain live across SystemC wait(). A deque preserves
  // those references while other rank processes append their endpoints.
  std::deque<DTEEndpoint> endpoints;
  std::set<int64_t> terminalRanks;
  std::optional<InvocationFailure> failure;
  std::optional<TargetModelResult> stagedResult;
  std::optional<TargetModelResult> publishedResult;
  std::string initializationDiagnostic;
  uint64_t nextEvent = 1;
  uint64_t issuedTransactionCount = 0;
  uint64_t formalNumericCommandCount = 0;
  uint64_t bulkNumericCommandCount = 0;
  uint64_t bulkMatmulInvocationCount = 0;
  uint64_t bulkReorderInvocationCount = 0;
  uint64_t bulkFormalFusedMultiplyAddCount = 0;
  std::vector<std::string> bulkAdmissionRecordDigests;
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

llvm::Expected<TargetModelResult>
executeSystemCTargetModel(compiler::TargetCallExecutable executable,
                          llvm::ArrayRef<TargetModelInputBinding> inputBindings,
                          TargetModelKernelBudget budget,
                          TargetModelExecutionPolicy policy) {
  if (!detail::isSystemCInitialElaboration())
    return systemCError(
        SystemCTargetModelErrorCode::InvalidLifecycle,
        "SystemC model invocation must begin during initial elaboration");
  llvm::Expected<InvocationAddressPlan> plan = InvocationAddressPlan::create(
      executable.getInvocationDescriptor(), inputBindings);
  if (!plan)
    return plan.takeError();
  llvm::Expected<InvocationMemoryRegistry> memory =
      InvocationMemoryRegistry::create(std::move(*plan));
  if (!memory)
    return memory.takeError();

  SystemCTargetModel model(std::move(executable), std::move(*memory), budget,
                           policy);
  if (llvm::Error error = model.start())
    return std::move(error);
  detail::startSystemCSimulation();
  return model.finish();
}

} // namespace wafer::model
