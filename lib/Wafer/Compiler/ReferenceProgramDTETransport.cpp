//===- ReferenceProgramDTETransport.cpp - Deterministic transport -----===//

#include "ReferenceProgramInterpreterInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::reference_detail {

char TransportBlockedError::ID;

namespace {

class DeterministicTransportCoordinator final : public TransportCoordinator {
public:
  void beginReplay(int64_t logicalRank) {
    currentRank = logicalRank;
    replaySends.clear();
    replayRecvs.clear();
  }

  llvm::Error issue(const TransportEventKey &key, bool isSend,
                    uint64_t byteCount, int64_t remoteReceiverOffset,
                    const BufferView &buffer) override {
    if ((isSend ? key.source : key.destination) != currentRank)
      return invalid("Direct DTE issue rank disagrees with event endpoint");
    std::set<TransportEventKey> &seen = isSend ? replaySends : replayRecvs;
    if (!seen.insert(key).second)
      return invalid(isSend ? "duplicate Direct DTE send instance"
                            : "duplicate Direct DTE recv instance");
    if (byteCount == 0 ||
        byteCount > static_cast<uint64_t>(buffer.physicalBytes))
      return invalid("Direct DTE byte count exceeds projected buffer");
    if (buffer.base < 0 || buffer.viewOffset < 0 ||
        buffer.base > std::numeric_limits<int64_t>::max() - buffer.viewOffset)
      return invalid("Direct DTE buffer address overflows");
    const int64_t absolute = buffer.base + buffer.viewOffset;
    if (absolute > static_cast<int64_t>(buffer.storage->bytes.size()) ||
        byteCount >
            buffer.storage->bytes.size() - static_cast<size_t>(absolute))
      return invalid("Direct DTE buffer exceeds rank-local storage");

    EventState &state = events[key];
    if (isSend) {
      std::vector<uint8_t> payload(byteCount);
      std::memcpy(payload.data(), buffer.storage->bytes.data() + absolute,
                  byteCount);
      if (!state.sendIssued) {
        state.sendIssued = true;
        state.sendBytes = byteCount;
        state.senderRemoteOffset = remoteReceiverOffset;
        state.payload = std::move(payload);
        madeProgress = true;
      } else if (state.sendBytes != byteCount ||
                 state.senderRemoteOffset != remoteReceiverOffset ||
                 state.payload != payload) {
        return invalid("replayed Direct DTE send is not deterministic");
      }
      return llvm::Error::success();
    }

    if (absolute != remoteReceiverOffset)
      return invalid(
          "Direct DTE recv address disagrees with accepted remote offset");
    if (!state.recvIssued) {
      state.recvIssued = true;
      state.recvBytes = byteCount;
      state.receiverOffset = absolute;
      madeProgress = true;
    } else if (state.recvBytes != byteCount ||
               state.receiverOffset != absolute) {
      return invalid("replayed Direct DTE recv is not deterministic");
    }
    if (state.transferred) {
      if (state.payload.size() != byteCount)
        return invalid("completed Direct DTE payload size is inconsistent");
      std::memcpy(buffer.storage->bytes.data() + absolute, state.payload.data(),
                  byteCount);
    }
    return llvm::Error::success();
  }

  bool isComplete(const TransportEventKey &key) const override {
    auto found = events.find(key);
    return found != events.end() && found->second.transferred;
  }

  llvm::Error matchReadyEvents() {
    for (auto &[key, state] : events) {
      if (state.transferred || !state.sendIssued || !state.recvIssued)
        continue;
      if (state.sendBytes != state.recvBytes)
        return invalid("matched Direct DTE endpoints disagree on byte count");
      if (state.senderRemoteOffset != state.receiverOffset)
        return invalid(
            "matched Direct DTE endpoints disagree on receiver address");
      state.transferred = true;
      madeProgress = true;
    }
    return llvm::Error::success();
  }

  bool takeProgress() {
    bool result = madeProgress;
    madeProgress = false;
    return result;
  }

  void describePending(llvm::raw_ostream &stream) const {
    for (const auto &[key, state] : events) {
      if (state.transferred)
        continue;
      stream << " [" << key.source << "->" << key.destination
             << " communication=" << key.communication << " phase=" << key.phase
             << " round=" << key.round << " slice=" << key.payloadSlice
             << " control=[";
      llvm::interleaveComma(key.controlInstance, stream);
      stream << "]"
             << " send=" << (state.sendIssued ? "issued" : "missing")
             << " recv=" << (state.recvIssued ? "issued" : "missing") << "]";
    }
  }

private:
  struct EventState {
    bool sendIssued = false;
    bool recvIssued = false;
    bool transferred = false;
    uint64_t sendBytes = 0;
    uint64_t recvBytes = 0;
    int64_t senderRemoteOffset = -1;
    int64_t receiverOffset = -1;
    std::vector<uint8_t> payload;
  };

  int64_t currentRank = -1;
  bool madeProgress = false;
  std::set<TransportEventKey> replaySends;
  std::set<TransportEventKey> replayRecvs;
  std::map<TransportEventKey, EventState> events;
};

llvm::Expected<std::vector<ReferenceGlobalOutputBinding>>
reassembleOutputs(llvm::ArrayRef<const ReferenceProgram::Impl *> programs,
                  llvm::ArrayRef<ReferenceExecutionResult> rankResults) {
  if (programs.empty() || programs.size() != rankResults.size())
    return invalid("multi-rank output domain is incomplete");

  std::vector<const RankProgramBinding *> firstOutputs;
  for (const RankProgramBinding &binding : programs.front()->programBindings)
    if (binding.role == ProgramResourceRole::Output)
      firstOutputs.push_back(&binding);
  llvm::sort(firstOutputs, [](const RankProgramBinding *left,
                              const RankProgramBinding *right) {
    return left->programIndex < right->programIndex;
  });
  for (const ReferenceProgram::Impl *program : programs) {
    size_t outputCount = llvm::count_if(
        program->programBindings, [](const RankProgramBinding &binding) {
          return binding.role == ProgramResourceRole::Output;
        });
    if (outputCount != firstOutputs.size())
      return invalid("rank output binding domain disagrees across bundle");
  }

  std::vector<ReferenceGlobalOutputBinding> globals;
  for (const RankProgramBinding *first : firstOutputs) {
    std::vector<const RankProgramBinding *> bindings;
    std::vector<const ReferenceTensor *> tensors;
    for (size_t rank = 0; rank < programs.size(); ++rank) {
      const RankProgramBinding *binding = nullptr;
      for (const RankProgramBinding &candidate :
           programs[rank]->programBindings)
        if (candidate.role == ProgramResourceRole::Output &&
            candidate.programIndex == first->programIndex) {
          if (binding)
            return invalid("rank contains duplicate output binding index");
          binding = &candidate;
        }
      if (!binding || binding->name != first->name ||
          binding->dtype != first->dtype ||
          binding->globalShape != first->globalShape ||
          binding->distribution != first->distribution)
        return invalid("rank output metadata disagrees across bundle");
      const ReferenceTensor *tensor = nullptr;
      for (const ReferenceOutputBinding &output :
           rankResults[rank].getOutputs())
        if (output.index == first->programIndex) {
          if (tensor)
            return invalid("rank result contains duplicate output index");
          tensor = &output.tensor;
        }
      if (!tensor || tensor->getDType() != binding->dtype ||
          tensor->getShape() != llvm::ArrayRef<int64_t>(binding->localShape))
        return invalid("rank result disagrees with typed output slice");
      bindings.push_back(binding);
      tensors.push_back(tensor);
    }

    if (first->distribution == frontend::ProgramDistributionKind::Replicated) {
      if (first->localShape != first->globalShape)
        return invalid("replicated output does not cover global shape");
      bool requireExactReplicas =
          programs.front()->transportContract == TransportContract::None ||
          first->dtype != "f32";
      for (size_t rankIndex = 1;
           requireExactReplicas && rankIndex < tensors.size(); ++rankIndex) {
        llvm::ArrayRef<uint8_t> canonical = tensors.front()->getBytes();
        llvm::ArrayRef<uint8_t> candidate = tensors[rankIndex]->getBytes();
        if (candidate == canonical)
          continue;
        auto mismatch = std::mismatch(canonical.begin(), canonical.end(),
                                      candidate.begin(), candidate.end());
        size_t byteOffset = static_cast<size_t>(
            std::distance(canonical.begin(), mismatch.first));
        std::string message;
        llvm::raw_string_ostream stream(message);
        stream << "replicated output index " << first->programIndex << " rank "
               << programs[rankIndex]->logicalRank
               << " differs from rank 0 at byte " << byteOffset;
        if (first->dtype == "f32" && byteOffset / 4 < canonical.size() / 4) {
          size_t element = byteOffset / 4;
          float rankZero = 0.0f;
          float rankValue = 0.0f;
          std::memcpy(&rankZero, canonical.data() + element * 4, 4);
          std::memcpy(&rankValue, candidate.data() + element * 4, 4);
          stream << " (element " << element << ": rank0=" << rankZero
                 << ", rank=" << rankValue << ")";
        }
        stream.flush();
        return invalid(message);
      }
      auto global = ReferenceTensor::create(first->dtype, first->globalShape,
                                            tensors.front()->getBytes());
      if (!global)
        return global.takeError();
      globals.push_back({first->programIndex, first->name, std::move(*global)});
      continue;
    }

    auto elementBytes = getCompactByteCount(first->dtype, {1});
    auto globalBytes = getCompactByteCount(first->dtype, first->globalShape);
    if (!elementBytes || !globalBytes || *elementBytes <= 0)
      return invalid("partitioned output has unsupported tensor type");
    const int64_t globalElements = *globalBytes / *elementBytes;
    std::vector<uint8_t> bytes(*globalBytes, 0);
    std::vector<uint8_t> coverage(globalElements, 0);
    for (size_t rankIndex = 0; rankIndex < bindings.size(); ++rankIndex) {
      const RankProgramBinding *binding = bindings[rankIndex];
      const ReferenceTensor *tensor = tensors[rankIndex];
      const auto &slice = binding->slice;
      const size_t rank = first->globalShape.size();
      if (slice.logicalRank != programs[rankIndex]->logicalRank ||
          binding->localShape != slice.sizes || slice.offsets.size() != rank ||
          slice.sizes.size() != rank || slice.strides.size() != rank)
        return invalid("partitioned output slice has inconsistent rank");
      int64_t localLinear = 0;
      if (llvm::Error error = forEachLogicalIndex(
              binding->localShape,
              [&](llvm::ArrayRef<int64_t> local) -> llvm::Error {
                int64_t globalLinear = 0;
                for (size_t dim = 0; dim < rank; ++dim) {
                  if (slice.offsets[dim] < 0 || slice.strides[dim] <= 0 ||
                      local[dim] < 0 ||
                      local[dim] > (std::numeric_limits<int64_t>::max() -
                                    slice.offsets[dim]) /
                                       slice.strides[dim])
                    return invalid("partitioned output slice overflows");
                  int64_t coordinate =
                      slice.offsets[dim] + local[dim] * slice.strides[dim];
                  if (coordinate < 0 || coordinate >= first->globalShape[dim])
                    return invalid("partitioned output slice is out of bounds");
                  globalLinear =
                      globalLinear * first->globalShape[dim] + coordinate;
                }
                if (coverage[globalLinear]++)
                  return invalid("partitioned output slices overlap");
                std::memcpy(bytes.data() + globalLinear * *elementBytes,
                            tensor->getBytes().data() +
                                localLinear * *elementBytes,
                            *elementBytes);
                ++localLinear;
                return llvm::Error::success();
              }))
        return std::move(error);
    }
    if (llvm::is_contained(coverage, uint8_t{0}))
      return invalid("partitioned output slices do not cover global tensor");
    auto global =
        ReferenceTensor::create(first->dtype, first->globalShape, bytes);
    if (!global)
      return global.takeError();
    globals.push_back({first->programIndex, first->name, std::move(*global)});
  }
  return globals;
}

} // namespace

llvm::Expected<ReferenceMultiRankExecutionResult>
interpretReferenceProgramsWithDTE(
    llvm::ArrayRef<const ReferenceProgram::Impl *> programs,
    llvm::ArrayRef<ReferenceRankInvocation> invocations,
    ReferenceExecutionOptions options) {
  if (programs.size() <= 1 || programs.size() != invocations.size())
    return invalid("multi-rank invocation domain is incomplete");
  std::vector<const ReferenceRankInvocation *> byRank(programs.size(), nullptr);
  for (const ReferenceRankInvocation &invocation : invocations) {
    if (invocation.logicalRank < 0 ||
        invocation.logicalRank >= static_cast<int64_t>(programs.size()) ||
        byRank[invocation.logicalRank])
      return invalid("multi-rank invocation domain is not all-and-only");
    byRank[invocation.logicalRank] = &invocation;
  }
  for (auto [rank, program] : llvm::enumerate(programs)) {
    if (!program || program->logicalRank != static_cast<int64_t>(rank) ||
        program->transportContract != programs.front()->transportContract)
      return invalid("projected multi-rank domain is not canonical");
  }

  DeterministicTransportCoordinator coordinator;
  std::vector<std::optional<ReferenceExecutionResult>> completed(
      programs.size());
  std::vector<std::string> blocked(programs.size());
  size_t remaining = programs.size();
  while (remaining != 0) {
    coordinator.takeProgress();
    bool roundProgress = false;
    for (size_t rank = 0; rank < programs.size(); ++rank) {
      if (completed[rank])
        continue;
      coordinator.beginReplay(static_cast<int64_t>(rank));
      ProgramInterpreter interpreter(*programs[rank], options, &coordinator);
      auto attempt = interpreter.runUntilBlocked(byRank[rank]->inputs);
      if (!attempt)
        return attempt.takeError();
      blocked[rank] = attempt->blockedReason;
      if (attempt->result) {
        completed[rank] = std::move(*attempt->result);
        --remaining;
        roundProgress = true;
      }
    }
    if (llvm::Error error = coordinator.matchReadyEvents())
      return std::move(error);
    roundProgress |= coordinator.takeProgress();
    if (remaining != 0 && !roundProgress) {
      std::string message;
      llvm::raw_string_ostream stream(message);
      stream << "deterministic Direct DTE no-progress/deadlock:";
      for (size_t rank = 0; rank < blocked.size(); ++rank)
        if (!completed[rank])
          stream << " {rank=" << rank << " reason=" << blocked[rank] << "}";
      stream << " pending=";
      coordinator.describePending(stream);
      return invalid(stream.str());
    }
  }

  std::vector<ReferenceExecutionResult> rankResults;
  rankResults.reserve(completed.size());
  for (std::optional<ReferenceExecutionResult> &result : completed)
    rankResults.push_back(std::move(*result));
  auto globals = reassembleOutputs(programs, rankResults);
  if (!globals)
    return globals.takeError();
  return ReferenceMultiRankExecutionResultBuilder::make(std::move(rankResults),
                                                        std::move(*globals));
}

} // namespace wafer::compiler::reference_detail
