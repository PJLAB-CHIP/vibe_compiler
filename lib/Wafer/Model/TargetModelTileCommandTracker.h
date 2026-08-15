//===- TargetModelTileCommandTracker.h - Track Tile commands -*- C++ -*-===//

#ifndef WAFER_MODEL_TARGETMODELTILECOMMANDTRACKER_H
#define WAFER_MODEL_TARGETMODELTILECOMMANDTRACKER_H

#include "Wafer/Target/TargetOperation.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>

namespace wafer::model::detail {

/// Tile-local command tracker shared by the SystemC scheduler
/// and its focused tests. Direct-DTE ordinals are deliberately absent from
/// the worker sets and can advance the prefix only through their own event
/// completion path.
class TargetModelTileCommandTracker {
public:
  bool beginIssue(uint64_t issueOrdinal) {
    if (issueOrdinal != nextIssuedOrdinal)
      return false;
    ++nextIssuedOrdinal;
    return true;
  }

  bool markComplete(uint64_t issueOrdinal) {
    if (issueOrdinal < nextCompletedOrdinal ||
        !completedOutOfOrder.insert(issueOrdinal).second)
      return false;
    while (completedOutOfOrder.erase(nextCompletedOrdinal))
      ++nextCompletedOrdinal;
    return true;
  }

  bool addNCCPending(TargetNCCWorker worker, uint64_t issueOrdinal) {
    const uint32_t workerOrdinal = static_cast<uint32_t>(worker);
    return workerOrdinal < kTargetNCCWorkerCount &&
           pendingNCCOrdinals[workerOrdinal].insert(issueOrdinal).second;
  }

  llvm::SmallVector<uint64_t, 8>
  takeNCCParticipantPending(uint32_t participantMask) {
    llvm::SmallVector<uint64_t, 8> pending;
    for (uint32_t worker = 0; worker < kTargetNCCWorkerCount; ++worker) {
      if ((participantMask & (uint32_t{1} << worker)) == 0)
        continue;
      pending.append(pendingNCCOrdinals[worker].begin(),
                     pendingNCCOrdinals[worker].end());
      pendingNCCOrdinals[worker].clear();
    }
    return pending;
  }

  uint64_t getNextIssuedOrdinal() const { return nextIssuedOrdinal; }
  uint64_t getNextCompletedOrdinal() const { return nextCompletedOrdinal; }

  size_t getPendingNCCCount(TargetNCCWorker worker) const {
    const uint32_t workerOrdinal = static_cast<uint32_t>(worker);
    return workerOrdinal < kTargetNCCWorkerCount
               ? pendingNCCOrdinals[workerOrdinal].size()
               : 0;
  }

  bool
  hasNCCPending(uint32_t participantMask = kAllTargetNCCWorkersMask) const {
    for (uint32_t worker = 0; worker < kTargetNCCWorkerCount; ++worker)
      if ((participantMask & (uint32_t{1} << worker)) != 0 &&
          !pendingNCCOrdinals[worker].empty())
        return true;
    return false;
  }

private:
  uint64_t nextIssuedOrdinal = 0;
  uint64_t nextCompletedOrdinal = 0;
  std::set<uint64_t> completedOutOfOrder;
  std::array<std::set<uint64_t>, kTargetNCCWorkerCount> pendingNCCOrdinals;
};

} // namespace wafer::model::detail

#endif // WAFER_MODEL_TARGETMODELTILECOMMANDTRACKER_H
