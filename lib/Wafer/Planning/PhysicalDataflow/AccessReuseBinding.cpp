//===- AccessReuseBinding.cpp - Bind choices to current read scopes -------===//
#include "Wafer/Planning/PhysicalDataflow/AccessReuse.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {
using Status = analysis::IndexRelationStatus;

AccessReuseBindingFailure
unavailable(const analysis::AccessReuseAnalysis &facts,
            llvm::StringRef detail) {
  return {facts.status == Status::Invalid ? Status::Invalid
          : facts.status == Status::ResourceExhausted ||
                  facts.indeterminateScopes
              ? Status::ResourceExhausted
              : Status::Unsupported,
          detail.str()};
}

bool sameSource(const AccessReuseSelection &selection, TileModuleOp tile,
                int64_t argument) {
  return int64_t(tile.getCardId()) == selection.card &&
         argument == selection.argument &&
         (selection.kind == AccessReuseKind::Peer ||
          int64_t(tile.getTileId()) == selection.tile);
}

bool sameAction(const AccessReuseAction &a, const AccessReuseAction &b) {
  return a.kind == b.kind && a.scope == b.scope &&
         a.innerScope == b.innerScope && a.shareWindow == b.shareWindow &&
         a.peerTopology == b.peerTopology && a.reads.size() == b.reads.size() &&
         llvm::all_of(a.reads, [&](auto read) {
           return llvm::is_contained(b.reads, read);
         });
}
} // namespace

bool operator==(const AccessReuseSelection &a, const AccessReuseSelection &b) {
  return a.kind == b.kind && a.card == b.card && a.tile == b.tile &&
         a.argument == b.argument && a.participants == b.participants &&
         a.scope == b.scope && a.innerScope == b.innerScope &&
         a.slidingAxis == b.slidingAxis && a.shareWindow == b.shareWindow &&
         a.peerTopology == b.peerTopology;
}

bool operator==(const AccessReuseIntent &a, const AccessReuseIntent &b) {
  return a.selections.size() == b.selections.size() &&
         llvm::all_of(a.selections, [&](const auto &selection) {
           return llvm::is_contained(b.selections, selection);
         });
}

AccessReuseBinding bindAccessReuse(const AccessReuseIntent &intent,
                                   const analysis::AccessReuseAnalysis &facts) {
  if (facts.status == Status::Invalid)
    return unavailable(facts, "invalid current read access facts");
  AccessReuseChoice result;
  llvm::DenseSet<mlir::Operation *> selectedReads;
  for (const auto &selection : intent.selections) {
    const auto start = result.actions.size();
    auto append = [&](llvm::ArrayRef<StorageLoadOp> reads,
                      mlir::scf::ForOp scope = {},
                      mlir::scf::ForOp inner = {}) {
      AccessReuseAction action;
      action.kind = selection.kind;
      llvm::append_range(action.reads, reads);
      action.scope = scope;
      action.innerScope = inner;
      action.shareWindow = selection.shareWindow;
      action.peerTopology = selection.peerTopology;
      result.actions.push_back(std::move(action));
    };
    if (selection.kind == AccessReuseKind::Peer) {
      for (const auto &group : facts.peers) {
        if (group.reads.empty() ||
            !sameSource(selection, group.reads.front().tile,
                        group.reads.front().argument))
          continue;
        llvm::SmallVector<StorageLoadOp, 4> reads;
        llvm::SmallVector<uint64_t, 4> participants;
        for (const auto &read : group.reads) {
          reads.push_back(read.load);
          auto tile = read.tile;
          participants.push_back(tile.getTileId());
        }
        llvm::sort(participants);
        participants.erase(
            std::unique(participants.begin(), participants.end()),
            participants.end());
        if (participants != selection.participants)
          continue;
        append(reads);
      }
    } else if (selection.kind == AccessReuseKind::Sliding) {
      for (const auto &window : facts.sliding) {
        auto scope = window.scope;
        if (sameSource(selection, window.access.tile, window.access.argument) &&
            getIterationCoordinates(scope) == selection.scope &&
            window.axis == selection.slidingAxis)
          append({window.access.load}, scope);
      }
    } else {
      for (const auto &window : facts.scopes) {
        auto scope = window.scope;
        if (!sameSource(selection, window.tile, window.argument) ||
            getIterationCoordinates(scope) != selection.scope)
          continue;
        if (selection.kind == AccessReuseKind::Resident) {
          append(window.reads, scope);
          continue;
        }
        mlir::scf::ForOp inner;
        for (const auto &candidate : facts.scopes) {
          auto candidateScope = candidate.scope;
          if (candidate.source != window.source ||
              !scope->isAncestor(candidateScope) ||
              getIterationCoordinates(candidateScope) != selection.innerScope)
            continue;
          if (inner && inner != candidateScope)
            return unavailable(
                facts, "reuse inner scope has multiple current instances");
          inner = candidateScope;
        }
        if (!inner)
          return unavailable(
              facts, "reuse inner scope is unavailable at this tile point");
        append(window.reads, scope, inner);
      }
    }
    if (result.actions.size() == start)
      return unavailable(facts,
                         "reuse selection has no exact current read scope");
    for (const auto &action : llvm::ArrayRef(result.actions).drop_front(start))
      for (auto read : action.reads)
        if (!selectedReads.insert(read).second)
          return unavailable(facts, "reuse selections overlap current reads");
  }
  return result;
}

AccessReuseCapture
captureAccessReuse(const AccessReuseChoice &choice,
                   const analysis::AccessReuseAnalysis &facts) {
  AccessReuseIntent result;
  for (const auto &action : choice.actions) {
    if (action.reads.empty())
      return AccessReuseBindingFailure{Status::Invalid,
                                       "reuse choice has no current reads"};
    auto read = llvm::find_if(facts.reads, [&](const auto &access) {
      return access.load == action.reads.front();
    });
    if (read == facts.reads.end() || read->argument < 0)
      return AccessReuseBindingFailure{
          Status::Invalid, "reuse choice lacks current input provenance"};
    auto tile = read->tile;
    AccessReuseSelection selection;
    selection.kind = action.kind;
    selection.card = tile.getCardId();
    selection.tile =
        action.kind == AccessReuseKind::Peer ? -1 : tile.getTileId();
    selection.argument = read->argument;
    selection.shareWindow = action.shareWindow;
    selection.peerTopology = action.peerTopology;
    if (action.kind == AccessReuseKind::Peer) {
      for (auto load : action.reads) {
        auto peer = llvm::find_if(facts.reads, [&](const auto &access) {
          return access.load == load;
        });
        if (peer == facts.reads.end() ||
            !sameSource(selection, peer->tile, peer->argument))
          return AccessReuseBindingFailure{
              Status::Invalid, "peer choice spans different input resources"};
        auto participant = peer->tile;
        selection.participants.push_back(participant.getTileId());
      }
      llvm::sort(selection.participants);
      selection.participants.erase(std::unique(selection.participants.begin(),
                                               selection.participants.end()),
                                   selection.participants.end());
    }
    if (action.kind != AccessReuseKind::Peer) {
      auto scope = action.scope;
      selection.scope =
          scope ? getIterationCoordinates(scope) : IterationCoordinatesAttr{};
      if (!selection.scope)
        return unavailable(facts, "reuse scope has no iteration coordinates");
    }
    if (action.kind == AccessReuseKind::TwoLevel) {
      auto inner = action.innerScope;
      selection.innerScope =
          inner ? getIterationCoordinates(inner) : IterationCoordinatesAttr{};
      if (!selection.innerScope)
        return unavailable(facts,
                           "reuse inner scope has no iteration coordinates");
    }
    if (action.kind == AccessReuseKind::Sliding) {
      auto sliding = llvm::find_if(facts.sliding, [&](const auto &window) {
        return window.access.load == action.reads.front() &&
               window.scope == action.scope;
      });
      if (sliding == facts.sliding.end())
        return AccessReuseBindingFailure{
            Status::Invalid, "selected sliding access lacks current proof"};
      selection.slidingAxis = sliding->axis;
    }
    if (!llvm::is_contained(result.selections, selection))
      result.selections.push_back(selection);
  }
  // Capturing a scope family must not widen a partial selection to other
  // current instances (for example a separately selected peeled tail).
  auto bound = bindAccessReuse(result, facts);
  if (auto *failure = std::get_if<AccessReuseBindingFailure>(&bound))
    return *failure;
  const auto &actions = std::get<AccessReuseChoice>(bound).actions;
  if (actions.size() != choice.actions.size() ||
      !llvm::all_of(actions, [&](const auto &action) {
        return llvm::any_of(choice.actions, [&](const auto &selected) {
          return sameAction(action, selected);
        });
      }))
    return unavailable(
        facts,
        "reuse choice does not cover its complete iteration scope family");
  return result;
}
} // namespace wafer::compiler::detail
