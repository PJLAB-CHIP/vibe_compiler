//===- AccessReuse.h - Explicit read reuse choices --------------*- C++ -*-===//
#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_ACCESSREUSE_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_ACCESSREUSE_H

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Analysis/Tile/AccessReuseAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/CollectiveAlgorithms.h"
#include "mlir/IR/IRMapping.h"

#include <variant>

namespace wafer::compiler::detail {

enum class PeerReuseTopology : uint8_t { Direct, SpanningTree };

enum class AccessReuseKind : uint8_t { Peer, Resident, Sliding, TwoLevel };

/// Current read operations and loop anchors only. The choice does not own
/// future allocations, copies, offsets, communication events or lifetimes.
struct AccessReuseAction {
  AccessReuseKind kind = AccessReuseKind::Peer;
  llvm::SmallVector<StorageLoadOp, 4> reads;
  mlir::scf::ForOp scope;
  mlir::scf::ForOp innerScope;
  bool shareWindow = false;
  PeerReuseTopology peerTopology = PeerReuseTopology::Direct;
};

/// Edges use indices in the explicit current participant list, never IR
/// traversal positions or a cross-clone correspondence.
mlir::FailureOr<llvm::SmallVector<BroadcastTreeEdge, 16>>
buildPeerReuseEdges(llvm::ArrayRef<StorageLoadOp> reads,
                    PeerReuseTopology topology);

struct AccessReuseChoice {
  llvm::SmallVector<AccessReuseAction, 4> actions;
};

/// A choice over a program input and a family of actual iteration coordinates.
/// It contains no read/loop handles from a previous materialization. Rebinding
/// always queries the new candidate's current accesses and effects.
struct AccessReuseSelection {
  AccessReuseKind kind = AccessReuseKind::Peer;
  int64_t card = -1;
  int64_t tile = -1;
  int64_t argument = -1;
  llvm::SmallVector<uint64_t, 4> participants;
  IterationCoordinatesAttr scope;
  IterationCoordinatesAttr innerScope;
  unsigned slidingAxis = 0;
  bool shareWindow = false;
  PeerReuseTopology peerTopology = PeerReuseTopology::Direct;

  friend bool operator==(const AccessReuseSelection &a,
                         const AccessReuseSelection &b);
};

struct AccessReuseIntent {
  llvm::SmallVector<AccessReuseSelection, 4> selections;
  friend bool operator==(const AccessReuseIntent &a,
                         const AccessReuseIntent &b);
};

struct AccessReuseBindingFailure {
  analysis::IndexRelationStatus status;
  std::string detail;
};
using AccessReuseCapture =
    std::variant<AccessReuseIntent, AccessReuseBindingFailure>;
using AccessReuseBinding =
    std::variant<AccessReuseChoice, AccessReuseBindingFailure>;

AccessReuseCapture
captureAccessReuse(const AccessReuseChoice &choice,
                   const analysis::AccessReuseAnalysis &facts);
AccessReuseBinding bindAccessReuse(const AccessReuseIntent &intent,
                                   const analysis::AccessReuseAnalysis &facts);

struct AccessReuseProposal {
  AccessReuseChoice choice;
  /// Ranking prior from current read windows; never a memory certificate.
  long double estimatedBenefitPicoseconds = 0;
};

struct AccessReuseProposals {
  llvm::SmallVector<AccessReuseProposal, 4> choices;
  uint64_t opportunities = 0;
  uint64_t lowBenefit = 0;
  uint64_t unknownBenefit = 0;
};

/// Explicit peer qualification uses the same materializer, without imposing
/// the automatic search profitability heuristic on mechanism coverage.
AccessReuseChoice
selectPeerAccessReuse(const analysis::AccessReuseAnalysis &facts);

AccessReuseProposals
proposeAccessReuse(const analysis::AccessReuseAnalysis &facts,
                   const analysis::SearchCostPolicy &policy);

mlir::FailureOr<AccessReuseChoice>
mapAccessReuseChoice(const AccessReuseChoice &choice,
                     const mlir::IRMapping &mapping);

} // namespace wafer::compiler::detail
#endif
