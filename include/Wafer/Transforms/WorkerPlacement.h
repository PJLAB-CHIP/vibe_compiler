//===- WorkerPlacement.h - Typed NCC worker alternatives -------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_WORKERPLACEMENT_H
#define WAFER_TRANSFORMS_WORKERPLACEMENT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <string>

namespace wafer {

/// One independently owned worker-placement actual clone. Lane count and
/// participant mask are derivation diagnostics only; worker attrs and typed
/// completion joins in the module are the semantic result.
struct NCCWorkerPlacementCandidate {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  unsigned dependencyLaneCount = 0;
  uint32_t participantMask = 0;
};

/// Derive the canonical bounded nonzero-worker alternative from an unplaced
/// complete-rank instruction module.
///
/// Stable issue order, SSA dependencies, and potentially conflicting
/// value-associated effects form directed work lanes. An issue without a
/// prior dependency starts a lane; a dependent issue continues its latest
/// predecessor lane. At a fan-in, earlier predecessor lanes remain distinct
/// until exact completion inserts the minimal cross-worker participant join,
/// instead of collapsing the useful producer window into one undirected
/// component. Alias roots, typed Wafer memory spaces, and static half-open view
/// byte ranges prove disjointness; an unknown same-space relation fails closed
/// to a dependency edge. Direct RAW/WAR/WAW chains therefore stay on one
/// worker, while RAR and provably disjoint producers may use distinct workers.
/// Lanes are assigned deterministically over the finite typed worker domain.
/// Existing compiler-generated typed joins are discarded in the private clone
/// and rebuilt from the resulting actual workers and effects, so cross-domain
/// observers and terminal joins name exactly the pending participants.
///
/// The source is never modified. Direct-DTE issues remain on their typed
/// engine and are selected only when every issue has one exact same-block
/// token wait; their memory effects still participate in the rebuilt
/// cross-engine joins. Existing nonzero placement, physical SPM/DDR offsets,
/// synchronous host writeback, legacy fences, or unsupported control/effect
/// structure reject the optional alternative.
mlir::FailureOr<NCCWorkerPlacementCandidate>
deriveDisjointNCCWorkerPlacementCandidate(mlir::ModuleOp sourceModule,
                                          std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_WORKERPLACEMENT_H
