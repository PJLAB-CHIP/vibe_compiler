//===- RankCandidateSelectionMode.h - Rank candidate selection modes -*- C++ -*-===//

#ifndef WAFER_COMPILER_RANKCANDIDATESELECTIONMODE_H
#define WAFER_COMPILER_RANKCANDIDATESELECTIONMODE_H

namespace wafer::compiler::detail {

/// Controls which already-generated, fully accepted whole-rank variant is
/// selected by the compiler. Production evaluates the accepted candidates;
/// ReservedBaseline selects the unique conservative candidate carried by each
/// rank candidates. Qualification modes select only an actual fixed-slot or
/// disjoint-component worker realization that passes the same late gates; they
/// do not alter production cost or preference. The characterization modes
/// select a fully accepted variant only when its final instruction IR contains
/// exactly the requested collective algorithm phase for that collective
/// family. The NoC-resident ring qualification additionally rejects reserved
/// candidates and proves from every accepted rank's current IR that DDR
/// movement is limited to entry-boundary reads and returned-output writes;
/// output-generation labels are not semantic residency. These are internal
/// test seams, not user options or alternate IR contracts.
enum class RankCandidateSelectionMode {
  Production,
  ReservedBaseline,
  QualifyStaticFixedSlot,
  QualifyDirectDTEComputeOverlap,
  SelectSerializedDirectDTEComputeBaseline,
  QualifyWorkerPlacement,
  QualifyNoCResidentFixedSlotWorker,
  CharacterizeAllGatherDirect,
  CharacterizeAllGatherRing,
  CharacterizeReduceScatterDirect,
  CharacterizeReduceScatterRing,
  CharacterizeAllReduceRing,
  QualifyNoCResidentAllReduceRing,
  CharacterizeAllReduceTree,
};

constexpr bool
isCollectiveCharacterizationSelection(RankCandidateSelectionMode mode) {
  switch (mode) {
  case RankCandidateSelectionMode::Production:
  case RankCandidateSelectionMode::ReservedBaseline:
  case RankCandidateSelectionMode::QualifyStaticFixedSlot:
  case RankCandidateSelectionMode::QualifyDirectDTEComputeOverlap:
  case RankCandidateSelectionMode::SelectSerializedDirectDTEComputeBaseline:
  case RankCandidateSelectionMode::QualifyWorkerPlacement:
  case RankCandidateSelectionMode::QualifyNoCResidentFixedSlotWorker:
    return false;
  case RankCandidateSelectionMode::CharacterizeAllGatherDirect:
  case RankCandidateSelectionMode::CharacterizeAllGatherRing:
  case RankCandidateSelectionMode::CharacterizeReduceScatterDirect:
  case RankCandidateSelectionMode::CharacterizeReduceScatterRing:
  case RankCandidateSelectionMode::CharacterizeAllReduceRing:
  case RankCandidateSelectionMode::QualifyNoCResidentAllReduceRing:
  case RankCandidateSelectionMode::CharacterizeAllReduceTree:
    return true;
  }
  return false;
}

constexpr const char *
getCollectiveCharacterizationAlternative(RankCandidateSelectionMode mode) {
  switch (mode) {
  case RankCandidateSelectionMode::CharacterizeAllGatherDirect:
    return "all-gather-direct";
  case RankCandidateSelectionMode::CharacterizeAllGatherRing:
    return "all-gather-ring";
  case RankCandidateSelectionMode::CharacterizeReduceScatterDirect:
    return "reduce-scatter-direct";
  case RankCandidateSelectionMode::CharacterizeReduceScatterRing:
    return "reduce-scatter-ring";
  case RankCandidateSelectionMode::CharacterizeAllReduceRing:
  case RankCandidateSelectionMode::QualifyNoCResidentAllReduceRing:
    return "all-reduce-ring";
  case RankCandidateSelectionMode::CharacterizeAllReduceTree:
    return "all-reduce-tree";
  case RankCandidateSelectionMode::Production:
  case RankCandidateSelectionMode::ReservedBaseline:
  case RankCandidateSelectionMode::QualifyStaticFixedSlot:
  case RankCandidateSelectionMode::QualifyDirectDTEComputeOverlap:
  case RankCandidateSelectionMode::SelectSerializedDirectDTEComputeBaseline:
  case RankCandidateSelectionMode::QualifyWorkerPlacement:
  case RankCandidateSelectionMode::QualifyNoCResidentFixedSlotWorker:
    return "";
  }
  return "";
}

/// Static fixed-slot qualification writes an attestation record in the same
/// no-replace directory transaction as its canonical package. The compound
/// qualification reuses that record schema while adding independent
/// NoC-residency and worker-placement selection predicates.
constexpr bool
writesStaticFixedSlotQualificationRecord(RankCandidateSelectionMode mode) {
  return mode == RankCandidateSelectionMode::QualifyStaticFixedSlot ||
         mode == RankCandidateSelectionMode::QualifyDirectDTEComputeOverlap ||
         mode == RankCandidateSelectionMode::QualifyNoCResidentFixedSlotWorker;
}

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_RANKCANDIDATESELECTIONMODE_H
