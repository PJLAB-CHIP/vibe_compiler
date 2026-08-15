//===- WholeVariantSelection.h - Whole-variant selection ------*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLEVARIANTSELECTION_H
#define WAFER_COMPILER_WHOLEVARIANTSELECTION_H

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
enum class WholeVariantSelectionMode {
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
isCollectiveCharacterizationSelection(WholeVariantSelectionMode mode) {
  switch (mode) {
  case WholeVariantSelectionMode::Production:
  case WholeVariantSelectionMode::ReservedBaseline:
  case WholeVariantSelectionMode::QualifyStaticFixedSlot:
  case WholeVariantSelectionMode::QualifyDirectDTEComputeOverlap:
  case WholeVariantSelectionMode::SelectSerializedDirectDTEComputeBaseline:
  case WholeVariantSelectionMode::QualifyWorkerPlacement:
  case WholeVariantSelectionMode::QualifyNoCResidentFixedSlotWorker:
    return false;
  case WholeVariantSelectionMode::CharacterizeAllGatherDirect:
  case WholeVariantSelectionMode::CharacterizeAllGatherRing:
  case WholeVariantSelectionMode::CharacterizeReduceScatterDirect:
  case WholeVariantSelectionMode::CharacterizeReduceScatterRing:
  case WholeVariantSelectionMode::CharacterizeAllReduceRing:
  case WholeVariantSelectionMode::QualifyNoCResidentAllReduceRing:
  case WholeVariantSelectionMode::CharacterizeAllReduceTree:
    return true;
  }
  return false;
}

constexpr const char *
getCollectiveCharacterizationAlternative(WholeVariantSelectionMode mode) {
  switch (mode) {
  case WholeVariantSelectionMode::CharacterizeAllGatherDirect:
    return "all-gather-direct";
  case WholeVariantSelectionMode::CharacterizeAllGatherRing:
    return "all-gather-ring";
  case WholeVariantSelectionMode::CharacterizeReduceScatterDirect:
    return "reduce-scatter-direct";
  case WholeVariantSelectionMode::CharacterizeReduceScatterRing:
    return "reduce-scatter-ring";
  case WholeVariantSelectionMode::CharacterizeAllReduceRing:
  case WholeVariantSelectionMode::QualifyNoCResidentAllReduceRing:
    return "all-reduce-ring";
  case WholeVariantSelectionMode::CharacterizeAllReduceTree:
    return "all-reduce-tree";
  case WholeVariantSelectionMode::Production:
  case WholeVariantSelectionMode::ReservedBaseline:
  case WholeVariantSelectionMode::QualifyStaticFixedSlot:
  case WholeVariantSelectionMode::QualifyDirectDTEComputeOverlap:
  case WholeVariantSelectionMode::SelectSerializedDirectDTEComputeBaseline:
  case WholeVariantSelectionMode::QualifyWorkerPlacement:
  case WholeVariantSelectionMode::QualifyNoCResidentFixedSlotWorker:
    return "";
  }
  return "";
}

/// Static fixed-slot qualification writes an attestation record in the same
/// no-replace directory transaction as its canonical package. The compound
/// qualification reuses that record schema while adding independent
/// NoC-residency and worker-placement selection predicates.
constexpr bool
writesStaticFixedSlotQualificationRecord(WholeVariantSelectionMode mode) {
  return mode == WholeVariantSelectionMode::QualifyStaticFixedSlot ||
         mode == WholeVariantSelectionMode::QualifyDirectDTEComputeOverlap ||
         mode == WholeVariantSelectionMode::QualifyNoCResidentFixedSlotWorker;
}

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTSELECTION_H
