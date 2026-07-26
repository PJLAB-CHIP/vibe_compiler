//===- WholeVariantSelection.h - Whole-variant selection ------*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLEVARIANTSELECTION_H
#define WAFER_COMPILER_WHOLEVARIANTSELECTION_H

namespace wafer::compiler::detail {

/// Controls which already-generated, fully accepted whole-rank variant is
/// committed by the compiler. Production evaluates the accepted frontier;
/// ReservedBaseline commits the unique conservative candidate carried by each
/// rank frontier. The characterization modes select a fully accepted variant
/// only when its final instruction IR contains exactly the requested
/// collective algorithm phase for that collective family. These are internal
/// test seams, not user options or alternate IR contracts.
enum class WholeVariantSelectionMode {
  Production,
  ReservedBaseline,
  CharacterizeAllGatherDirect,
  CharacterizeAllGatherRing,
  CharacterizeReduceScatterDirect,
  CharacterizeReduceScatterRing,
  CharacterizeAllReduceRing,
  CharacterizeAllReduceTree,
};

constexpr bool
isCollectiveCharacterizationSelection(WholeVariantSelectionMode mode) {
  switch (mode) {
  case WholeVariantSelectionMode::Production:
  case WholeVariantSelectionMode::ReservedBaseline:
    return false;
  case WholeVariantSelectionMode::CharacterizeAllGatherDirect:
  case WholeVariantSelectionMode::CharacterizeAllGatherRing:
  case WholeVariantSelectionMode::CharacterizeReduceScatterDirect:
  case WholeVariantSelectionMode::CharacterizeReduceScatterRing:
  case WholeVariantSelectionMode::CharacterizeAllReduceRing:
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
    return "all-reduce-ring";
  case WholeVariantSelectionMode::CharacterizeAllReduceTree:
    return "all-reduce-tree";
  case WholeVariantSelectionMode::Production:
  case WholeVariantSelectionMode::ReservedBaseline:
    return "";
  }
  return "";
}

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTSELECTION_H
