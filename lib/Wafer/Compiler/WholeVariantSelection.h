//===- WholeVariantSelection.h - Whole-variant selection ------*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLEVARIANTSELECTION_H
#define WAFER_COMPILER_WHOLEVARIANTSELECTION_H

namespace wafer::compiler::detail {

/// Controls which already-generated, fully accepted whole-rank variant is
/// committed by the compiler. Production evaluates the accepted frontier;
/// ReservedBaseline commits the unique conservative candidate carried by each
/// rank frontier. This is an internal compilation seam, not a user option or
/// an alternate IR contract.
enum class WholeVariantSelectionMode {
  Production,
  ReservedBaseline,
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTSELECTION_H
