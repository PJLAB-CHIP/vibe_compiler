//===- CardExecutableSearch.h - Card executable selection -----*- C++ -*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLESEARCH_H
#define WAFER_COMPILER_CARDEXECUTABLESEARCH_H

#include "Wafer/Compiler/Executable/CardExecutableLowering.h"

namespace wafer::compiler::detail {

/// Starts the current search session from an already accepted move-only
/// baseline. Q51.Core has no production mechanisms yet, so the exact same
/// executable owner is returned without cloning, lowering, or recompiling IR.
/// Q50 mechanisms extend this boundary with typed assignments and transitions.
CardExecutableLoweringResult
runCardExecutableSearch(CardExecutableLoweringResult baseline);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLESEARCH_H
