//===- CardExecutableSearch.cpp - Card executable selection -----------===//

#include "Wafer/Compiler/Search/CardExecutableSearch.h"

#include "Wafer/Compiler/Search/TensorProgramAlternative.h"

namespace wafer::compiler::detail {

mlir::FailureOr<CardExecutableLoweringResult>
runCardExecutableSearch(mlir::ModuleOp tensorProgram,
                        CardExecutableLoweringResult baseline) {
  mlir::FailureOr<TensorProgramAlternativeDomain> alternatives =
      getTensorProgramAlternativeDomain(tensorProgram);
  if (mlir::failed(alternatives))
    return mlir::failure();
  // Q50.S assignments remain partial until Q50.B-K supply all physical axes.
  // Querying establishes the typed mechanism boundary; enumerating or cloning
  // these roots now would perform work that no complete candidate can consume.
  return baseline;
}

} // namespace wafer::compiler::detail
