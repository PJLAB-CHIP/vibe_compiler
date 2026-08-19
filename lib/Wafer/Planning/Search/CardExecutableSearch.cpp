//===- CardExecutableSearch.cpp - Card executable selection -----------===//

#include "Wafer/Planning/Search/CardExecutableSearch.h"

#include "Wafer/Planning/Search/SpatialPlacement.h"
#include "Wafer/Planning/Search/TensorProgramAlternative.h"

namespace wafer::compiler::detail {

mlir::FailureOr<CardExecutableLoweringResult>
runCardExecutableSearch(mlir::ModuleOp tensorProgram,
                        const CardProgramAnalysis &programAnalysis,
                        CardExecutableLoweringResult baseline) {
  mlir::FailureOr<TensorProgramAlternativeDomain> alternatives =
      getTensorProgramAlternativeDomain(tensorProgram);
  if (mlir::failed(alternatives))
    return mlir::failure();
  mlir::FailureOr<CardSpatialPlacementDomain> spatialPlacements =
      CardSpatialPlacementDomain::create(programAnalysis.dag,
                                         programAnalysis.availableTileIds);
  if (mlir::failed(spatialPlacements))
    return mlir::failure();
  // Q50.S/B assignments remain partial until Q50.C-K supply all physical axes.
  // Querying establishes the typed mechanism boundary; enumerating or cloning
  // these roots now would perform work that no complete candidate can consume.
  return baseline;
}

} // namespace wafer::compiler::detail
