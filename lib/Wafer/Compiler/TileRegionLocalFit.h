//===- TileRegionLocalFit.h - Region-local SPM evaluation ----*- C++ -*-===//

#ifndef WAFER_COMPILER_TILEREGIONLOCALFIT_H
#define WAFER_COMPILER_TILEREGIONLOCALFIT_H

#include "PhysicalTileFinalization.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"

#include <cstdint>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace wafer::compiler::detail {

enum class TileRegionLocalFitStatus : uint8_t {
  Fits,
  ProvenExactRejection,
  IndeterminateFailure,
};

/// Invocation-local result for one already materialized TileRegion. The
/// evaluation owns no selection policy: its isolated transformation
/// transaction clones the region operation, lowers only its Tile actions,
/// derives region-local lifetimes and queries the fixed-capacity SPM allocator
/// without assigning offsets. It does not create a synthetic module/function,
/// run function completion/bufferization, choose another temporal tile, or
/// perform whole-card admission.
struct TileRegionLocalFitResult {
  TileRegionLocalFitStatus status =
      TileRegionLocalFitStatus::IndeterminateFailure;
  std::string gate;
  std::string detail;
  PhysicalTileFinalizationFailure finalization;

  bool fits() const { return status == TileRegionLocalFitStatus::Fits; }
  bool isProvenExactRejection() const {
    return status == TileRegionLocalFitStatus::ProvenExactRejection;
  }
};

TileRegionLocalFitResult
evaluateTileRegionLocalFit(TileRegionOp region,
                           TileRegionToInstrLoweringSession &loweringSession,
                           llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_TILEREGIONLOCALFIT_H
