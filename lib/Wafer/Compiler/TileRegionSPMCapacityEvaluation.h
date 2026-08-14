//===- TileRegionSPMCapacityEvaluation.h - Region SPM evaluation -*- C++ -*-===//

#ifndef WAFER_COMPILER_TILEREGIONSPMCAPACITYEVALUATION_H
#define WAFER_COMPILER_TILEREGIONSPMCAPACITYEVALUATION_H

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/MemoryPlanning.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace wafer::compiler::detail {

enum class TileRegionSPMCapacityStatus : uint8_t {
  Fits,
  RequiresFunctionScope,
  CapacityExceeded,
  AnalysisFailure,
};

enum class TileRegionSPMCapacityPhase : uint8_t {
  None,
  InputValidation,
  InstructionLowering,
  StaticPacking,
};

/// Classification produced by one fixed-capacity TileRegion SPM query. The
/// query clones the region into a private evaluation scope, lowers its Tile
/// dataflow operations, places the joins required within that isolated scope,
/// derives lifetimes, and invokes static SPM packing without assigning
/// offsets. It owns no candidate-selection policy and does not create a
/// synthetic module/function or run function-boundary bufferization.
struct TileRegionSPMCapacityEvaluation {
  TileRegionSPMCapacityStatus status =
      TileRegionSPMCapacityStatus::AnalysisFailure;
  TileRegionSPMCapacityPhase phase = TileRegionSPMCapacityPhase::None;
  std::string detail;
  SPMMemoryPlanningFailure planningFailure;

  bool fits() const {
    return status == TileRegionSPMCapacityStatus::Fits;
  }
  bool capacityExceeded() const {
    return status == TileRegionSPMCapacityStatus::CapacityExceeded;
  }
  bool requiresFunctionScope() const {
    return status == TileRegionSPMCapacityStatus::RequiresFunctionScope;
  }
  llvm::StringRef getPhaseDiagnosticLabel() const;
};

TileRegionSPMCapacityEvaluation evaluateTileRegionSPMCapacity(
    TileRegionOp region, TileRegionToInstrLoweringSession &loweringSession,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_TILEREGIONSPMCAPACITYEVALUATION_H
