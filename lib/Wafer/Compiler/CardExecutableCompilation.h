//===- CardExecutableCompilation.h - Policy-free executable seam -*- C++
//-*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H
#define WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H

#include "PhysicalTileFinalization.h"
#include "WholeCardExecutableAdmission.h"

#include "Wafer/Frontend/Program.h"
#include "Wafer/Target/PhysicalIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace wafer::compiler::detail {

enum class CardExecutableCompilationStatus : uint8_t {
  Accepted,
  ProvenExactRejection,
  IndeterminateFailure,
};

struct CardExecutableTileFailure {
  PhysicalTileId tileId{0};
  std::string gate;
  std::string detail;
  PhysicalTileFinalizationFailure finalization;
  SelectedBufferMaterializationFailure selectedBuffer;
};

/// Move-only result of compiling one already selected CardProgram.  An exact
/// rejection is backed by verifier/unsupported/capacity evidence in the
/// returned gate and optional per-Tile failures.  Resource exhaustion,
/// unclassified allocator failure and internal pipeline failure remain
/// indeterminate and therefore cannot become a search no-good.
struct CardExecutableCompilationResult {
  CardExecutableCompilationStatus status =
      CardExecutableCompilationStatus::IndeterminateFailure;
  std::optional<AcceptedWholeCardExecutable> executable;
  std::string gate;
  std::string detail;
  llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures;
  uint64_t rotatingSlotAllocationsMaterialized = 0;

  bool isAccepted() const {
    return status == CardExecutableCompilationStatus::Accepted;
  }
  bool isProvenExactRejection() const {
    return status == CardExecutableCompilationStatus::ProvenExactRejection;
  }

  AcceptedWholeCardExecutable takeExecutable() {
    return std::move(*executable);
  }
};

/// Compiles exactly one owned, verifier-legal, already selected CardProgram.
/// The function performs no candidate enumeration and never changes spatial,
/// temporal, layout, movement or buffering choices.  It projects every
/// physical Tile, lowers TileRegion to Instr, rebuilds completion, performs
/// fixed-capacity SPM/DDR planning and runs transport/resource/ABI admission.
///
/// `selectedBufferRequests`, when nonempty, must have one entry for every
/// expected Tile in canonical physical-Tile order.  They are typed assignments
/// owned by the caller; this boundary only materializes and verifies them.
CardExecutableCompilationResult compileCardProgramToExecutable(
    mlir::OwningOpRef<mlir::ModuleOp> cardProgram,
    PhysicalCardId expectedCardId,
    llvm::ArrayRef<PhysicalTileId> expectedTileIds,
    llvm::ArrayRef<llvm::SmallVector<SelectedBufferRequest, 4>>
        selectedBufferRequests,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeCardSynthesisStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H
