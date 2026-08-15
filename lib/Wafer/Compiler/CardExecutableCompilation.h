//===- CardExecutableCompilation.h - Policy-free executable seam -*- C++
//-*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H
#define WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H

#include "PhysicalTileMemoryPlanning.h"
#include "WholeCardExecutableLowering.h"

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

/// Query-local relation from one operation in an accepted Instr module to a
/// structured DAG node whose materialized buffer it reads, writes or forwards.
/// The operation pointer is valid only while the returned executable remains
/// unchanged.
struct AcceptedOperationNodeRelation {
  mlir::Operation *operation = nullptr;
  uint32_t structuredNodeId = 0;
};

enum class CardExecutableCompilationStatus : uint8_t {
  Accepted,
  ProvenExactRejection,
  IndeterminateFailure,
};

struct CardExecutableTileFailure {
  PhysicalTileId tileId{0};
  std::string gate;
  std::string detail;
  PhysicalTileMemoryPlanningFailure memoryPlanning;
  SelectedBufferMaterializationFailure selectedBuffer;
};

/// Returns true only when physical-Tile memory planning carries an explicit SPM
/// capacity-overflow proof. Verifier, unsupported-lifetime and pipeline
/// failures remain indeterminate rather than becoming candidate no-goods.
bool isProvenExactPhysicalTileMemoryPlanningFailure(
    const PhysicalTileMemoryPlanningFailure &failure);

/// Move-only result of compiling one already selected CardProgram.  An exact
/// rejection is backed by explicit capacity evidence in the returned
/// per-Tile failures. Unsupported IR, resource exhaustion, unclassified
/// allocator failure and internal pipeline failure remain indeterminate and
/// therefore cannot become a search no-good.
struct CardExecutableCompilationResult {
  CardExecutableCompilationStatus status =
      CardExecutableCompilationStatus::IndeterminateFailure;
  std::optional<WholeCardExecutable> executable;
  std::string gate;
  std::string detail;
  llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures;
  llvm::SmallVector<AcceptedOperationNodeRelation, 64> operationNodeRelations;
  /// Same-invocation diagnostic snapshots captured at the Tile dataflow to
  /// Instr boundary. They are not part of the accepted physical-Tile modules.
  std::vector<std::string> tileDataflowIRTrace;
  uint64_t rotatingSlotAllocationsMaterialized = 0;

  bool isAccepted() const {
    return status == CardExecutableCompilationStatus::Accepted;
  }
  bool isProvenExactRejection() const {
    return status == CardExecutableCompilationStatus::ProvenExactRejection;
  }

  WholeCardExecutable takeExecutable() { return std::move(*executable); }
};

/// Compiles exactly one owned, verifier-legal, already selected CardProgram.
/// The function performs no candidate enumeration and never changes spatial,
/// temporal, layout, movement or buffering choices.  It materializes every
/// physical Tile, lowers TileRegion to Instr, recomputes required NCC joins,
/// fixed-capacity SPM/DDR planning, Direct-DTE lowering, resource validation,
/// and target ABI/LLVM verification.
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
    const StructuredMaterializationRelations &materializationRelations,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeCardSynthesisStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H
