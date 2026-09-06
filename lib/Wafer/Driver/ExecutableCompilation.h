//===- ExecutableCompilation.h - Policy-free executable seam -*- C++
//-*-===//

#ifndef WAFER_DRIVER_EXECUTABLECOMPILATION_H
#define WAFER_DRIVER_EXECUTABLECOMPILATION_H

#include "Wafer/Driver/ExecutableLowering.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"

#include "Wafer/Frontend/Program.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
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

/// Move-only input for the actual memory/target leaf. Every module is already
/// function-boundary-bufferized, lowered to canonical Instr, assigned its final
/// worker/order and closed by the required completion operations. The leaf owns
/// these modules and their current owner relations and destroys all of them on
/// any failure.
struct CanonicalInstructionTile {
  CardId card{0};
  TileId tile{0};
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

enum class ExecutableCompilationStatus : uint8_t {
  Accepted,
  ProvenExactRejection,
  UnsupportedFailure,
  IndeterminateFailure,
  CompilerFailure,
};

enum class ExecutableFailureScope : uint8_t {
  /// Another Temporal choice over the same structural current IR may change
  /// the result; no stronger conclusion is available.
  TemporalChoiceMayChange,
  /// The failure depends only on the closed Spatial/Region current IR and
  /// remains unchanged for every Temporal choice in that structural domain.
  StructuralChoiceInvariant,
};

struct ExecutableTileFailure {
  TileId tileId{0};
  std::string gate;
  std::string detail;
  TileMemoryPlanningFailure memoryPlanning;
};

/// Bounded read-only summary of the accepted movement-closed physical IR.
/// It contains no operation/value handles and is never consulted by search,
/// legality, memory planning, target lowering or package serialization.
struct PhysicalTileIRInventory {
  TileId tile{0};
  uint64_t regions = 0;
  uint64_t nestedOperations = 0;
  uint64_t tileDataflowOperations = 0;
};

struct PhysicalDataflowIRInventory {
  uint64_t tileModules = 0;
  uint64_t tileRegions = 0;
  uint64_t nestedOperations = 0;
  uint64_t minimumNestedOperations = 0;
  uint64_t maximumNestedOperations = 0;
  uint64_t tileDataflowOperations = 0;
  uint64_t minimumTileDataflowOperations = 0;
  uint64_t maximumTileDataflowOperations = 0;
  uint64_t singletonDataflowRegions = 0;
  std::vector<PhysicalTileIRInventory> tiles;
};

/// Returns true only when Tile memory planning carries an explicit SPM
/// capacity-overflow proof. Every other typed failure remains outside the
/// exact-rejection class and therefore cannot become a candidate no-good.
bool isProvenExactTileMemoryPlanningFailure(
    const TileMemoryPlanningFailure &failure);

/// Classifies one Tile memory-planning outcome without inspecting diagnostic
/// text. Only a proven capacity result is exact; deterministic search resource
/// exhaustion remains indeterminate, unsupported lifetime is unsupported, and
/// a missing completion or malformed leaf input is a compiler failure.
ExecutableCompilationStatus
classifyTileMemoryPlanningFailure(const TileMemoryPlanningFailure &failure);

/// Runs the unique exact full-buffer transfer cleanup on canonical Instr and
/// retargets caller-owned current buffer relations in the same transaction.
/// Failure means the cleanup produced stale relations or invalid IR; zero is a
/// successful no-op.
mlir::FailureOr<unsigned> cleanupCanonicalInstructionTransfers(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations);

/// Move-only result of compiling one selected Tile-module collection. An exact
/// rejection is backed by explicit capacity evidence in the returned per-Tile
/// failures. Unsupported IR, resource exhaustion and compiler failures retain
/// distinct typed classifications and cannot become a search no-good.
struct ExecutableCompilationResult {
  ExecutableCompilationStatus status =
      ExecutableCompilationStatus::IndeterminateFailure;
  ExecutableFailureScope failureScope =
      ExecutableFailureScope::TemporalChoiceMayChange;
  std::optional<ExecutableLoweringResult> executable;
  std::string gate;
  std::string detail;
  llvm::SmallVector<ExecutableTileFailure, 4> tileFailures;
  std::optional<PhysicalDataflowIRInventory> physicalIRInventory;
  /// Same-invocation diagnostic snapshots captured at the Tile dataflow to
  /// Instr boundary. They are not part of the accepted Tile modules.
  std::vector<std::string> tileDataflowIRTrace;

  bool isAccepted() const {
    return status == ExecutableCompilationStatus::Accepted;
  }
  bool isProvenExactRejection() const {
    return status == ExecutableCompilationStatus::ProvenExactRejection;
  }

  ExecutableLoweringResult takeExecutable() { return std::move(*executable); }
};

/// Consumes all-and-only completion-closed canonical Instr modules for one
/// selected device candidate. This is the unique actual
/// SPM/DDR/transport/target leaf shared by baseline and search. It does not
/// bufferize, lower Tile
/// dataflow, choose worker/order, rebuild completion, repair memory pressure or
/// construct another candidate.
ExecutableCompilationResult compileCanonicalInstructionTilesToExecutable(
    std::vector<CanonicalInstructionTile> tiles, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    ExecutableLoweringStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_EXECUTABLECOMPILATION_H
