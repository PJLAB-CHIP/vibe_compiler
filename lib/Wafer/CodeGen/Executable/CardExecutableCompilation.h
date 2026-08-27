//===- CardExecutableCompilation.h - Policy-free executable seam -*- C++
//-*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H
#define WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H

#include "Wafer/CodeGen/Executable/CardExecutableLowering.h"
#include "Wafer/CodeGen/Executable/TileMemoryPlanning.h"

#include "Wafer/Frontend/Program/Program.h"
#include "Wafer/Target/Core/TopologyIds.h"

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

struct CandidateTileDataflowIR {
  CardId card{0};
  TileId tile{0};
  mlir::OwningOpRef<mlir::ModuleOp> *owner = nullptr;
  StructuredMaterializationRelations *relations = nullptr;

  mlir::ModuleOp getModule() const {
    return owner && *owner ? owner->get() : mlir::ModuleOp{};
  }
};

struct CandidateInstructionIR {
  CardId card{0};
  TileId tile{0};
  mlir::OwningOpRef<mlir::ModuleOp> *owner = nullptr;
  StructuredMaterializationRelations *relations = nullptr;
  struct RegionNodeRelation {
    mlir::Operation *region = nullptr;
    std::vector<uint32_t> structuredNodes;
  };
  std::vector<RegionNodeRelation> *regionNodes = nullptr;

  mlir::ModuleOp getModule() const {
    return owner && *owner ? owner->get() : mlir::ModuleOp{};
  }
};

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

enum class CardExecutablePreparationFailureKind : uint8_t {
  Unsupported,
  ExactRejection,
  Indeterminate,
  CompilerBug,
};

struct CardExecutablePreparationFailure {
  CardExecutablePreparationFailureKind kind =
      CardExecutablePreparationFailureKind::CompilerBug;
  std::string detail;
};

/// Invocation-local selected-candidate handoff. Implementations may mutate
/// only the borrowed current candidate modules during the call and must not
/// retain IR pointers. The enclosing compiler function owns rollback by
/// destroying the whole candidate on failure.
class CardExecutablePreparation {
public:
  virtual ~CardExecutablePreparation() = default;

  virtual bool ownsInstructionCompletion() const { return false; }

  virtual mlir::LogicalResult
  prepareTileDataflow(llvm::MutableArrayRef<CandidateTileDataflowIR> tiles,
                      CardExecutablePreparationFailure &failure) {
    (void)tiles;
    failure = {};
    return mlir::success();
  }

  virtual mlir::LogicalResult
  prepareInstructionIR(llvm::MutableArrayRef<CandidateInstructionIR> tiles,
                       CardExecutablePreparationFailure &failure) {
    (void)tiles;
    failure = {};
    return mlir::success();
  }
};

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
  UnsupportedFailure,
  IndeterminateFailure,
  CompilerFailure,
};

struct CardExecutableTileFailure {
  TileId tileId{0};
  std::string gate;
  std::string detail;
  TileMemoryPlanningFailure memoryPlanning;
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
CardExecutableCompilationStatus classifyTileMemoryPlanningFailure(
    const TileMemoryPlanningFailure &failure);

/// Runs the unique exact full-buffer transfer cleanup on canonical Instr and
/// retargets caller-owned current buffer relations in the same transaction.
/// Failure means the cleanup produced stale relations or invalid IR; zero is a
/// successful no-op.
mlir::FailureOr<unsigned> cleanupCanonicalInstructionTransfers(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations);

/// Move-only result of compiling one already selected CardModule.  An exact
/// rejection is backed by explicit capacity evidence in the returned per-Tile
/// failures. Unsupported IR, resource exhaustion and compiler failures retain
/// distinct typed classifications and cannot become a search no-good.
struct CardExecutableCompilationResult {
  CardExecutableCompilationStatus status =
      CardExecutableCompilationStatus::IndeterminateFailure;
  std::optional<CardExecutableLoweringResult> executable;
  std::string gate;
  std::string detail;
  llvm::SmallVector<CardExecutableTileFailure, 4> tileFailures;
  llvm::SmallVector<AcceptedOperationNodeRelation, 64> operationNodeRelations;
  /// Same-invocation diagnostic snapshots captured at the Tile dataflow to
  /// Instr boundary. They are not part of the accepted Tile modules.
  std::vector<std::string> tileDataflowIRTrace;

  bool isAccepted() const {
    return status == CardExecutableCompilationStatus::Accepted;
  }
  bool isProvenExactRejection() const {
    return status == CardExecutableCompilationStatus::ProvenExactRejection;
  }

  CardExecutableLoweringResult takeExecutable() {
    return std::move(*executable);
  }
};

/// Consumes all-and-only completion-closed canonical Instr modules for one
/// selected Card candidate. This is the unique actual SPM/DDR/transport/target
/// leaf shared by baseline and search. It does not bufferize, lower Tile
/// dataflow, choose worker/order, rebuild completion, repair memory pressure or
/// construct another candidate.
CardExecutableCompilationResult compileCanonicalInstructionTilesToExecutable(
    std::vector<CanonicalInstructionTile> tiles, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    CardExecutableLoweringStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

/// Compiles exactly one owned, verifier-legal, already selected CardModule.
/// The function performs no candidate enumeration and never changes spatial,
/// temporal, layout, movement or buffering choices. It materializes every Tile,
/// performs the currently selected upstream preparation and delegates the
/// resulting completion-closed canonical Instr owners to
/// `compileCanonicalInstructionTilesToExecutable`.
///
CardExecutableCompilationResult compileCardModuleToExecutable(
    mlir::OwningOpRef<mlir::ModuleOp> cardModule, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    const StructuredMaterializationRelations &materializationRelations,
    CardExecutablePreparation &preparation,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    CardExecutableLoweringStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0, bool captureTileIRTrace = false);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLECOMPILATION_H
