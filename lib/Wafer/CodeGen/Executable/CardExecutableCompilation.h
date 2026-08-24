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

  mlir::ModuleOp getModule() const {
    return owner && *owner ? owner->get() : mlir::ModuleOp{};
  }
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
/// capacity-overflow proof. Verifier, unsupported-lifetime and pipeline
/// failures remain indeterminate rather than becoming candidate no-goods.
bool isProvenExactTileMemoryPlanningFailure(
    const TileMemoryPlanningFailure &failure);

/// Move-only result of compiling one already selected CardModule.  An exact
/// rejection is backed by explicit capacity evidence in the returned
/// per-Tile failures. Unsupported IR, resource exhaustion, unclassified
/// allocator failure and internal pipeline failure remain indeterminate and
/// therefore cannot become a search no-good.
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

/// Compiles exactly one owned, verifier-legal, already selected CardModule.
/// The function performs no candidate enumeration and never changes spatial,
/// temporal, layout, movement or buffering choices.  It materializes every
/// Tile, lowers TileRegion to Instr, recomputes required NCC joins,
/// fixed-capacity SPM/DDR planning, Direct-DTE lowering, resource validation,
/// and target ABI/LLVM verification.
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
