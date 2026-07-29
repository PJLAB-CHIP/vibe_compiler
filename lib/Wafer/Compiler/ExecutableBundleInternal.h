//===- ExecutableBundleInternal.h - Internal bundle construction -*- C++
//-*-===//

#ifndef WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H
#define WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H

#include "Wafer/Compiler/Compilation.h"
#include "WholeVariantCoordinator.h"
#include "WholeVariantSelection.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Internal construction seam shared by the rank-frontier coordinator and
/// final bundle owner.  Construction remains unavailable to public callers.
struct ExecutableBundleBuilder {
  static RankExecutable
  makeRank(int64_t logicalRank, mlir::OwningOpRef<mlir::ModuleOp> module,
           llvm::StringRef entrySymbol,
           std::vector<RankProgramBinding> programBindings,
           TransportContract transportContract) {
    return RankExecutable(logicalRank, std::move(module), entrySymbol,
                          std::move(programBindings), transportContract);
  }

  static ExecutableBundle
  makeBundle(ExecutionConfig executionConfig,
             RuntimeLaunchContract runtimeLaunchContract,
             std::shared_ptr<mlir::MLIRContext> context,
             std::vector<RankExecutable> ranks) {
    return ExecutableBundle(executionConfig, std::move(runtimeLaunchContract),
                            std::move(context), std::move(ranks));
  }
};

namespace detail {

/// Invocation-local cross-context transport for one finalized rank candidate.
/// The module text is parsed only when the fixed whole-variant attempt plan can
/// reach this original frontier slot through its correspondence precheck.
struct SerializedRankVariantCandidate {
  std::string moduleText;
  int64_t stableOrdinal = 0;
  wafer::RankArtifactKind artifactKind = wafer::RankArtifactKind::Spill;
  bool reservedBaseline = false;
  wafer::RankBufferingKind bufferingKind = wafer::RankBufferingKind::Single;
  uint32_t bufferingPlanOrdinal = 0;
  wafer::RankWorkerPlacementKind workerPlacementKind =
      wafer::RankWorkerPlacementKind::Unplaced;
  uint32_t workerPlacementPlanOrdinal = 0;
};

using SerializedRankVariantFrontier =
    std::vector<SerializedRankVariantCandidate>;

/// Imports exactly the module slots required by the existing bounded
/// whole-variant attempt sequence. Every original slot and its metadata remain
/// in the returned frontiers only through this import-audit boundary; an
/// unreachable slot retains a null module.
llvm::Expected<std::vector<RankVariantFrontier>>
importRankVariantFrontiersIntoOwnerContext(
    mlir::MLIRContext &ownerContext,
    llvm::ArrayRef<SerializedRankVariantFrontier> serializedFrontiers,
    int64_t expectedRankCount);

/// Ends the import-audit boundary by removing metadata-only tombstones.
/// Frontier indices after this point name actual owner-context artifacts and
/// intentionally do not preserve original serialized slot positions.
void compactImportedRankVariantFrontiers(
    std::vector<RankVariantFrontier> &frontiers);

mlir::LogicalResult
verifyExactExecutionConfig(mlir::ModuleOp module,
                           const ExecutionConfig &executionConfig);

llvm::Expected<ExecutableBundle> buildExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    WholeVariantSelectionMode selectionMode =
        WholeVariantSelectionMode::Production);

} // namespace detail
} // namespace wafer::compiler

#endif // WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H
