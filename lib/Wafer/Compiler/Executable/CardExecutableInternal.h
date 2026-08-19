//===- CardExecutableInternal.h - Internal construction -*- C++
//-*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLEINTERNAL_H
#define WAFER_COMPILER_CARDEXECUTABLEINTERNAL_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/ProgramData.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Internal constructors for verified card executables.
struct CardExecutableBuilder {
  static TileExecutable
  makeTileExecutable(CardId cardId, TileId tileId,
                     LaunchSlotId launchSlotId,
                     mlir::OwningOpRef<mlir::ModuleOp> module,
                     llvm::StringRef entrySymbol,
                     std::vector<ProgramResourceBinding> programBindings,
                     TransportContract transportContract) {
    return TileExecutable(
        cardId, tileId, launchSlotId, std::move(module),
        entrySymbol, std::move(programBindings), transportContract);
  }

  static CardExecutable
  makeCardExecutable(ExecutionConfig executionConfig,
                     RuntimeLaunchContract runtimeLaunchContract,
                     std::shared_ptr<mlir::MLIRContext> context,
                     std::vector<TileExecutable> tiles,
                     std::unique_ptr<ProgramDataHandoff> programData) {
    return CardExecutable(
        executionConfig, std::move(runtimeLaunchContract), std::move(context),
        std::move(tiles), std::move(programData));
  }
};

namespace detail {

mlir::LogicalResult
verifyExactExecutionConfig(mlir::ModuleOp module,
                           const ExecutionConfig &executionConfig);

llvm::Expected<CardExecutable> buildCardExecutable(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData);

llvm::Expected<CardExecutable> buildCardExecutableWithIRTrace(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace);

} // namespace detail
} // namespace wafer::compiler

#endif // WAFER_COMPILER_CARDEXECUTABLEINTERNAL_H
