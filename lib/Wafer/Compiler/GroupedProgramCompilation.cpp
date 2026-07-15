//===- GroupedProgramCompilation.cpp - Grouped artifact compilation -----===//

#include "CompilationInternal.h"
#include "ExecutableBundleInternal.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

namespace wafer::compiler::detail {

llvm::Expected<ExecutableBundle> compileGroupedProgramToExecutableBundleImpl(
    llvm::StringRef groupedProgramDirectory, ExecutionConfig executionConfig,
    llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    reject(diagnostics, message);
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };
  if (groupedProgramDirectory.empty())
    return fail("grouped program directory must not be empty");
  if (!isDirectory(groupedProgramDirectory))
    return fail("grouped program path is not a directory");
  if (validateRegularDirectoryTree(groupedProgramDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "grouped program tree is not regular");

  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        diagnostic.print(diagnostics);
        diagnostics << "\n";
        return mlir::success();
      });

  mlir::OwningOpRef<mlir::ModuleOp> groupedModule =
      parseProgramDirectoryModule(groupedProgramDirectory, *context);
  if (!groupedModule)
    return fail("failed to parse verified grouped program module");
  if (mlir::failed(
          verifyExactExecutionConfigInternal(*groupedModule, executionConfig)))
    return fail("grouped program does not match ExecutionConfig");
  if (!hasPostSpmdMarker(*groupedModule, groupedProgramDirectory))
    return fail("grouped program is missing its post-SPMD marker");
  if (containsDialectSemantics(*groupedModule, "stablehlo") ||
      containsDialectSemantics(*groupedModule, "sdy") ||
      mlir::failed(verifyGroupedStageOperations(*groupedModule)))
    return fail("input is not a verified grouped-program artifact");

  frontend::FrontendProgramVerificationResult program;
  if (mlir::failed(verifyProgramDirectoryMetadata(
          *groupedModule, groupedProgramDirectory, diagnostics, &program)))
    return fail("grouped program metadata verification failed");
  if (program.logicalRankCount != executionConfig.getRankCount())
    return fail("typed program rank domain does not match ExecutionConfig");
  if (program.parameters.size() != program.programParameterCount ||
      program.constants.size() != program.programConstantCount)
    return fail("typed program resources do not cover all parameters and "
                "constants");

  return detail::buildExecutableBundle(context, *groupedModule,
                                       std::move(program), executionConfig,
                                       diagnostics, failAfterLogicalRank);
}

} // namespace wafer::compiler::detail
