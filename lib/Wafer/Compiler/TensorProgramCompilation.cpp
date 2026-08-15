//===- TensorProgramCompilation.cpp - Tensor-program compilation --------===//

#include "CompilationInternal.h"
#include "PhysicalTileExecutablesInternal.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

namespace wafer::compiler::detail {

template <typename ProductT, typename BuilderT>
static llvm::Expected<ProductT> compileTensorProgram(
    llvm::StringRef tensorProgramDirectory, ExecutionConfig executionConfig,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    BuilderT &&builder) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    reject(diagnostics, message);
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };
  if (tensorProgramDirectory.empty())
    return fail("tensor program directory must not be empty");
  if (!isDirectory(tensorProgramDirectory))
    return fail("tensor program path is not a directory");
  if (validateRegularDirectoryTree(tensorProgramDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "tensor program tree is not regular");

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

  mlir::OwningOpRef<mlir::ModuleOp> tensorModule =
      parseProgramDirectoryModule(tensorProgramDirectory, *context);
  if (!tensorModule)
    return fail("failed to parse verified tensor program module");
  if (mlir::failed(
          verifyExactExecutionConfigInternal(*tensorModule, executionConfig)))
    return fail("tensor program does not match ExecutionConfig");
  if (!hasPostSpmdMarker(*tensorModule, tensorProgramDirectory))
    return fail("tensor program is missing its post-SPMD marker");
  if (containsDialectSemantics(*tensorModule, "stablehlo") ||
      containsDialectSemantics(*tensorModule, "sdy") ||
      mlir::failed(verifyTensorProgramStageOperations(*tensorModule)))
    return fail("input is not a verified structured tensor program");

  frontend::FrontendProgramVerificationResult program;
  if (mlir::failed(verifyProgramDirectoryMetadata(
          *tensorModule, tensorProgramDirectory, diagnostics, &program)))
    return fail("tensor program metadata verification failed");
  if (program.numPartitions != executionConfig.getNumPartitions())
    return fail(
        "typed program card-partition domain does not match ExecutionConfig");
  if (program.parameters.size() != program.programParameterCount ||
      program.constants.size() != program.programConstantCount)
    return fail("typed program resources do not cover all parameters and "
                "constants");

  return builder(context, *tensorModule, std::move(program), executionConfig,
                 diagnostics, failAfterLaunchSlot);
}

llvm::Expected<PhysicalTileExecutables>
compileTensorProgramToPhysicalTileExecutables(
    llvm::StringRef tensorProgramDirectory, ExecutionConfig executionConfig,
    OptimizationConfig optimizations, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot, CompilationIRTrace &irTrace) {
  return compileTensorProgram<PhysicalTileExecutables>(
      tensorProgramDirectory, executionConfig, diagnostics, failAfterLaunchSlot,
      [optimizations,
       &irTrace](std::shared_ptr<mlir::MLIRContext> &context,
                 mlir::ModuleOp tensorModule,
                 frontend::FrontendProgramVerificationResult program,
                 ExecutionConfig config, llvm::raw_ostream &output,
                 std::optional<int64_t> failAfterLaunchSlot) {
        return buildPhysicalTileExecutablesWithIRTrace(
            context, tensorModule, std::move(program), config, optimizations,
            output, failAfterLaunchSlot, irTrace);
      });
}

} // namespace wafer::compiler::detail
