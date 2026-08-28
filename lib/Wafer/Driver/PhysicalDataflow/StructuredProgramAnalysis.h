//===- StructuredProgramAnalysis.h - Driver planning input adapter -*- C++ -*-===//

#ifndef WAFER_COMPILER_DRIVER_PHYSICALDATAFLOW_STRUCTUREDPROGRAMANALYSIS_H
#define WAFER_COMPILER_DRIVER_PHYSICALDATAFLOW_STRUCTUREDPROGRAMANALYSIS_H

#include "Wafer/Driver/Compilation.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredProgramAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace llvm {
class raw_ostream;
}

namespace wafer::compiler::detail {

mlir::FailureOr<std::unique_ptr<StructuredProgramAnalysis>>
analyzeStructuredProgram(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_DRIVER_PHYSICALDATAFLOW_STRUCTUREDPROGRAMANALYSIS_H
