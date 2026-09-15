//===- InlineConstantData.h - Own tensor literals ----------------------===//

#ifndef WAFER_DRIVER_INLINECONSTANTDATA_H
#define WAFER_DRIVER_INLINECONSTANTDATA_H
#include "Wafer/Driver/ProgramData/ProgramData.h"
#include "mlir/IR/BuiltinOps.h"
namespace wafer::compiler::detail {
/// Moves live nonsplat tensor literals into the existing constant data ABI.
/// The current function and its typed metadata are updated together; the
/// handoff owns every payload before an argument can refer to it.
llvm::Error
outlineInlineConstantData(mlir::ModuleOp module,
                          frontend::FrontendProgramVerificationResult &program,
                          ProgramDataHandoff &data);
} // namespace wafer::compiler::detail
#endif
