//===- DirectDTETransport.h - Physical Direct DTE acceptance -*- C++ -*-===//

#ifndef WAFER_COMPILER_DIRECTDTETRANSPORT_H
#define WAFER_COMPILER_DIRECTDTETRANSPORT_H

#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Matches and validates every logical Direct DTE issue across the complete
/// rank domain, then writes typed physical bindings into the candidate modules.
/// No binding is externally observable unless the caller subsequently commits
/// the whole ExecutableBundle.
mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> rankModules);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_DIRECTDTETRANSPORT_H
