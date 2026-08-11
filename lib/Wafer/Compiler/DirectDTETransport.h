//===- DirectDTETransport.h - Physical Direct DTE acceptance -*- C++ -*-===//

#ifndef WAFER_COMPILER_DIRECTDTETRANSPORT_H
#define WAFER_COMPILER_DIRECTDTETRANSPORT_H

#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Recomputes logical message matching, structured dynamic occurrences and the
/// whole-card Direct-DTE wait graph from the supplied physical Tile modules.
/// This validation does not attach physical bindings or retain analysis state.
mlir::LogicalResult verifyDirectDTETransportSchedule(
    llvm::ArrayRef<mlir::ModuleOp> physicalTileModules);

/// Matches and validates every logical Direct DTE issue across the supplied
/// physical Tile domain, then writes typed physical bindings into the modules.
/// No binding is externally observable unless the caller subsequently commits
/// the whole ExecutableBundle.
mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_DIRECTDTETRANSPORT_H
