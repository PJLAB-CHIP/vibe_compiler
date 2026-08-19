//===- DirectDTETransport.h - Physical Direct DTE binding -----*- C++ -*-===//

#ifndef WAFER_COMPILER_DIRECTDTETRANSPORT_H
#define WAFER_COMPILER_DIRECTDTETRANSPORT_H

#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Recomputes logical message matching, structured dynamic occurrences and the
/// card Direct-DTE wait graph from the supplied Tile modules.
/// This validation does not attach physical bindings or retain analysis state.
mlir::LogicalResult verifyDirectDTETransportSchedule(
    llvm::ArrayRef<mlir::ModuleOp> tileModules);

/// Matches and validates every logical Direct DTE issue across the supplied
/// Tile domain, then writes typed physical bindings into the modules.
/// Analysis completes for the full domain before any binding is written.
mlir::FailureOr<TransportContract>
bindDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> tileModules);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_DIRECTDTETRANSPORT_H
