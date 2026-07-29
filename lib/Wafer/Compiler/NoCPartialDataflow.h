//===- NoCPartialDataflow.h - Typed partial/reduction residency -*- C++ -*-===//

#ifndef WAFER_LIB_COMPILER_NOCPARTIALDATAFLOW_H
#define WAFER_LIB_COMPILER_NOCPARTIALDATAFLOW_H

#include "Wafer/Frontend/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Removes a complete-rank partial-result DDR spill/reload cut only when the
/// current instruction IR proves that every reloaded value feeds one explicit
/// typed ordered-tree or ring all-reduce and reaches the verified output
/// publisher.
///
/// The proof is derived from DTE message phases, typed elementwise combiners,
/// SSA/memory effects, and frontend output tile relations.  No logical
/// collective is invented.  A rejected or incomplete proposal leaves every
/// module unchanged.  The caller owns tuple cloning and the subsequent
/// completion, SPM, Direct-DTE, and whole-variant resource gates.
///
/// Ring admission additionally binds every message payload slice to the actual
/// root-relative send/receive subview, proves reduce-scatter origin
/// multiplicity, and requires the final slice set to exactly cover the
/// publisher without gaps or overlap.
unsigned materializeNoCPartialReductions(
    llvm::MutableArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program);

} // namespace wafer::compiler::detail

#endif // WAFER_LIB_COMPILER_NOCPARTIALDATAFLOW_H
