//===- NoCPartialDataflow.h - Typed partial/reduction residency -*- C++ -*-===//

#ifndef WAFER_LIB_COMPILER_NOCPARTIALDATAFLOW_H
#define WAFER_LIB_COMPILER_NOCPARTIALDATAFLOW_H

#include "Wafer/Frontend/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Read-only capability query over current all-rank canonical Instr SSA. It
/// performs the same complete protocol and reduction-value proof as
/// materialization and retains no Operation pointers after returning.
bool hasNoCPartialReductionOpportunity(
    llvm::ArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program);

/// Removes a complete-rank partial-result DDR spill/reload cut only when the
/// current instruction IR proves that every reloaded value feeds one explicit
/// typed ordered-tree or ring all-reduce and reaches the verified output
/// writer.
///
/// The proof is derived from DTE message phases, typed elementwise combiners,
/// SSA/memory effects, and frontend output tile relations.  No logical
/// collective is invented.  A rejected or incomplete proposal leaves every
/// module unchanged.  The caller owns tuple cloning and the subsequent
/// completion, SPM, Direct-DTE, and rank-candidate resource gates.
///
/// Ring matching additionally binds every message payload slice to the actual
/// root-relative send/receive subview, proves reduce-scatter origin
/// multiplicity, and requires the final slice set to exactly cover the
/// writer without gaps or overlap.
unsigned materializeNoCPartialReductions(
    llvm::MutableArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program);

} // namespace wafer::compiler::detail

#endif // WAFER_LIB_COMPILER_NOCPARTIALDATAFLOW_H
