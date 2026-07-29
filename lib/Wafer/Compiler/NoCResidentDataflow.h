//===- NoCResidentDataflow.h - All-rank resident tile synthesis -*- C++ -*-===//

#ifndef WAFER_LIB_COMPILER_NOCRESIDENTDATAFLOW_H
#define WAFER_LIB_COMPILER_NOCRESIDENTDATAFLOW_H

#include "WholeVariantCoordinator.h"

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

/// Invocation-local bound on correspondence-valid complete-rank tuples that
/// may seed NoC-resident composition. The reserved baseline consumes one slot.
inline constexpr size_t kNoCResidentCompleteTupleSeedLimit = 8;

/// Resolve one rank-zero static loop to exactly one current-IR occurrence on
/// every rank. Correspondence requires the same structured operation path and
/// exact constant lower/upper/step bounds; ambiguity or any missing rank fails
/// atomically without rewriting a module.
mlir::FailureOr<llvm::SmallVector<mlir::scf::ForOp, 16>>
resolveExactStaticLoopCorrespondence(llvm::ArrayRef<mlir::ModuleOp> modules,
                                     mlir::scf::ForOp anchor);

/// Atomically appends complete-rank NoC-resident dataflow siblings derived
/// from bounded, correspondence-valid Single or StaticFixedSlot actual
/// tuples, with either canonical unplaced or typed worker-placement metadata.
/// Standard-interface-derived actual traversal, input, intermediate,
/// collective-partial, and required-output opportunities are rediscovered
/// from current IR and accumulated in bounded Direct/ReceiveForward clones.
/// On failure no frontier is changed; accepted schedules live only in the
/// rewritten rank IR.
mlir::LogicalResult appendNoCResidentDataflowCandidates(
    std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, std::string *failureReason);

} // namespace wafer::compiler::detail

#endif // WAFER_LIB_COMPILER_NOCRESIDENTDATAFLOW_H
