//===- StaticFixedSlotQualification.h - Test qualification companion -*- C++
//-*-===//

#ifndef WAFER_COMPILER_STATICFIXEDSLOTQUALIFICATION_H
#define WAFER_COMPILER_STATICFIXEDSLOTQUALIFICATION_H

#include "Wafer/Compiler/Compilation.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
}

namespace wafer::compiler::detail {

/// Recompute the fixed-slot attestation predicate from one current accepted
/// rank IR. This is shared by qualification selection and companion
/// publication so typed candidate metadata cannot substitute for a static
/// rotating SPM loop and its accepted instruction contracts.
llvm::Error
verifyStaticFixedSlotQualificationEvidence(const RankExecutable &rank);

/// Boolean convenience for selection filters. The detailed verifier above is
/// intentionally non-printing; this wrapper consumes its error so rejected
/// speculative candidates cannot leak diagnostics to stderr.
bool hasStaticFixedSlotQualificationEvidence(const RankExecutable &rank);

/// Recompute the stronger closed companion inventory for one accepted rank.
/// Unlike the existential selection witness, every static loop and planned SPM
/// root in the accepted call closure must be auditable. Proven loop-invariant
/// SPM iter args are retained as non-rotating facts; unknown or conflicting
/// recurrences are rejected.
llvm::Error verifyStaticFixedSlotCompanionEvidence(const RankExecutable &rank);

/// Derive and stage the closed test-only static fixed-slot attestation from
/// the final accepted instruction bundle. The package manifest is consumed
/// only as immutable bytes for its publication digest.
mlir::LogicalResult stageStaticFixedSlotQualificationCompanion(
    llvm::StringRef companionRoot, llvm::StringRef packageRoot,
    const ExecutableBundle &bundle, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_STATICFIXEDSLOTQUALIFICATION_H
