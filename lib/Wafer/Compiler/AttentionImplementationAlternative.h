//===- AttentionImplementationAlternative.h ------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_ATTENTIONIMPLEMENTATIONALTERNATIVE_H
#define WAFER_COMPILER_ATTENTIONIMPLEMENTATIONALTERNATIVE_H

#include "StructuredImplementationAlternative.h"

namespace wafer::compiler::detail {

/// Current-SSA provider for the online-recurrence and functional-decode
/// split-K/V attention implementations. The provider only enumerates legal
/// typed points and materializes one requested point into an isolated clone.
class AttentionImplementationAlternativeProvider final
    : public StructuredImplementationAlternativeProvider {
public:
  llvm::StringRef getStableKey() const final;

  mlir::LogicalResult
  query(mlir::ModuleOp currentStructuredModule,
        const StructuredImplementationAlternativeQuery &query,
        StructuredImplementationAlternativePoints &points,
        std::string *failureReason = nullptr) const final;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_ATTENTIONIMPLEMENTATIONALTERNATIVE_H
