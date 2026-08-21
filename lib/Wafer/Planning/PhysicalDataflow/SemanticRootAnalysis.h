//===- SemanticRootAnalysis.h - Observable semantic root keys -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEMANTICROOTANALYSIS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEMANTICROOTANALYSIS_H

#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/SemanticRoot.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace wafer::compiler::detail {

/// Query-local association between a stable semantic key and its current IR
/// operation. The operation pointer is only a lookup handle for this immutable
/// IR borrow; it is never part of key equality or ordering.
struct SemanticRootBinding {
  SemanticRootKey key;
  mlir::Operation *operation = nullptr;
};

/// Derives one stable key for every structured DAG node from typed observable
/// SSA paths. Any IR mutation invalidates both the bindings and their operation
/// handles.
class SemanticRootAnalysis {
public:
  static mlir::FailureOr<SemanticRootAnalysis>
  create(const StructuredDAGAnalysis &dag,
         std::string *failureReason = nullptr);

  llvm::ArrayRef<SemanticRootBinding> getRoots() const { return roots; }
  const SemanticRootBinding *find(mlir::Operation *operation) const;
  const SemanticRootBinding *find(const SemanticRootKey &key) const;

private:
  explicit SemanticRootAnalysis(
      llvm::SmallVector<SemanticRootBinding, 16> roots)
      : roots(std::move(roots)) {}

  llvm::SmallVector<SemanticRootBinding, 16> roots;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEMANTICROOTANALYSIS_H
