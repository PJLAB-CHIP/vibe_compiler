//===- SemanticRootAnalysis.h - Observable semantic root keys -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEMANTICROOTANALYSIS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEMANTICROOTANALYSIS_H

#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"
#include "Wafer/Analysis/PhysicalDataflow/SemanticRoot.h"

#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Value.h"

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

struct SemanticValueBinding {
  SemanticRootKey key;
  mlir::Value value;
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
  llvm::ArrayRef<SemanticValueBinding> getValues() const { return values; }
  const SemanticRootBinding *find(mlir::Operation *operation) const;
  const SemanticRootBinding *find(const SemanticRootKey &key) const;
  const SemanticValueBinding *find(mlir::Value value) const;

private:
  explicit SemanticRootAnalysis(
      llvm::SmallVector<SemanticRootBinding, 16> roots,
      llvm::SmallVector<SemanticValueBinding, 32> values)
      : roots(std::move(roots)), values(std::move(values)) {}

  llvm::SmallVector<SemanticRootBinding, 16> roots;
  llvm::SmallVector<SemanticValueBinding, 32> values;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEMANTICROOTANALYSIS_H
