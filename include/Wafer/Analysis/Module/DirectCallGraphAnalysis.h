//===- DirectCallGraphAnalysis.h - Typed func.call graph facts -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_DIRECTCALLGRAPHANALYSIS_H
#define WAFER_ANALYSIS_DIRECTCALLGRAPHANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer::analysis {

/// Operation-anchored facts for direct `func.call` relations in a module.
///
/// The analysis records structure rather than imposing a stage policy:
/// unresolved calls, non-`func.call` call-like operations, external
/// declarations, recursion, visibility, ABI, and reachability are exposed for
/// consumers to classify according to their own IR contract.
class DirectCallGraphAnalysis {
public:
  explicit DirectCallGraphAnalysis(mlir::Operation *scope);

  bool hasModuleScope() const { return static_cast<bool>(module); }
  mlir::ModuleOp getModule() const { return module; }

  llvm::ArrayRef<mlir::func::FuncOp> getFunctions() const { return functions; }
  llvm::ArrayRef<mlir::func::CallOp> getCalls(mlir::func::FuncOp caller) const;
  mlir::func::FuncOp getCallee(mlir::func::CallOp call) const;
  llvm::ArrayRef<mlir::func::FuncOp>
  getCallees(mlir::func::FuncOp caller) const;

  llvm::ArrayRef<mlir::func::CallOp> getUnresolvedCalls() const {
    return unresolvedCalls;
  }
  llvm::ArrayRef<mlir::Operation *> getUnsupportedCallOperations() const {
    return unsupportedCallOperations;
  }
  bool hasRecursiveCycle() const { return recursiveCycleOrigin != nullptr; }
  mlir::Operation *getRecursiveCycleOrigin() const {
    return recursiveCycleOrigin;
  }

  llvm::ArrayRef<mlir::func::FuncOp> getCalleeFirstOrder() const {
    return calleeFirstOrder;
  }
  bool isCalled(mlir::func::FuncOp function) const {
    return calledFunctions.contains(function.getOperation());
  }
  llvm::SmallVector<mlir::func::FuncOp, 2> getRootFunctions() const;

private:
  mlir::ModuleOp module;
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::func::CallOp, 4>>
      callsByFunction;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::func::FuncOp, 4>>
      calleesByFunction;
  llvm::DenseMap<mlir::Operation *, mlir::func::FuncOp> calleesByCall;
  llvm::DenseSet<mlir::Operation *> calledFunctions;
  llvm::SmallVector<mlir::func::CallOp, 4> unresolvedCalls;
  llvm::SmallVector<mlir::Operation *, 4> unsupportedCallOperations;
  llvm::SmallVector<mlir::func::FuncOp, 8> calleeFirstOrder;
  mlir::Operation *recursiveCycleOrigin = nullptr;
};

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_DIRECTCALLGRAPHANALYSIS_H
