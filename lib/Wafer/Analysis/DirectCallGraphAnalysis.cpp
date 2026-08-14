//===- DirectCallGraphAnalysis.cpp - Typed func.call graph facts ---------===//

#include "Wafer/Analysis/DirectCallGraphAnalysis.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"

namespace wafer::analysis {

DirectCallGraphAnalysis::DirectCallGraphAnalysis(mlir::Operation *scope)
    : module(mlir::dyn_cast_or_null<mlir::ModuleOp>(scope)) {
  if (!module)
    return;

  mlir::SymbolTableCollection symbolTables;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    functions.push_back(function);

  module.walk([&](mlir::Operation *operation) {
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
      if (mlir::func::FuncOp caller =
              call->getParentOfType<mlir::func::FuncOp>())
        callsByFunction[caller.getOperation()].push_back(call);
      mlir::func::FuncOp callee =
          symbolTables.lookupNearestSymbolFrom<mlir::func::FuncOp>(
              call, call.getCalleeAttr());
      if (!callee || callee->getParentOp() != module.getOperation()) {
        unresolvedCalls.push_back(call);
        return;
      }
      calleesByCall.try_emplace(call.getOperation(), callee);
      if (mlir::func::FuncOp caller =
              call->getParentOfType<mlir::func::FuncOp>())
        calleesByFunction[caller.getOperation()].push_back(callee);
      calledFunctions.insert(callee.getOperation());
      return;
    }
    if (mlir::isa<mlir::CallOpInterface>(operation))
      unsupportedCallOperations.push_back(operation);
  });

  llvm::DenseMap<mlir::Operation *, unsigned> visitState;
  auto visit = [&](auto &&self, mlir::func::FuncOp function) -> void {
    unsigned &state = visitState[function.getOperation()];
    if (state == 2)
      return;
    if (state == 1) {
      if (!recursiveCycleOrigin)
        recursiveCycleOrigin = function.getOperation();
      return;
    }
    state = 1;
    for (mlir::func::FuncOp callee : getCallees(function)) {
      if (visitState[callee.getOperation()] == 1 && !recursiveCycleOrigin)
        recursiveCycleOrigin = callee.getOperation();
      self(self, callee);
    }
    state = 2;
    calleeFirstOrder.push_back(function);
  };
  for (mlir::func::FuncOp function : functions)
    visit(visit, function);
}

llvm::ArrayRef<mlir::func::CallOp>
DirectCallGraphAnalysis::getCalls(mlir::func::FuncOp caller) const {
  auto found = callsByFunction.find(caller.getOperation());
  return found == callsByFunction.end()
             ? llvm::ArrayRef<mlir::func::CallOp>{}
             : llvm::ArrayRef<mlir::func::CallOp>(found->second);
}

mlir::func::FuncOp
DirectCallGraphAnalysis::getCallee(mlir::func::CallOp call) const {
  auto found = calleesByCall.find(call.getOperation());
  return found == calleesByCall.end() ? mlir::func::FuncOp{} : found->second;
}

llvm::ArrayRef<mlir::func::FuncOp>
DirectCallGraphAnalysis::getCallees(mlir::func::FuncOp caller) const {
  auto found = calleesByFunction.find(caller.getOperation());
  return found == calleesByFunction.end()
             ? llvm::ArrayRef<mlir::func::FuncOp>{}
             : llvm::ArrayRef<mlir::func::FuncOp>(found->second);
}

llvm::SmallVector<mlir::func::FuncOp, 2>
DirectCallGraphAnalysis::getRootFunctions() const {
  llvm::SmallVector<mlir::func::FuncOp, 2> roots;
  for (mlir::func::FuncOp function : functions)
    if (!isCalled(function))
      roots.push_back(function);
  return roots;
}

} // namespace wafer::analysis
