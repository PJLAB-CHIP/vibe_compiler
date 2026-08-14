//===- AcceptedCallClosure.cpp - Accepted executable call graph ----------===//

#include "AcceptedCallClosure.h"

#include "Wafer/Analysis/DirectCallGraphAnalysis.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {
namespace {

llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

} // namespace

llvm::Expected<AcceptedCallClosure>
analyzeAcceptedCallClosure(mlir::ModuleOp module,
                           std::optional<llvm::StringRef> expectedEntry) {
  AcceptedCallClosure closure;
  analysis::DirectCallGraphAnalysis callGraph(module.getOperation());
  if (!callGraph.hasModuleScope())
    return invalid("accepted call closure requires a module");
  closure.functions.append(callGraph.getFunctions().begin(),
                           callGraph.getFunctions().end());
  if (closure.functions.empty())
    return invalid("accepted physical-Tile executable has no entry function");

  llvm::SmallVector<mlir::func::FuncOp> externallyVisible;
  for (mlir::func::FuncOp function : closure.functions) {
    if (function.isDeclaration())
      return invalid(("accepted call closure contains external function @" +
                      function.getSymName())
                         .str());
    if (!function.isPrivate())
      externallyVisible.push_back(function);
  }
  if (externallyVisible.size() == 1) {
    closure.entry = externallyVisible.front();
  } else {
    return invalid(
        "accepted physical-Tile executable must have one externally visible "
        "entry function");
  }
  if (expectedEntry && closure.entry.getSymName() != *expectedEntry)
    return invalid("accepted entry symbol disagrees with the typed artifact");

  for (mlir::func::FuncOp function : closure.functions) {
    if (function != closure.entry && !function.isPrivate())
      return invalid(
          ("non-entry function @" + function.getSymName() + " must be private")
              .str());
    for (auto [index, type] :
         llvm::enumerate(function.getFunctionType().getInputs()))
      if (!wafer::isWaferDDRMemRefType(type))
        return invalid(("accepted function @" + function.getSymName() +
                        " argument #" + std::to_string(index) +
                        " is not a Wafer DDR memref")
                           .str());
    for (auto [index, type] :
         llvm::enumerate(function.getFunctionType().getResults()))
      if (!wafer::isWaferDDRMemRefType(type))
        return invalid(("accepted function @" + function.getSymName() +
                        " result #" + std::to_string(index) +
                        " is not a Wafer DDR memref")
                           .str());

    if (function == closure.entry)
      continue;
    mlir::memref::AllocOp privateDDRAllocation;
    function.walk([&](mlir::memref::AllocOp allocation) {
      if (!privateDDRAllocation &&
          wafer::isWaferDDRMemRefType(allocation.getType()))
        privateDDRAllocation = allocation;
    });
    if (privateDDRAllocation)
      return invalid(("private helper @" + function.getSymName() +
                      " cannot own compiler-managed DDR storage")
                         .str());
  }

  if (!callGraph.getUnresolvedCalls().empty()) {
    mlir::func::CallOp call = callGraph.getUnresolvedCalls().front();
    mlir::func::FuncOp caller = call->getParentOfType<mlir::func::FuncOp>();
    return invalid(("direct call from @" + caller.getSymName() +
                    " has unresolved callee @" + call.getCallee())
                       .str());
  }
  if (!callGraph.getUnsupportedCallOperations().empty()) {
    mlir::Operation *operation =
        callGraph.getUnsupportedCallOperations().front();
    return invalid(("accepted call closure contains unsupported indirect or "
                    "non-func call operation " +
                    operation->getName().getStringRef())
                       .str());
  }
  if (callGraph.hasRecursiveCycle()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(
        callGraph.getRecursiveCycleOrigin());
    return invalid(("accepted call closure is recursive at @" +
                    function.getSymName())
                       .str());
  }

  llvm::DenseMap<mlir::Operation *, uint8_t> state;
  llvm::DenseSet<mlir::Operation *> reachable;
  auto visit = [&](auto &&self, mlir::func::FuncOp function) -> llvm::Error {
    uint8_t &current = state[function.getOperation()];
    if (current == 1)
      return invalid(
          ("accepted call closure is recursive at @" + function.getSymName())
              .str());
    if (current == 2)
      return llvm::Error::success();
    current = 1;
    reachable.insert(function.getOperation());
    for (mlir::func::FuncOp callee : callGraph.getCallees(function))
      if (llvm::Error error = self(self, callee))
        return error;
    current = 2;
    return llvm::Error::success();
  };
  if (llvm::Error error = visit(visit, closure.entry))
    return std::move(error);
  for (mlir::func::FuncOp function : closure.functions)
    if (!reachable.contains(function.getOperation()))
      return invalid(("private function @" + function.getSymName() +
                      " is outside the entry call closure")
                         .str());
  return closure;
}

} // namespace wafer::compiler::detail
