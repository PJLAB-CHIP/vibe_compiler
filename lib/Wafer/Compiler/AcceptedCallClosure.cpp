//===- AcceptedCallClosure.cpp - Accepted executable call graph ----------===//

#include "AcceptedCallClosure.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/CallInterfaces.h"

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
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    closure.functions.push_back(function);
  if (closure.functions.empty())
    return invalid("accepted rank has no entry function");

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
  } else if (externallyVisible.empty() && closure.functions.size() == 1) {
    closure.entry = closure.functions.front();
  } else {
    return invalid(
        "accepted rank must have one externally visible entry function");
  }
  if (expectedEntry && closure.entry.getSymName() != *expectedEntry)
    return invalid("accepted entry symbol disagrees with the typed artifact");

  llvm::DenseSet<mlir::Operation *> functionSet;
  for (mlir::func::FuncOp function : closure.functions) {
    functionSet.insert(function.getOperation());
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

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::func::FuncOp, 2>>
      callees;
  for (mlir::func::FuncOp function : closure.functions) {
    llvm::Error error = llvm::Error::success();
    function.walk([&](mlir::Operation *operation) {
      if (error)
        return mlir::WalkResult::interrupt();
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
        mlir::func::FuncOp callee =
            module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
        if (!callee || !functionSet.contains(callee.getOperation())) {
          error = invalid(("direct call from @" + function.getSymName() +
                           " has unresolved callee @" + call.getCallee())
                              .str());
          return mlir::WalkResult::interrupt();
        }
        callees[function.getOperation()].push_back(callee);
        return mlir::WalkResult::advance();
      }
      if (mlir::isa<mlir::CallOpInterface>(operation)) {
        error = invalid(("accepted call closure contains unsupported indirect "
                         "or non-func call operation " +
                         operation->getName().getStringRef())
                            .str());
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (error)
      return std::move(error);
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
    for (mlir::func::FuncOp callee : callees[function.getOperation()])
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
