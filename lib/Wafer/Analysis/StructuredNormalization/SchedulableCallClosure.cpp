//===- SchedulableCallClosure.cpp - Rank-local call closure -------------===//

#include "Wafer/Analysis/SchedulableCallClosure.h"

#include "mlir/Interfaces/CallInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace wafer {
namespace {

static SchedulableCallClosure fail(SchedulableCallClosureStatus status,
                                   SchedulableCallClosureReason reason,
                                   mlir::Operation *operation) {
  SchedulableCallClosure closure;
  closure.status = status;
  closure.reason = reason;
  closure.diagnosticOperation = operation;
  return closure;
}

static bool hasTensorBoundary(mlir::Operation *operation) {
  return llvm::any_of(operation->getOperandTypes(),
                      [](mlir::Type type) {
                        return mlir::isa<mlir::TensorType>(type);
                      }) ||
         llvm::any_of(operation->getResultTypes(), [](mlir::Type type) {
           return mlir::isa<mlir::TensorType>(type);
         });
}

} // namespace

static SchedulableCallClosure
analyzeFromEntries(mlir::ModuleOp module,
                   llvm::ArrayRef<mlir::func::FuncOp> allFunctions,
                   llvm::ArrayRef<mlir::func::FuncOp> entries) {
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::func::FuncOp, 4>>
      callees;
  llvm::DenseMap<mlir::Operation *, SchedulableCallClosure> functionFailures;
  for (mlir::func::FuncOp function : allFunctions) {
    function.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *operation) {
      if (functionFailures.count(function))
        return mlir::WalkResult::interrupt();
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
        mlir::func::FuncOp callee =
            module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
        if (!callee || callee.isDeclaration()) {
          if (hasTensorBoundary(operation))
            functionFailures.try_emplace(
                function,
                fail(SchedulableCallClosureStatus::UnsupportedSemantic,
                     SchedulableCallClosureReason::
                         TensorExternalCallWithoutSummary,
                     operation));
          return functionFailures.count(function)
                     ? mlir::WalkResult::interrupt()
                     : mlir::WalkResult::advance();
        }
        if (!callee.isPrivate() && callee != function) {
          functionFailures.try_emplace(
              function,
              fail(SchedulableCallClosureStatus::InvalidIR,
                   SchedulableCallClosureReason::NonPrivateCallee, operation));
          return mlir::WalkResult::interrupt();
        }
        callees[function].push_back(callee);
        return mlir::WalkResult::advance();
      }
      if (mlir::isa<mlir::CallOpInterface>(operation)) {
        functionFailures.try_emplace(
            function,
            fail(SchedulableCallClosureStatus::UnsupportedSemantic,
                 SchedulableCallClosureReason::IndirectCall, operation));
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
  }

  llvm::DenseMap<mlir::Operation *, uint8_t> state;
  llvm::DenseSet<mlir::Operation *> reachable;
  std::optional<SchedulableCallClosure> failure;
  auto visit = [&](auto &&self, mlir::func::FuncOp function) -> void {
    if (failure)
      return;
    if (auto found = functionFailures.find(function);
        found != functionFailures.end()) {
      failure = found->second;
      return;
    }
    uint8_t &current = state[function];
    if (current == 1) {
      failure = fail(SchedulableCallClosureStatus::UnsupportedSemantic,
                     SchedulableCallClosureReason::RecursiveCall, function);
      return;
    }
    if (current == 2)
      return;
    current = 1;
    reachable.insert(function);
    for (mlir::func::FuncOp callee : callees[function])
      self(self, callee);
    current = 2;
  };
  for (mlir::func::FuncOp entry : entries)
    visit(visit, entry);
  if (failure)
    return *failure;

  SchedulableCallClosure closure;
  closure.status = SchedulableCallClosureStatus::Success;
  if (entries.size() == 1)
    closure.entry = entries.front();
  for (mlir::func::FuncOp function : allFunctions)
    if (reachable.contains(function))
      closure.functions.push_back(function);
  return closure;
}

SchedulableCallClosure
analyzeSchedulableCallClosure(mlir::ModuleOp module,
                              std::optional<llvm::StringRef> expectedEntry) {
  llvm::SmallVector<mlir::func::FuncOp, 8> allFunctions;
  llvm::SmallVector<mlir::func::FuncOp, 2> visibleFunctions;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    allFunctions.push_back(function);
    if (!function.isPrivate())
      visibleFunctions.push_back(function);
  }
  if (allFunctions.empty())
    return fail(SchedulableCallClosureStatus::InvalidIR,
                SchedulableCallClosureReason::NoEntry, module);

  mlir::func::FuncOp entry;
  if (visibleFunctions.size() == 1)
    entry = visibleFunctions.front();
  else if (visibleFunctions.empty() && allFunctions.size() == 1)
    entry = allFunctions.front();
  else
    return fail(SchedulableCallClosureStatus::InvalidIR,
                SchedulableCallClosureReason::MultipleEntries, module);
  if (expectedEntry && entry.getSymName() != *expectedEntry)
    return fail(SchedulableCallClosureStatus::InvalidIR,
                SchedulableCallClosureReason::EntryBindingMismatch, entry);
  return analyzeFromEntries(module, allFunctions,
                            llvm::ArrayRef<mlir::func::FuncOp>(entry));
}

SchedulableCallClosure
analyzeAllSchedulableCallClosures(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::func::FuncOp, 8> allFunctions;
  llvm::SmallVector<mlir::func::FuncOp, 8> entries;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    allFunctions.push_back(function);
    if (!function.isPrivate() && !function.isDeclaration())
      entries.push_back(function);
  }
  if (entries.empty() && allFunctions.size() == 1 &&
      !allFunctions.front().isDeclaration())
    entries.push_back(allFunctions.front());
  if (entries.empty())
    return fail(SchedulableCallClosureStatus::InvalidIR,
                SchedulableCallClosureReason::NoEntry, module);
  return analyzeFromEntries(module, allFunctions, entries);
}

llvm::StringRef
describeSchedulableCallClosureReason(SchedulableCallClosureReason reason) {
  switch (reason) {
  case SchedulableCallClosureReason::NoEntry:
    return "rank-local module has no schedulable entry";
  case SchedulableCallClosureReason::MultipleEntries:
    return "rank-local module has multiple possible schedulable entries";
  case SchedulableCallClosureReason::EntryBindingMismatch:
    return "schedulable entry disagrees with the typed program binding";
  case SchedulableCallClosureReason::NonPrivateCallee:
    return "schedulable path calls a defined non-private helper";
  case SchedulableCallClosureReason::TensorExternalCallWithoutSummary:
    return "tensor external call has no typed semantic/effect summary";
  case SchedulableCallClosureReason::IndirectCall:
    return "schedulable path contains an indirect or non-func call";
  case SchedulableCallClosureReason::RecursiveCall:
    return "schedulable path contains a recursive call cycle";
  }
  return "unknown schedulable call-closure reason";
}

} // namespace wafer
