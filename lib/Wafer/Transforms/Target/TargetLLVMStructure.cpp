//===- Target LLVM lowering implementation -------------------------------===//

#include "Target/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetFormat.h"
#include "Wafer/Transforms/TargetConversion.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace wafer::target_llvm_detail {

namespace {
static bool isWaferTargetMetadata(mlir::Operation *op) {
  return mlir::isa<TargetTopologyOp, ExecutionMeshOp>(op);
}
} // namespace

mlir::LogicalResult flattenTileRegions(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<TileRegionOp, 8> tileRegions;
  moduleOp.walk(
      [&](TileRegionOp tileRegion) { tileRegions.push_back(tileRegion); });

  mlir::IRRewriter rewriter(moduleOp.getContext());
  for (TileRegionOp tileRegion : llvm::reverse(tileRegions)) {
    if (!tileRegion.getBody().hasOneBlock())
      return tileRegion.emitError()
             << "unsupported_target_structure: wafer.tile.region must have "
                "exactly one block";
    mlir::Block &body = tileRegion.getBody().front();
    auto yield = mlir::dyn_cast<TileYieldOp>(body.getTerminator());
    if (!yield)
      return tileRegion.emitError()
             << "unsupported_target_structure: wafer.tile.region must end "
                "with wafer.tile.yield";

    llvm::SmallVector<mlir::Value, 4> yieldedValues;
    for (mlir::Value value : yield.getValues()) {
      auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
      if (blockArg && blockArg.getOwner() == &body) {
        if (blockArg.getArgNumber() >= tileRegion.getInputs().size())
          return tileRegion.emitError()
                 << "unsupported_target_structure: tile yield block argument "
                    "has no matching boundary input";
        value = tileRegion.getInputs()[blockArg.getArgNumber()];
      }
      yieldedValues.push_back(value);
    }
    if (yieldedValues.size() != tileRegion.getNumResults())
      return tileRegion.emitError()
             << "unsupported_target_structure: tile yield/result count "
                "mismatch";

    rewriter.inlineBlockBefore(&body, tileRegion.getOperation(),
                               tileRegion.getInputs());
    rewriter.eraseOp(yield);
    for (auto [result, replacement] :
         llvm::zip_equal(tileRegion.getResults(), yieldedValues))
      result.replaceAllUsesWith(replacement);
    rewriter.eraseOp(tileRegion);
  }
  return mlir::success();
}
mlir::LogicalResult
analyzeDirectCallGraph(mlir::ModuleOp moduleOp, DirectCallGraph &graph,
                       int64_t defaultDDRArenaArgumentIndex,
                       int64_t transportStatusArgumentIndex) {
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  for (mlir::func::FuncOp funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
    if (funcOp.isDeclaration())
      return funcOp.emitError()
             << "unsupported_target_call: external func.func declarations "
                "are not accepted by target LLVM lowering";
    functions.push_back(funcOp);

    for (auto [index, type] :
         llvm::enumerate(funcOp.getFunctionType().getInputs()))
      if (!isWaferDDRMemRefType(type) &&
          !(static_cast<int64_t>(index) == defaultDDRArenaArgumentIndex &&
            type.isInteger(64)) &&
          !(static_cast<int64_t>(index) == transportStatusArgumentIndex &&
            type.isInteger(64)))
        return funcOp.emitError()
               << "unsupported_target_function: argument #" << index
               << " must be a Wafer DDR memref target binding";

    mlir::WalkResult result =
        funcOp.walk([&](mlir::Operation *op) -> mlir::WalkResult {
          if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            mlir::func::FuncOp callee =
                moduleOp.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
            if (!callee) {
              call.emitError()
                  << "unsupported_target_call: unresolved direct callee @"
                  << call.getCallee();
              return mlir::WalkResult::interrupt();
            }
            graph.calls[funcOp.getOperation()].push_back(call);
            graph.calledFunctions.insert(callee.getOperation());
            return mlir::WalkResult::advance();
          }
          if (mlir::isa<mlir::CallOpInterface>(op)) {
            op->emitError()
                << "unsupported_target_call: indirect or unknown call-like "
                   "operation '"
                << op->getName() << "' is not supported";
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted())
      return mlir::failure();
  }

  llvm::DenseMap<mlir::Operation *, unsigned> state;
  auto visit = [&](auto &&self,
                   mlir::func::FuncOp funcOp) -> mlir::LogicalResult {
    unsigned &currentState = state[funcOp.getOperation()];
    if (currentState == 2)
      return mlir::success();
    if (currentState == 1)
      return funcOp.emitError()
             << "unsupported_target_call: recursive direct call graph is not "
                "supported";
    currentState = 1;
    for (mlir::func::CallOp call : graph.calls[funcOp.getOperation()]) {
      mlir::func::FuncOp callee =
          moduleOp.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (state[callee.getOperation()] == 1)
        return call.emitError()
               << "unsupported_target_call: recursive call to @"
               << call.getCallee() << " is not supported";
      if (mlir::failed(self(self, callee)))
        return mlir::failure();
    }
    currentState = 2;
    graph.calleeFirstOrder.push_back(funcOp);
    return mlir::success();
  };

  for (mlir::func::FuncOp funcOp : functions)
    if (mlir::failed(visit(visit, funcOp)))
      return mlir::failure();
  return mlir::success();
}

mlir::FailureOr<mlir::func::FuncOp>
findUniqueRootFunction(mlir::ModuleOp moduleOp, const DirectCallGraph &graph) {
  mlir::func::FuncOp root;
  for (mlir::func::FuncOp function : moduleOp.getOps<mlir::func::FuncOp>()) {
    if (graph.calledFunctions.contains(function.getOperation()))
      continue;
    if (root)
      return function.emitError()
             << "unsupported_target_transport: Direct DTE status ABI "
                "requires one unique entry function";
    root = function;
  }
  if (!root)
    return moduleOp.emitError()
           << "unsupported_target_transport: Direct DTE status ABI has no "
              "entry function";
  return root;
}

static mlir::FailureOr<unsigned> resolveDDRAliasToFunctionArgument(
    mlir::Value value, mlir::func::FuncOp funcOp, mlir::ModuleOp moduleOp,
    const llvm::DenseMap<mlir::Operation *, AliasSummary> &summaries,
    llvm::DenseSet<mlir::Value> &visiting) {
  while (true) {
    mlir::Value root = resolveReturnedMemRefRoot(value);
    if (root != value) {
      value = root;
      continue;
    }
    if (auto castOp = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = castOp.getSource();
      continue;
    }
    break;
  }

  if (!visiting.insert(value).second)
    return mlir::failure();
  auto eraseVisiting = llvm::make_scope_exit([&] { visiting.erase(value); });

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = blockArg.getOwner();
    if (owner == &funcOp.getBody().front()) {
      if (!isWaferDDRMemRefType(blockArg.getType()))
        return mlir::failure();
      return blockArg.getArgNumber();
    }

    std::optional<unsigned> commonAlias;
    bool sawPredecessor = false;
    for (mlir::Block *predecessor : owner->getPredecessors()) {
      auto branch =
          mlir::dyn_cast<mlir::BranchOpInterface>(predecessor->getTerminator());
      if (!branch)
        return mlir::failure();
      for (unsigned successorIndex = 0,
                    end = predecessor->getTerminator()->getNumSuccessors();
           successorIndex < end; ++successorIndex) {
        if (predecessor->getTerminator()->getSuccessor(successorIndex) != owner)
          continue;
        mlir::SuccessorOperands successorOperands =
            branch.getSuccessorOperands(successorIndex);
        if (blockArg.getArgNumber() >= successorOperands.size())
          return mlir::failure();
        mlir::Value incoming = successorOperands[blockArg.getArgNumber()];
        if (!incoming)
          return mlir::failure();
        if (incoming == value) {
          sawPredecessor = true;
          continue;
        }
        llvm::DenseSet<mlir::Value> branchVisiting = visiting;
        mlir::FailureOr<unsigned> alias = resolveDDRAliasToFunctionArgument(
            incoming, funcOp, moduleOp, summaries, branchVisiting);
        if (mlir::failed(alias) || (commonAlias && *commonAlias != *alias))
          return mlir::failure();
        commonAlias = *alias;
        sawPredecessor = true;
      }
    }
    if (sawPredecessor && commonAlias)
      return *commonAlias;
    return mlir::failure();
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return mlir::failure();
  auto call = mlir::dyn_cast<mlir::func::CallOp>(result.getOwner());
  if (!call)
    return mlir::failure();
  mlir::func::FuncOp callee =
      moduleOp.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return mlir::failure();
  auto summaryIt = summaries.find(callee.getOperation());
  if (summaryIt == summaries.end() ||
      result.getResultNumber() >= summaryIt->second.size())
    return mlir::failure();
  unsigned calleeArg = summaryIt->second[result.getResultNumber()];
  if (calleeArg >= call.getNumOperands())
    return mlir::failure();
  return resolveDDRAliasToFunctionArgument(call.getOperand(calleeArg), funcOp,
                                           moduleOp, summaries, visiting);
}

mlir::LogicalResult analyzeDDRAliasContracts(
    mlir::ModuleOp moduleOp, const DirectCallGraph &graph,
    llvm::DenseMap<mlir::Operation *, AliasSummary> &summaries) {
  for (mlir::func::FuncOp funcOp : graph.calleeFirstOrder) {
    unsigned resultCount = funcOp.getFunctionType().getNumResults();
    AliasSummary summary(resultCount);
    if (resultCount == 0) {
      summaries[funcOp.getOperation()] = std::move(summary);
      continue;
    }

    for (auto [index, type] :
         llvm::enumerate(funcOp.getFunctionType().getResults()))
      if (!isWaferDDRMemRefType(type))
        return funcOp.emitError()
               << "unsupported_target_alias: function result #" << index
               << " must be a Wafer DDR memref";

    llvm::SmallVector<std::optional<unsigned>, 4> aliases(resultCount);
    unsigned returnCount = 0;
    mlir::LogicalResult valid = mlir::success();
    funcOp.walk([&](mlir::func::ReturnOp returnOp) {
      if (mlir::failed(valid) ||
          returnOp->getParentOfType<mlir::func::FuncOp>() != funcOp)
        return;
      ++returnCount;
      if (returnOp.getNumOperands() != resultCount) {
        valid = returnOp.emitError()
                << "unsupported_target_alias: return operand count does not "
                   "match function result count";
        return;
      }
      for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
        llvm::DenseSet<mlir::Value> visiting;
        mlir::FailureOr<unsigned> alias = resolveDDRAliasToFunctionArgument(
            operand, funcOp, moduleOp, summaries, visiting);
        if (mlir::failed(alias)) {
          valid = returnOp.emitError()
                  << "unsupported_target_alias: DDR result #" << index
                  << " must provably alias one DDR function argument through "
                     "views, CFG forwarding, or direct-call alias results";
          return;
        }
        if (aliases[index] && *aliases[index] != *alias) {
          valid = returnOp.emitError()
                  << "unsupported_target_alias: DDR result #" << index
                  << " aliases different function arguments on different "
                     "return paths";
          return;
        }
        aliases[index] = *alias;
      }
    });
    if (mlir::failed(valid))
      return mlir::failure();
    if (returnCount == 0)
      return funcOp.emitError()
             << "unsupported_target_alias: result-bearing function has no "
                "func.return";
    for (unsigned index = 0; index < resultCount; ++index) {
      if (!aliases[index])
        return funcOp.emitError()
               << "unsupported_target_alias: could not prove result #" << index
               << " alias";
      summary[index] = *aliases[index];
    }
    summaries[funcOp.getOperation()] = std::move(summary);
  }
  return mlir::success();
}

void dropRootAliasResults(mlir::ModuleOp moduleOp,
                          const DirectCallGraph &graph) {
  for (mlir::func::FuncOp funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
    if (graph.calledFunctions.contains(funcOp.getOperation()) ||
        funcOp.getFunctionType().getNumResults() == 0)
      continue;
    funcOp.setFunctionType(mlir::FunctionType::get(
        moduleOp.getContext(), funcOp.getFunctionType().getInputs(), {}));
    funcOp.removeResAttrsAttr();
    funcOp.walk([&](mlir::func::ReturnOp returnOp) {
      if (returnOp->getParentOfType<mlir::func::FuncOp>() == funcOp)
        returnOp->setOperands({});
    });
  }
}

void eraseTargetMetadata(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::Operation *, 4> toErase;
  moduleOp.walk([&](mlir::Operation *op) {
    if (op != moduleOp && isWaferTargetMetadata(op))
      toErase.push_back(op);
  });
  for (mlir::Operation *op : toErase)
    op->erase();
}
mlir::LogicalResult lowerSCFToControlFlow(mlir::ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(moduleOp.getContext());
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::ConversionTarget target(*moduleOp.getContext());
  target.addIllegalDialect<mlir::scf::SCFDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  if (mlir::failed(
          mlir::applyPartialConversion(moduleOp, target, std::move(patterns))))
    return moduleOp.emitError()
           << "unsupported_target_structure: failed to lower SCF control "
              "flow to CFG";
  return mlir::success();
}

} // namespace wafer::target_llvm_detail
