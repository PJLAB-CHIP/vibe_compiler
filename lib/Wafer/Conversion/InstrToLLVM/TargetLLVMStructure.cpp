//===- Target LLVM lowering implementation -------------------------------===//

#include "Wafer/Conversion/InstrToLLVM/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetFormat.h"
#include "Wafer/Target/TargetMemory.h"
#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"

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
    std::optional<analysis::SingleExecutionRegionFlow> flow =
        analysis::getSingleExecutionRegionFlow(tileRegion);
    if (!flow)
      return tileRegion.emitError()
             << "unsupported_target_structure: wafer.tile.region must expose "
                "one exact parent-to-region-to-parent control-flow edge";
    mlir::Block &body = flow->region->front();

    llvm::SmallVector<mlir::Value, 4> yieldedValues;
    for (mlir::Value value : flow->exitOperands) {
      auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
      if (blockArg && blockArg.getOwner() == &body) {
        auto entryArgument = llvm::find(flow->entryArguments, blockArg);
        if (entryArgument == flow->entryArguments.end())
          return tileRegion.emitError()
                 << "unsupported_target_structure: tile yield block argument "
                    "has no matching boundary input";
        value =
            flow->entryOperands[entryArgument - flow->entryArguments.begin()];
      }
      yieldedValues.push_back(value);
    }
    if (yieldedValues.size() != flow->results.size())
      return tileRegion.emitError()
             << "unsupported_target_structure: tile yield/result count "
                "mismatch";

    mlir::Operation *terminator = body.getTerminator();
    rewriter.inlineBlockBefore(&body, tileRegion.getOperation(),
                               flow->entryOperands);
    rewriter.eraseOp(terminator);
    for (auto [result, replacement] :
         llvm::zip_equal(flow->results, yieldedValues))
      rewriter.replaceAllUsesWith(result, replacement);
    rewriter.eraseOp(tileRegion);
  }
  return mlir::success();
}
mlir::LogicalResult
validateDirectCallsForTarget(const analysis::DirectCallGraphAnalysis &graph,
                             int64_t defaultDDRArenaArgumentIndex,
                             int64_t transportStatusArgumentIndex,
                             int64_t profileRecordArgumentIndex) {
  mlir::ModuleOp moduleOp = graph.getModule();
  if (!moduleOp)
    return mlir::failure();
  for (mlir::func::FuncOp funcOp : graph.getFunctions()) {
    if (funcOp.isDeclaration())
      return funcOp.emitError()
             << "unsupported_target_call: external func.func declarations "
                "are not accepted by target LLVM lowering";
    for (auto [index, type] :
         llvm::enumerate(funcOp.getFunctionType().getInputs()))
      if (!isWaferDDRMemRefType(type) &&
          !(static_cast<int64_t>(index) == defaultDDRArenaArgumentIndex &&
            type.isInteger(64)) &&
          !(static_cast<int64_t>(index) == transportStatusArgumentIndex &&
            type.isInteger(64)) &&
          !(static_cast<int64_t>(index) == profileRecordArgumentIndex &&
            type.isInteger(64)))
        return funcOp.emitError()
               << "unsupported_target_function: argument #" << index
               << " must be a Wafer DDR memref target binding";
  }
  if (!graph.getUnresolvedCalls().empty()) {
    mlir::func::CallOp call = graph.getUnresolvedCalls().front();
    return call.emitError()
           << "unsupported_target_call: unresolved direct callee @"
           << call.getCallee();
  }
  if (!graph.getUnsupportedCallOperations().empty()) {
    mlir::Operation *operation = graph.getUnsupportedCallOperations().front();
    return operation->emitError()
           << "unsupported_target_call: indirect or unknown call-like "
              "operation '"
           << operation->getName() << "' is not supported";
  }
  if (graph.hasRecursiveCycle())
    return graph.getRecursiveCycleOrigin()->emitError()
           << "unsupported_target_call: recursive direct call graph is not "
              "supported";
  return mlir::success();
}

mlir::FailureOr<mlir::func::FuncOp>
findUniqueRootFunction(const analysis::DirectCallGraphAnalysis &graph) {
  llvm::SmallVector<mlir::func::FuncOp, 2> roots = graph.getRootFunctions();
  if (roots.size() > 1)
    return roots[1].emitError()
           << "unsupported_target_transport: compiler-managed entry "
              "arguments require one unique entry function";
  if (roots.empty())
    return graph.getModule().emitError()
           << "unsupported_target_transport: compiler-managed entry "
              "arguments have no entry function";
  return roots.front();
}

static mlir::FailureOr<unsigned> resolveDDRAliasToFunctionArgument(
    mlir::Value value, mlir::func::FuncOp funcOp,
    const analysis::DirectCallGraphAnalysis &graph,
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
            incoming, funcOp, graph, summaries, branchVisiting);
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
  mlir::func::FuncOp callee = graph.getCallee(call);
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
                                           graph, summaries, visiting);
}

mlir::LogicalResult analyzeDDRAliasContracts(
    mlir::ModuleOp moduleOp, const analysis::DirectCallGraphAnalysis &graph,
    llvm::DenseMap<mlir::Operation *, AliasSummary> &summaries) {
  for (mlir::func::FuncOp funcOp : graph.getCalleeFirstOrder()) {
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
            operand, funcOp, graph, summaries, visiting);
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
                          const analysis::DirectCallGraphAnalysis &graph) {
  for (mlir::func::FuncOp funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
    if (graph.isCalled(funcOp) || funcOp.getFunctionType().getNumResults() == 0)
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
