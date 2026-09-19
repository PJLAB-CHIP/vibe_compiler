//===- ScalarExecution.cpp - Materialize scalar execution choices --------===//

#include "ScalarExecution.h"
#include "StructuredBufferRelations.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {

bool isRegisterScalar(mlir::Value value, llvm::DenseSet<mlir::Value> &visited) {
  if (!mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
          value.getType()))
    return false;
  if (!visited.insert(value).second)
    return true;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto *parent = argument.getOwner()->getParentOp();
    if (mlir::isa<mlir::func::FuncOp>(parent))
      return true;
    if (auto region = mlir::dyn_cast<TileRegionOp>(parent))
      return isRegisterScalar(region.getInputs()[argument.getArgNumber()],
                              visited);
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent);
    return loop && argument == loop.getInductionVar();
  }
  auto *definition = value.getDefiningOp();
  return definition && definition->getDialect() &&
         definition->getDialect()->getTypeID() ==
             mlir::TypeID::get<mlir::arith::ArithDialect>() &&
         llvm::all_of(definition->getOperands(), [&](mlir::Value operand) {
           return isRegisterScalar(operand, visited);
         });
}

mlir::Operation *getCPUArithmetic(mlir::linalg::GenericOp operation) {
  if (!operation->getParentOfType<TileRegionOp>() ||
      !operation.hasPureTensorSemantics() || operation.getNumLoops() ||
      operation.getNumResults() != 1 || operation.getNumDpsInits() != 1 ||
      operation.getBody()->getOperations().size() != 2 ||
      !operation.getDpsInits().front().getDefiningOp<mlir::tensor::EmptyOp>() ||
      !operation.getRegionOutputArgs().front().use_empty())
    return nullptr;
  auto *scalar = &operation.getBody()->front();
  if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
                 mlir::arith::DivFOp>(scalar) ||
      !scalar->getResult(0).getType().isF32() ||
      operation.getBody()->getTerminator()->getOperand(0) !=
          scalar->getResult(0) ||
      operation.getResult(0).use_empty())
    return nullptr;
  llvm::DenseSet<mlir::Value> visited;
  for (mlir::Value operand : scalar->getOperands()) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand))
      if (argument.getOwner() == operation.getBody())
        operand = operation.getDpsInputs()[argument.getArgNumber()];
    if (!isRegisterScalar(operand, visited))
      return nullptr;
  }
  for (mlir::OpOperand &use : operation.getResult(0).getUses()) {
    auto consumer = mlir::dyn_cast<mlir::linalg::GenericOp>(use.getOwner());
    if (!consumer || !consumer.hasPureTensorSemantics() ||
        !consumer.isDpsInput(&use) ||
        consumer.getMatchingIndexingMap(&use).getNumResults())
      return nullptr;
  }
  return scalar;
}

} // namespace

bool hasCPUScalarAlternative(mlir::ModuleOp module) {
  return module
      .walk([](mlir::linalg::GenericOp operation) {
        return getCPUArithmetic(operation) ? mlir::WalkResult::interrupt()
                                           : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

mlir::FailureOr<unsigned>
materializeCPUScalarAlternative(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations) {
  if (mlir::failed(mlir::verify(module)))
    return mlir::failure();
  llvm::SmallVector<mlir::linalg::GenericOp> operations;
  module.walk([&](mlir::linalg::GenericOp op) { operations.push_back(op); });
  StructuredBufferReplacementListener listener(relations);
  mlir::IRRewriter rewriter(module.getContext(), &listener);
  unsigned count = 0;
  for (auto operation : operations) {
    auto *scalar = getCPUArithmetic(operation);
    if (!scalar)
      continue;
    mlir::IRMapping mapping;
    for (auto [argument, input] : llvm::zip_equal(
             operation.getRegionInputArgs(), operation.getDpsInputs()))
      mapping.map(argument, input);
    rewriter.setInsertionPoint(operation);
    auto *lowered = rewriter.clone(*scalar, mapping);
    listener.recordLoweredOperation(operation, lowered);
    rewriter.replaceOp(operation, lowered->getResults());
    ++count;
  }
  if (!listener.finalizeAfterRewrite() || mlir::failed(mlir::verify(module)))
    return mlir::failure();
  rebuildCurrentBufferOwnerRelations(module, relations);
  if (mlir::failed(checkStructuredBufferRelationsCurrent(module, relations)))
    return mlir::failure();
  return count;
}

} // namespace wafer::compiler::detail
