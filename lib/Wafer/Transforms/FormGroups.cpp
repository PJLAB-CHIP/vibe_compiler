//===- FormGroups.cpp - Form tensor-level Wafer groups -------------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {
namespace {

static bool isTensorValue(mlir::Value value) {
  return mlir::isa<mlir::TensorType>(value.getType());
}

static bool canFormSingleMatmulGroup(mlir::linalg::MatmulOp matmul) {
  if (matmul->getParentOfType<wafer::GroupOp>())
    return false;
  if (matmul->getNumResults() != 1 || matmul.getOutputs().size() != 1)
    return false;
  if (!llvm::all_of(matmul.getInputs(), isTensorValue))
    return false;
  if (!llvm::all_of(matmul.getOutputs(), isTensorValue))
    return false;
  return true;
}

static void addMappedBlockArguments(mlir::Block *block, mlir::ValueRange values,
                                    mlir::IRMapping &mapping) {
  for (mlir::Value value : values) {
    mlir::BlockArgument arg =
        block->addArgument(value.getType(), value.getLoc());
    mapping.map(value, arg);
  }
}

static void formGroup(mlir::linalg::MatmulOp matmul) {
  llvm::SmallVector<mlir::Value> inputs(matmul.getInputs().begin(),
                                        matmul.getInputs().end());
  llvm::SmallVector<mlir::Value> outs(matmul.getOutputs().begin(),
                                      matmul.getOutputs().end());

  mlir::OpBuilder builder(matmul);
  auto group = builder.create<wafer::GroupOp>(
      matmul.getLoc(), matmul->getResultTypes(), inputs, outs);

  mlir::Block *body = new mlir::Block();
  group.getBody().push_back(body);

  mlir::IRMapping mapping;
  addMappedBlockArguments(body, inputs, mapping);
  addMappedBlockArguments(body, outs, mapping);

  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(body);
  mlir::Operation *cloned = bodyBuilder.clone(*matmul.getOperation(), mapping);
  bodyBuilder.create<wafer::GroupYieldOp>(matmul.getLoc(),
                                          cloned->getResults());

  matmul->replaceAllUsesWith(group->getResults());
  matmul.erase();
}

struct FormGroupsPass
    : public mlir::PassWrapper<FormGroupsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormGroupsPass)

  llvm::StringRef getArgument() const final { return "wafer-form-groups"; }

  llvm::StringRef getDescription() const final {
    return "form conservative tensor-level Wafer logical groups";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, wafer::WaferDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      if (canFormSingleMatmulGroup(matmul))
        matmuls.push_back(matmul);
    });

    for (mlir::linalg::MatmulOp matmul : matmuls)
      formGroup(matmul);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFormGroupsPass() {
  return std::make_unique<FormGroupsPass>();
}

} // namespace wafer
