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

#include <optional>

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

static std::optional<wafer::ComputeElementwiseKind>
mapElementwiseKind(mlir::linalg::ElementwiseKind kind) {
  switch (kind) {
  case mlir::linalg::ElementwiseKind::add:
    return wafer::ComputeElementwiseKind::Add;
  case mlir::linalg::ElementwiseKind::sub:
    return wafer::ComputeElementwiseKind::Sub;
  case mlir::linalg::ElementwiseKind::mul:
    return wafer::ComputeElementwiseKind::Mul;
  case mlir::linalg::ElementwiseKind::div:
    return wafer::ComputeElementwiseKind::Div;
  case mlir::linalg::ElementwiseKind::max_signed:
    return wafer::ComputeElementwiseKind::Max;
  case mlir::linalg::ElementwiseKind::min_signed:
    return wafer::ComputeElementwiseKind::Min;
  case mlir::linalg::ElementwiseKind::negf:
    return wafer::ComputeElementwiseKind::Neg;
  case mlir::linalg::ElementwiseKind::reciprocal:
    return wafer::ComputeElementwiseKind::Recip;
  case mlir::linalg::ElementwiseKind::sqrt:
    return wafer::ComputeElementwiseKind::Sqrt;
  case mlir::linalg::ElementwiseKind::rsqrt:
    return wafer::ComputeElementwiseKind::Rsqrt;
  case mlir::linalg::ElementwiseKind::exp:
    return wafer::ComputeElementwiseKind::Exp;
  case mlir::linalg::ElementwiseKind::tanh:
    return wafer::ComputeElementwiseKind::Tanh;
  default:
    return std::nullopt;
  }
}

static bool canFormSingleElementwiseGroup(
    mlir::linalg::ElementwiseOp elementwise) {
  if (elementwise->getParentOfType<wafer::GroupOp>())
    return false;
  if (!mapElementwiseKind(elementwise.getKind()))
    return false;
  if (elementwise->getNumResults() != 1 || elementwise.getOutputs().size() != 1)
    return false;
  for (mlir::AffineMap map : elementwise.getIndexingMapsArray()) {
    if (!map.isIdentity())
      return false;
  }
  if (elementwise.getIndexingMapsArray().size() !=
      elementwise.getInputs().size() + elementwise.getOutputs().size())
    return false;
  if (!llvm::all_of(elementwise.getInputs(), isTensorValue))
    return false;
  if (!llvm::all_of(elementwise.getOutputs(), isTensorValue))
    return false;

  mlir::Type resultType = elementwise->getResult(0).getType();
  for (mlir::Value input : elementwise.getInputs()) {
    if (input.getType() != resultType)
      return false;
  }
  return elementwise.getOutputs()[0].getType() == resultType;
}

static void addMappedBlockArguments(mlir::Block *block, mlir::ValueRange values,
                                    mlir::IRMapping &mapping) {
  for (mlir::Value value : values) {
    mlir::BlockArgument arg =
        block->addArgument(value.getType(), value.getLoc());
    mapping.map(value, arg);
  }
}

static void formGroup(mlir::Operation *root, mlir::ValueRange rootInputs,
                      mlir::ValueRange rootOuts) {
  llvm::SmallVector<mlir::Value> inputs(rootInputs.begin(), rootInputs.end());
  llvm::SmallVector<mlir::Value> outs(rootOuts.begin(), rootOuts.end());

  mlir::OpBuilder builder(root);
  auto group = builder.create<wafer::GroupOp>(
      root->getLoc(), root->getResultTypes(), inputs, outs);

  mlir::Block *body = new mlir::Block();
  group.getBody().push_back(body);

  mlir::IRMapping mapping;
  addMappedBlockArguments(body, inputs, mapping);
  addMappedBlockArguments(body, outs, mapping);

  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(body);
  mlir::Operation *cloned = bodyBuilder.clone(*root, mapping);
  bodyBuilder.create<wafer::GroupYieldOp>(root->getLoc(),
                                          cloned->getResults());

  root->replaceAllUsesWith(group->getResults());
  root->erase();
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
    llvm::SmallVector<mlir::Operation *> roots;
    getOperation().walk([&](mlir::Operation *op) {
      if (auto matmul = mlir::dyn_cast<mlir::linalg::MatmulOp>(op)) {
        if (canFormSingleMatmulGroup(matmul))
          roots.push_back(op);
        return;
      }
      if (auto elementwise =
              mlir::dyn_cast<mlir::linalg::ElementwiseOp>(op)) {
        if (canFormSingleElementwiseGroup(elementwise))
          roots.push_back(op);
      }
    });

    for (mlir::Operation *root : roots) {
      if (auto matmul = mlir::dyn_cast<mlir::linalg::MatmulOp>(root))
        formGroup(root, matmul.getInputs(), matmul.getOutputs());
      else if (auto elementwise =
                   mlir::dyn_cast<mlir::linalg::ElementwiseOp>(root))
        formGroup(root, elementwise.getInputs(), elementwise.getOutputs());
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFormGroupsPass() {
  return std::make_unique<FormGroupsPass>();
}

} // namespace wafer
