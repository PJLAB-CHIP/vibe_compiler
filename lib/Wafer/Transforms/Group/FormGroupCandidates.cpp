//===- FormGroupCandidates.cpp - Wafer group candidate formation ---------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {
namespace {

static bool isTensorType(mlir::Type type) {
  return mlir::isa<mlir::TensorType>(type);
}

static bool hasTensorResults(mlir::Operation *op) {
  return op->getNumResults() > 0 &&
         llvm::all_of(op->getResultTypes(), isTensorType);
}

static bool hasTensorInits(mlir::DestinationStyleOpInterface dpsOp) {
  return dpsOp.getNumDpsInits() > 0 &&
         llvm::all_of(dpsOp.getDpsInits().getTypes(), isTensorType);
}

static bool hasDuplicateBoundaryValues(mlir::ValueRange inputs,
                                       mlir::ValueRange outs) {
  llvm::DenseSet<mlir::Value> seen;
  for (mlir::Value input : inputs) {
    if (!seen.insert(input).second)
      return true;
  }
  for (mlir::Value out : outs) {
    if (!seen.insert(out).second)
      return true;
  }
  return false;
}

static bool isTensorCollectiveRoot(mlir::Operation *op) {
  return op->getName().getStringRef().starts_with("wafer.tensor_collective.") &&
         !mlir::isa<TensorCollectiveYieldOp>(op);
}

static bool isLinalgRoot(mlir::Operation *op) {
  if (!mlir::isa<mlir::linalg::LinalgOp>(op))
    return false;

  // A fill is usually the init producer for the real root. Grouping it alone
  // adds a boundary that does not help downstream tile feasibility.
  return op->getName().getStringRef() != "linalg.fill";
}

static bool isCandidateRoot(mlir::Operation *op) {
  if (op->getParentOfType<GroupOp>())
    return false;
  if (!hasTensorResults(op))
    return false;

  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp || !hasTensorInits(dpsOp))
    return false;

  return isLinalgRoot(op) || isTensorCollectiveRoot(op);
}

static mlir::LogicalResult formGroupForRoot(mlir::Operation *root) {
  auto dpsOp = mlir::cast<mlir::DestinationStyleOpInterface>(root);
  llvm::SmallVector<mlir::Value> inputs = dpsOp.getDpsInputs();
  llvm::SmallVector<mlir::Value> outs(dpsOp.getDpsInits().begin(),
                                      dpsOp.getDpsInits().end());

  if (hasDuplicateBoundaryValues(inputs, outs))
    return mlir::success();

  mlir::OpBuilder builder(root);
  auto group = builder.create<GroupOp>(root->getLoc(), root->getResultTypes(),
                                       inputs, outs);

  mlir::Region &body = group.getBody();
  body.push_back(new mlir::Block);
  mlir::Block &block = body.front();
  for (mlir::Value input : inputs)
    block.addArgument(input.getType(), input.getLoc());
  for (mlir::Value out : outs)
    block.addArgument(out.getType(), out.getLoc());

  mlir::IRMapping mapping;
  unsigned blockArgIndex = 0;
  for (mlir::Value input : inputs)
    mapping.map(input, block.getArgument(blockArgIndex++));
  for (mlir::Value out : outs)
    mapping.map(out, block.getArgument(blockArgIndex++));

  mlir::OpBuilder bodyBuilder(&block, block.end());
  mlir::Operation *clonedRoot = bodyBuilder.clone(*root, mapping);
  bodyBuilder.create<GroupYieldOp>(root->getLoc(), clonedRoot->getResults());

  root->replaceAllUsesWith(group.getResults());
  root->erase();
  return mlir::success();
}

struct FormGroupCandidatesPass
    : public mlir::PassWrapper<FormGroupCandidatesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormGroupCandidatesPass)

  llvm::StringRef getArgument() const final {
    return "wafer-form-group-candidates";
  }

  llvm::StringRef getDescription() const final {
    return "form root-seeded tensor-level wafer.group candidates";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::math::MathDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect, wafer::WaferDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::Operation *> roots;
    getOperation().walk([&](mlir::Operation *op) {
      if (isCandidateRoot(op))
        roots.push_back(op);
    });

    for (mlir::Operation *root : roots) {
      if (!root->getBlock())
        continue;
      if (mlir::failed(formGroupForRoot(root))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFormGroupCandidatesPass() {
  return std::make_unique<FormGroupCandidatesPass>();
}

} // namespace wafer
