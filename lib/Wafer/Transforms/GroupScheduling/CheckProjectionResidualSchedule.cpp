//===- CheckProjectionResidualSchedule.cpp - Validate projection schedule -===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {
namespace {

static std::optional<int64_t> getRankedTensorRank(mlir::Value value) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType)
    return std::nullopt;
  return tensorType.getRank();
}

static bool hasRank(mlir::Value value, int64_t rank) {
  std::optional<int64_t> valueRank = getRankedTensorRank(value);
  return valueRank && *valueRank == rank;
}

static bool hasSingleResult(mlir::Operation *op) {
  return op && op->getNumResults() == 1;
}

static bool isAdd(mlir::linalg::ElementwiseOp elementwise) {
  return hasSingleResult(elementwise) &&
         elementwise.getKind() == mlir::linalg::ElementwiseKind::add &&
         elementwise.getInputs().size() == 2 &&
         hasRank(elementwise->getResult(0), 2);
}

static bool isBiasAdd(mlir::linalg::ElementwiseOp elementwise,
                      mlir::Value projected) {
  if (!isAdd(elementwise))
    return false;
  mlir::Value lhs = elementwise.getInputs()[0];
  mlir::Value rhs = elementwise.getInputs()[1];
  return (lhs == projected && hasRank(rhs, 1)) ||
         (rhs == projected && hasRank(lhs, 1));
}

static bool isResidualAdd(mlir::linalg::ElementwiseOp elementwise,
                          mlir::Value biased) {
  if (!isAdd(elementwise))
    return false;
  mlir::Value lhs = elementwise.getInputs()[0];
  mlir::Value rhs = elementwise.getInputs()[1];
  return (lhs == biased && hasRank(rhs, 2)) ||
         (rhs == biased && hasRank(lhs, 2));
}

struct ProjectionResidualStagePresence {
  bool sawMatmul = false;
  bool sawBiasAdd = false;
  bool sawResidualAdd = false;
};

static ProjectionResidualStagePresence findProjectionResidualStages(
    llvm::ArrayRef<mlir::linalg::MatmulOp> matmuls,
    llvm::ArrayRef<mlir::linalg::ElementwiseOp> elementwiseOps) {
  ProjectionResidualStagePresence stages;

  for (mlir::linalg::MatmulOp matmul : matmuls) {
    if (!hasSingleResult(matmul) || !hasRank(matmul->getResult(0), 2))
      continue;
    stages.sawMatmul = true;
    mlir::Value projected = matmul->getResult(0);

    for (mlir::linalg::ElementwiseOp biasAdd : elementwiseOps) {
      if (!isBiasAdd(biasAdd, projected))
        continue;
      stages.sawBiasAdd = true;
      mlir::Value biased = biasAdd->getResult(0);

      for (mlir::linalg::ElementwiseOp residualAdd : elementwiseOps) {
        if (!isResidualAdd(residualAdd, biased))
          continue;
        stages.sawResidualAdd = true;
        return stages;
      }
    }
  }

  return stages;
}

struct CheckProjectionResidualSchedulePass
    : public mlir::PassWrapper<CheckProjectionResidualSchedulePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CheckProjectionResidualSchedulePass)

  llvm::StringRef getArgument() const final {
    return "wafer-check-projection-residual-schedule";
  }

  llvm::StringRef getDescription() const final {
    return "validate output projection bias and residual staged schedule";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    llvm::SmallVector<mlir::linalg::ElementwiseOp> elementwiseOps;
    module.walk(
        [&](mlir::linalg::MatmulOp matmul) { matmuls.push_back(matmul); });
    module.walk([&](mlir::linalg::ElementwiseOp elementwise) {
      elementwiseOps.push_back(elementwise);
    });

    ProjectionResidualStagePresence stages =
        findProjectionResidualStages(matmuls, elementwiseOps);
    if (!stages.sawMatmul) {
      module.emitError(
          "projection residual schedule requires a rank-2 linalg.matmul");
      signalPassFailure();
      return;
    }
    if (!stages.sawBiasAdd) {
      module.emitError("projection residual schedule requires a rank-2/"
                       "rank-1 bias add stage");
      signalPassFailure();
      return;
    }
    if (!stages.sawResidualAdd) {
      module.emitError(
          "projection residual schedule requires a rank-2 residual add stage");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckProjectionResidualSchedulePass() {
  return std::make_unique<CheckProjectionResidualSchedulePass>();
}

} // namespace wafer
