//===- CheckMlpSchedule.cpp - Validate MLP staged schedules --------------===//

#include "Wafer/Transforms/Passes.h"

#include "Support/ElementwiseUtils.h"

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

static bool isRank2Matmul(mlir::linalg::MatmulOp matmul) {
  return hasSingleResult(matmul) && hasRank(matmul->getResult(0), 2);
}

static bool consumesValue(mlir::linalg::MatmulOp matmul, mlir::Value value) {
  for (mlir::Value input : matmul.getInputs())
    if (input == value)
      return true;
  return false;
}

static bool isTanhActivation(mlir::linalg::GenericOp elementwise) {
  std::optional<wafer::ComputeElementwiseKind> kind =
      matchElementwiseGeneric(elementwise);
  llvm::SmallVector<mlir::Value> inputs = elementwise.getDpsInputs();
  return hasSingleResult(elementwise) &&
         kind == wafer::ComputeElementwiseKind::Tanh &&
         inputs.size() == 1 && hasRank(inputs[0], 2) &&
         hasRank(elementwise->getResult(0), 2);
}

static mlir::linalg::MatmulOp getDefiningRank2Matmul(mlir::Value value) {
  auto matmul = value.getDefiningOp<mlir::linalg::MatmulOp>();
  if (!matmul || !isRank2Matmul(matmul))
    return {};
  return matmul;
}

static bool isGatedMultiply(mlir::linalg::GenericOp elementwise,
                            mlir::Value activated) {
  std::optional<wafer::ComputeElementwiseKind> kind =
      matchElementwiseGeneric(elementwise);
  llvm::SmallVector<mlir::Value> inputs = elementwise.getDpsInputs();
  if (!hasSingleResult(elementwise) || kind != wafer::ComputeElementwiseKind::Mul ||
      inputs.size() != 2 ||
      !hasRank(elementwise->getResult(0), 2))
    return false;

  mlir::Value lhs = inputs[0];
  mlir::Value rhs = inputs[1];
  if (lhs == activated)
    return static_cast<bool>(getDefiningRank2Matmul(rhs));
  if (rhs == activated)
    return static_cast<bool>(getDefiningRank2Matmul(lhs));
  return false;
}

struct MlpStagePresence {
  bool sawActivatedProjection = false;
  bool sawGatedMultiply = false;
  bool sawDownProjection = false;
};

static MlpStagePresence findMlpStages(
    llvm::ArrayRef<mlir::linalg::MatmulOp> matmuls,
    llvm::ArrayRef<mlir::linalg::GenericOp> elementwiseOps) {
  MlpStagePresence stages;

  for (mlir::linalg::GenericOp activation : elementwiseOps) {
    if (!isTanhActivation(activation))
      continue;
    llvm::SmallVector<mlir::Value> activationInputs =
        activation.getDpsInputs();
    if (!getDefiningRank2Matmul(activationInputs[0]))
      continue;
    stages.sawActivatedProjection = true;
    mlir::Value activated = activation->getResult(0);

    for (mlir::linalg::GenericOp multiply : elementwiseOps) {
      if (!isGatedMultiply(multiply, activated))
        continue;
      stages.sawGatedMultiply = true;
      mlir::Value gated = multiply->getResult(0);

      for (mlir::linalg::MatmulOp downProjection : matmuls) {
        if (!isRank2Matmul(downProjection) ||
            !consumesValue(downProjection, gated))
          continue;
        stages.sawDownProjection = true;
        return stages;
      }
    }
  }

  return stages;
}

struct CheckMlpSchedulePass
    : public mlir::PassWrapper<CheckMlpSchedulePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckMlpSchedulePass)

  llvm::StringRef getArgument() const final {
    return "wafer-check-mlp-schedule";
  }

  llvm::StringRef getDescription() const final {
    return "validate MLP projection, activation, gate, and down projection";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    llvm::SmallVector<mlir::linalg::GenericOp> elementwiseOps;
    module.walk(
        [&](mlir::linalg::MatmulOp matmul) { matmuls.push_back(matmul); });
    module.walk([&](mlir::linalg::GenericOp elementwise) {
      if (matchElementwiseGeneric(elementwise))
        elementwiseOps.push_back(elementwise);
    });

    MlpStagePresence stages = findMlpStages(matmuls, elementwiseOps);
    if (!stages.sawActivatedProjection) {
      module.emitError(
          "MLP schedule requires an activated rank-2 projection matmul");
      signalPassFailure();
      return;
    }
    if (!stages.sawGatedMultiply) {
      module.emitError("MLP schedule requires an elementwise multiply between "
                       "activation and up projection");
      signalPassFailure();
      return;
    }
    if (!stages.sawDownProjection) {
      module.emitError("MLP schedule requires a down projection matmul "
                       "consuming the gated activation");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckMlpSchedulePass() {
  return std::make_unique<CheckMlpSchedulePass>();
}

} // namespace wafer
