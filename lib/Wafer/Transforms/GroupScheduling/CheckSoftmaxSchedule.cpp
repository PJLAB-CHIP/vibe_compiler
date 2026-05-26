//===- CheckSoftmaxSchedule.cpp - Validate softmax staged schedules -------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {
namespace {

enum class ReduceKind { Sum, Max, Other };

static std::optional<int64_t> getRankedTensorRank(mlir::Value value) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType)
    return std::nullopt;
  return tensorType.getRank();
}

static bool isLastDimReduce(mlir::linalg::ReduceOp reduce) {
  if (reduce.getInputs().empty())
    return false;
  std::optional<int64_t> inputRank = getRankedTensorRank(reduce.getInputs()[0]);
  if (!inputRank || *inputRank <= 0)
    return false;
  llvm::ArrayRef<int64_t> dims = reduce.getDimensions();
  return dims.size() == 1 && dims.front() == *inputRank - 1;
}

static bool areBlockArguments(mlir::Value lhs, mlir::Value rhs,
                              mlir::BlockArgument arg0,
                              mlir::BlockArgument arg1) {
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

static ReduceKind getReduceKind(mlir::linalg::ReduceOp reduce) {
  if (reduce->getNumResults() != 1 || reduce.getInputs().size() != 1 ||
      reduce.getRegion().empty())
    return ReduceKind::Other;

  mlir::Block &body = reduce.getRegion().front();
  if (body.getNumArguments() != 2 || !body.getTerminator() ||
      body.getTerminator()->getNumOperands() != 1)
    return ReduceKind::Other;

  mlir::Value yielded = body.getTerminator()->getOperand(0);
  if (auto add = yielded.getDefiningOp<mlir::arith::AddFOp>())
    if (areBlockArguments(add->getOperand(0), add->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return ReduceKind::Sum;
  if (auto add = yielded.getDefiningOp<mlir::arith::AddIOp>())
    if (areBlockArguments(add->getOperand(0), add->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return ReduceKind::Sum;

  if (auto max = yielded.getDefiningOp<mlir::arith::MaximumFOp>())
    if (areBlockArguments(max->getOperand(0), max->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return ReduceKind::Max;
  if (auto max = yielded.getDefiningOp<mlir::arith::MaxSIOp>())
    if (areBlockArguments(max->getOperand(0), max->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return ReduceKind::Max;

  return ReduceKind::Other;
}

static bool hasSingleResult(mlir::Operation *op) {
  return op && op->getNumResults() == 1;
}

static bool hasKind(mlir::linalg::ReduceOp reduce, ReduceKind kind) {
  return isLastDimReduce(reduce) && getReduceKind(reduce) == kind;
}

static bool isBinaryElementwise(mlir::linalg::ElementwiseOp elementwise,
                                mlir::linalg::ElementwiseKind kind,
                                mlir::Value lhs, mlir::Value rhs) {
  return hasSingleResult(elementwise) && elementwise.getKind() == kind &&
         elementwise.getInputs().size() == 2 &&
         elementwise.getInputs()[0] == lhs &&
         elementwise.getInputs()[1] == rhs;
}

static bool isUnaryElementwise(mlir::linalg::ElementwiseOp elementwise,
                               mlir::linalg::ElementwiseKind kind,
                               mlir::Value input) {
  return hasSingleResult(elementwise) && elementwise.getKind() == kind &&
         elementwise.getInputs().size() == 1 &&
         elementwise.getInputs()[0] == input;
}

struct SoftmaxStagePresence {
  bool sawMaxReduce = false;
  bool sawSubtract = false;
  bool sawExp = false;
  bool sawSumReduce = false;
  bool sawNormalizeDiv = false;
};

static SoftmaxStagePresence
findSoftmaxStages(llvm::ArrayRef<mlir::linalg::ReduceOp> reduces,
                  llvm::ArrayRef<mlir::linalg::ElementwiseOp> elementwiseOps) {
  SoftmaxStagePresence stages;

  for (mlir::linalg::ReduceOp maxReduce : reduces) {
    if (!hasKind(maxReduce, ReduceKind::Max))
      continue;
    stages.sawMaxReduce = true;
    mlir::Value scores = maxReduce.getInputs()[0];
    mlir::Value rowMax = maxReduce->getResult(0);

    for (mlir::linalg::ElementwiseOp subtract : elementwiseOps) {
      if (!isBinaryElementwise(subtract, mlir::linalg::ElementwiseKind::sub,
                               scores, rowMax))
        continue;
      stages.sawSubtract = true;
      mlir::Value shifted = subtract->getResult(0);

      for (mlir::linalg::ElementwiseOp exp : elementwiseOps) {
        if (!isUnaryElementwise(exp, mlir::linalg::ElementwiseKind::exp,
                                shifted))
          continue;
        stages.sawExp = true;
        mlir::Value expScores = exp->getResult(0);

        for (mlir::linalg::ReduceOp sumReduce : reduces) {
          if (!hasKind(sumReduce, ReduceKind::Sum) ||
              sumReduce.getInputs()[0] != expScores)
            continue;
          stages.sawSumReduce = true;
          mlir::Value rowSum = sumReduce->getResult(0);

          for (mlir::linalg::ElementwiseOp div : elementwiseOps) {
            if (!isBinaryElementwise(div, mlir::linalg::ElementwiseKind::div,
                                     expScores, rowSum))
              continue;
            stages.sawNormalizeDiv = true;
            return stages;
          }
        }
      }
    }
  }

  return stages;
}

struct CheckSoftmaxSchedulePass
    : public mlir::PassWrapper<CheckSoftmaxSchedulePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckSoftmaxSchedulePass)

  llvm::StringRef getArgument() const final {
    return "wafer-check-softmax-schedule";
  }

  llvm::StringRef getDescription() const final {
    return "validate staged softmax reduce and elementwise schedule";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::linalg::ReduceOp> reduces;
    llvm::SmallVector<mlir::linalg::ElementwiseOp> elementwiseOps;
    module.walk(
        [&](mlir::linalg::ReduceOp reduce) { reduces.push_back(reduce); });
    module.walk([&](mlir::linalg::ElementwiseOp elementwise) {
      elementwiseOps.push_back(elementwise);
    });

    SoftmaxStagePresence stages = findSoftmaxStages(reduces, elementwiseOps);
    if (!stages.sawMaxReduce) {
      module.emitError(
          "softmax schedule requires a hidden-dimension reduce max stage");
      signalPassFailure();
      return;
    }
    if (!stages.sawSubtract) {
      module.emitError("softmax schedule requires row max subtraction before "
                       "exp");
      signalPassFailure();
      return;
    }
    if (!stages.sawExp) {
      module.emitError("softmax schedule requires an exp elementwise stage "
                       "after row max subtraction");
      signalPassFailure();
      return;
    }
    if (!stages.sawSumReduce) {
      module.emitError("softmax schedule requires a hidden-dimension reduce "
                       "sum stage over exp scores");
      signalPassFailure();
      return;
    }
    if (!stages.sawNormalizeDiv) {
      module.emitError(
          "softmax schedule requires final exp/sum broadcast divide stage");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckSoftmaxSchedulePass() {
  return std::make_unique<CheckSoftmaxSchedulePass>();
}

} // namespace wafer
