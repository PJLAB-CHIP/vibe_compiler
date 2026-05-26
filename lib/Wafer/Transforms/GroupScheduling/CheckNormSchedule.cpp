//===- CheckNormSchedule.cpp - Validate norm staged schedules ------------===//

#include "Wafer/Transforms/Passes.h"

#include "Support/ElementwiseUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace wafer {
namespace {

static std::optional<int64_t> getRankedTensorRank(mlir::Value value) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType)
    return std::nullopt;
  return tensorType.getRank();
}

static bool isHiddenDimReduce(mlir::linalg::ReduceOp reduce) {
  if (reduce.getInputs().empty())
    return false;
  std::optional<int64_t> inputRank = getRankedTensorRank(reduce.getInputs()[0]);
  if (!inputRank || *inputRank <= 0)
    return false;
  llvm::ArrayRef<int64_t> dims = reduce.getDimensions();
  return dims.size() == 1 && dims.front() == *inputRank - 1;
}

static bool isBroadcastMul(mlir::linalg::GenericOp elementwise) {
  std::optional<wafer::ComputeElementwiseKind> kind =
      matchElementwiseGeneric(elementwise);
  if (kind != wafer::ComputeElementwiseKind::Mul)
    return false;
  if (elementwise.getDpsInits().empty())
    return false;
  std::optional<int64_t> outputRank =
      getRankedTensorRank(elementwise.getDpsInits()[0]);
  if (!outputRank || *outputRank < 2)
    return false;

  bool hasFullRankInput = false;
  bool hasReducedRankInput = false;
  for (mlir::Value input : elementwise.getDpsInputs()) {
    std::optional<int64_t> inputRank = getRankedTensorRank(input);
    if (!inputRank)
      continue;
    hasFullRankInput |= *inputRank == *outputRank;
    hasReducedRankInput |= *inputRank == *outputRank - 1;
  }
  return hasFullRankInput && hasReducedRankInput;
}

struct CheckNormSchedulePass
    : public mlir::PassWrapper<CheckNormSchedulePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckNormSchedulePass)

  llvm::StringRef getArgument() const final {
    return "wafer-check-norm-schedule";
  }

  llvm::StringRef getDescription() const final {
    return "validate staged RMSNorm/LayerNorm reduce and elementwise schedule";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    bool sawReduce = false;
    bool sawHiddenDimReduce = false;
    bool sawRsqrt = false;
    bool sawBroadcastMul = false;
    mlir::linalg::ReduceOp firstNonHiddenReduce;

    module.walk([&](mlir::linalg::ReduceOp reduce) {
      sawReduce = true;
      if (isHiddenDimReduce(reduce)) {
        sawHiddenDimReduce = true;
        return;
      }
      if (!firstNonHiddenReduce)
        firstNonHiddenReduce = reduce;
    });

    module.walk([&](mlir::linalg::GenericOp elementwise) {
      std::optional<wafer::ComputeElementwiseKind> kind =
          matchElementwiseGeneric(elementwise);
      if (kind == wafer::ComputeElementwiseKind::Rsqrt)
        sawRsqrt = true;
      if (isBroadcastMul(elementwise))
        sawBroadcastMul = true;
    });

    if (!sawReduce) {
      module.emitError("norm schedule requires at least one linalg.reduce");
      signalPassFailure();
      return;
    }
    if (!sawHiddenDimReduce) {
      if (firstNonHiddenReduce)
        firstNonHiddenReduce.emitOpError(
            "norm schedule requires a hidden-dimension reduction");
      else
        module.emitError(
            "norm schedule requires a hidden-dimension reduction");
      signalPassFailure();
      return;
    }
    if (!sawRsqrt) {
      module.emitError("norm schedule requires an rsqrt elementwise stage");
      signalPassFailure();
      return;
    }
    if (!sawBroadcastMul) {
      module.emitError(
          "norm schedule requires a rank-N/rank-(N-1) broadcast multiply "
          "stage");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckNormSchedulePass() {
  return std::make_unique<CheckNormSchedulePass>();
}

} // namespace wafer
