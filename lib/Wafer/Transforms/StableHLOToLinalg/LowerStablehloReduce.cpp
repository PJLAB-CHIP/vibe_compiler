//===- LowerStablehloReduce.cpp - StableHLO reduce lowering --------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <optional>

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
namespace {

#ifdef WAFER_ENABLE_STABLEHLO
using ReduceCombinerBuilder = mlir::Value (*)(mlir::OpBuilder &,
                                              mlir::Location, mlir::Value,
                                              mlir::Value);
using ReduceCombinerSupport = bool (*)(mlir::Type);

struct ScalarReduceCombiner {
  ReduceCombinerSupport supports;
  ReduceCombinerBuilder build;
};

static bool areBlockArguments(mlir::Value lhs, mlir::Value rhs,
                              mlir::BlockArgument arg0,
                              mlir::BlockArgument arg1) {
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

static bool supportsFloatOrInteger(mlir::Type type) {
  return mlir::isa<mlir::FloatType, mlir::IntegerType>(type);
}

static mlir::Value createSumCombiner(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::Value accumulator,
                                     mlir::Value input) {
  mlir::Type elementType = accumulator.getType();
  if (mlir::isa<mlir::FloatType>(elementType))
    return builder.create<mlir::arith::AddFOp>(loc, accumulator, input);
  if (mlir::isa<mlir::IntegerType>(elementType))
    return builder.create<mlir::arith::AddIOp>(loc, accumulator, input);
  return {};
}

static mlir::Value createMaxCombiner(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::Value accumulator,
                                     mlir::Value input) {
  mlir::Type elementType = accumulator.getType();
  if (mlir::isa<mlir::FloatType>(elementType))
    return builder.create<mlir::arith::MaximumFOp>(loc, accumulator, input);
  if (mlir::isa<mlir::IntegerType>(elementType))
    return builder.create<mlir::arith::MaxSIOp>(loc, accumulator, input);
  return {};
}

static constexpr ScalarReduceCombiner kSumCombiner{supportsFloatOrInteger,
                                                   createSumCombiner};
static constexpr ScalarReduceCombiner kMaxCombiner{supportsFloatOrInteger,
                                                   createMaxCombiner};

static std::optional<ScalarReduceCombiner>
getSupportedReduceCombiner(mlir::stablehlo::ReduceOp reduce) {
  if (reduce.getInputs().size() != 1 || reduce.getInitValues().size() != 1 ||
      reduce->getNumResults() != 1)
    return std::nullopt;

  mlir::Block &body = reduce.getBody().front();
  if (body.getNumArguments() != 2)
    return std::nullopt;

  auto terminator =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(body.getTerminator());
  if (!terminator || terminator.getResults().size() != 1)
    return std::nullopt;

  mlir::Value result = *terminator.getResults().begin();
  if (auto add = result.getDefiningOp<mlir::stablehlo::AddOp>())
    if (areBlockArguments(add.getLhs(), add.getRhs(), body.getArgument(0),
                          body.getArgument(1)))
      return kSumCombiner;

  if (auto max = result.getDefiningOp<mlir::stablehlo::MaxOp>())
    if (areBlockArguments(max.getLhs(), max.getRhs(), body.getArgument(0),
                          body.getArgument(1)))
      return kMaxCombiner;

  return std::nullopt;
}

static bool lowerReduce(mlir::stablehlo::ReduceOp reduce,
                        const ScalarReduceCombiner &combiner) {
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(
      reduce.getInputs().front().getType());
  auto initType = mlir::dyn_cast<mlir::RankedTensorType>(
      reduce.getInitValues().front().getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(reduce->getResult(0).getType());
  if (!inputType || !initType || !resultType || initType.getRank() != 0 ||
      !resultType.hasStaticShape())
    return false;
  if (!combiner.supports(resultType.getElementType()))
    return false;

  mlir::OpBuilder builder(reduce);
  mlir::Value scalarInit = builder.create<mlir::tensor::ExtractOp>(
      reduce.getLoc(), reduce.getInitValues().front(), mlir::ValueRange{});
  mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
      reduce.getLoc(), resultType.getShape(), resultType.getElementType());
  auto filled = builder.create<mlir::linalg::FillOp>(
      reduce.getLoc(), mlir::TypeRange{resultType},
      mlir::ValueRange{scalarInit}, mlir::ValueRange{empty});

  auto lowered = builder.create<mlir::linalg::ReduceOp>(
      reduce.getLoc(), mlir::ValueRange{reduce.getInputs().front()},
      mlir::ValueRange{filled.getResult(0)}, reduce.getDimensions(),
      [combiner](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
                 mlir::ValueRange args) {
        mlir::Value combined =
            combiner.build(nestedBuilder, nestedLoc, args[1], args[0]);
        assert(combined && "prechecked scalar reduce combiner failed");
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, combined);
      });

  reduce->getResult(0).replaceAllUsesWith(lowered->getResult(0));
  reduce.erase();
  return true;
}
#endif

struct LowerStablehloReducePass
    : public mlir::PassWrapper<LowerStablehloReducePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStablehloReducePass)

  llvm::StringRef getArgument() const final {
    return "wafer-lower-stablehlo-reduce";
  }

  llvm::StringRef getDescription() const final {
    return "lower simple StableHLO reduce sum/max ops to linalg.reduce";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect>();
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::stablehlo::ReduceOp> reduces;
    getOperation().walk([&](mlir::stablehlo::ReduceOp reduce) {
      if (getSupportedReduceCombiner(reduce))
        reduces.push_back(reduce);
    });

    for (mlir::stablehlo::ReduceOp reduce : reduces)
      (void)lowerReduce(reduce, *getSupportedReduceCombiner(reduce));
#endif
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerStablehloReducePass() {
  return std::make_unique<LowerStablehloReducePass>();
}

} // namespace wafer
