//===- CheckRootTileCandidates.cpp - Root tile feasibility -------===//

#include "Wafer/Transforms/Passes.h"

#include "Support/AttentionGemmUtils.h"
#include "Support/ElementwiseUtils.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <utility>

namespace wafer {
namespace {

struct RootTileCandidate {
  llvm::SmallVector<int64_t, 4> shape;
};

struct FeasibilityResult {
  static FeasibilityResult success() { return {true, ""}; }
  static FeasibilityResult failure(std::string reason) {
    return {false, std::move(reason)};
  }

  bool feasible;
  std::string reason;
};

static mlir::Operation *getSingleBodyOp(wafer::GroupOp group) {
  mlir::Operation *root = nullptr;
  mlir::Block &block = group.getBody().front();
  for (mlir::Operation &op : block) {
    if (&op == block.getTerminator())
      continue;
    if (root)
      return nullptr;
    root = &op;
  }
  return root;
}

static bool hasStaticMismatch(int64_t lhs, int64_t rhs) {
  return lhs != mlir::ShapedType::kDynamic &&
         rhs != mlir::ShapedType::kDynamic && lhs != rhs;
}

static std::optional<mlir::Attribute>
getScalarConstantAttr(mlir::Attribute attr) {
  if (mlir::isa<mlir::FloatAttr, mlir::IntegerAttr>(attr))
    return attr;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return std::nullopt;
  return dense.getSplatValue<mlir::Attribute>();
}

static std::optional<mlir::Attribute> getScalarConstantAttr(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    return getScalarConstantAttr(constant.getValue());
  if (auto extract = value.getDefiningOp<mlir::tensor::ExtractOp>()) {
    if (extract.getIndices().empty())
      return getScalarConstantAttr(extract.getTensor());
  }
  return std::nullopt;
}

static std::optional<mlir::Attribute>
getReduceInitValueAttr(mlir::Value output) {
  auto fill = output.getDefiningOp<mlir::linalg::FillOp>();
  if (!fill || fill.getInputs().size() != 1)
    return std::nullopt;
  return getScalarConstantAttr(fill.getInputs()[0]);
}

static bool
isLimitedBroadcastElementwiseRoot(mlir::linalg::GenericOp generic) {
  return isLimitedBroadcastElementwiseGeneric(generic);
}

static bool areBlockArguments(mlir::Value lhs, mlir::Value rhs,
                              mlir::BlockArgument arg0,
                              mlir::BlockArgument arg1) {
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

static std::optional<wafer::ComputeReduceKind>
mapReduceKind(mlir::linalg::ReduceOp reduce) {
  if (reduce->getNumResults() != 1 || reduce.getInputs().size() != 1 ||
      reduce.getInits().size() != 1 || reduce.getRegion().empty())
    return std::nullopt;

  mlir::Block &body = reduce.getRegion().front();
  if (body.getNumArguments() != 2 || !body.getTerminator() ||
      body.getTerminator()->getNumOperands() != 1)
    return std::nullopt;

  mlir::Value yielded = body.getTerminator()->getOperand(0);
  if (auto add = yielded.getDefiningOp<mlir::arith::AddFOp>())
    if (areBlockArguments(add->getOperand(0), add->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Sum;
  if (auto add = yielded.getDefiningOp<mlir::arith::AddIOp>())
    if (areBlockArguments(add->getOperand(0), add->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Sum;
  if (auto max = yielded.getDefiningOp<mlir::arith::MaximumFOp>())
    if (areBlockArguments(max->getOperand(0), max->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Max;
  if (auto max = yielded.getDefiningOp<mlir::arith::MaxSIOp>())
    if (areBlockArguments(max->getOperand(0), max->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Max;
  if (auto min = yielded.getDefiningOp<mlir::arith::MinimumFOp>())
    if (areBlockArguments(min->getOperand(0), min->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Min;
  if (auto min = yielded.getDefiningOp<mlir::arith::MinSIOp>())
    if (areBlockArguments(min->getOperand(0), min->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Min;

  return std::nullopt;
}

static bool isSupportedReduceRoot(wafer::GroupOp group,
                                  mlir::linalg::ReduceOp reduce) {
  if (!mapReduceKind(reduce) || group.getOuts().size() != 1 ||
      !getReduceInitValueAttr(group.getOuts()[0]))
    return false;

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(reduce.getInputs()[0].getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(reduce->getResult(0).getType());
  if (!inputType || !resultType || reduce.getInits()[0].getType() != resultType)
    return false;
  if (inputType.getElementType() != resultType.getElementType())
    return false;

  llvm::DenseSet<int64_t> reducedDims;
  for (int64_t dim : reduce.getDimensions()) {
    if (dim < 0 || dim >= inputType.getRank())
      return false;
    if (!reducedDims.insert(dim).second)
      return false;
  }
  if (resultType.getRank() !=
      inputType.getRank() - static_cast<int64_t>(reducedDims.size()))
    return false;

  int64_t resultDim = 0;
  for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
    if (reducedDims.contains(inputDim))
      continue;
    if (hasStaticMismatch(inputType.getDimSize(inputDim),
                          resultType.getDimSize(resultDim)))
      return false;
    ++resultDim;
  }
  return true;
}

static std::optional<RootTileCandidate>
buildRootTileCandidate(wafer::GroupOp group, std::string &reason) {
  if (group.getNumResults() != 1) {
    reason = "multi-result group requires traversal-domain split";
    return std::nullopt;
  }

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(group.getResult(0).getType());
  if (!resultType) {
    reason = "root result must be a ranked tensor";
    return std::nullopt;
  }
  if (resultType.getRank() <= 0) {
    reason = "scalar root result requires explicit scalar tile policy";
    return std::nullopt;
  }
  if (!resultType.hasStaticShape()) {
    reason = "dynamic result shape requires a bounded tile policy";
    return std::nullopt;
  }

  RootTileCandidate candidate;
  for (int64_t dim : resultType.getShape()) {
    if (dim <= 0) {
      reason = "empty result shape has no root tile candidate";
      return std::nullopt;
    }
    candidate.shape.push_back(dim);
  }
  return candidate;
}

static bool isStaticRankedTensor(mlir::Type type) {
  auto rankedTensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return rankedTensor && rankedTensor.hasStaticShape();
}

static FeasibilityResult checkBoundaryTensors(wafer::GroupOp group) {
  for (mlir::Value input : group.getInputs()) {
    if (!isStaticRankedTensor(input.getType()))
      return FeasibilityResult::failure(
          "group inputs must be static ranked tensors for root tile feasibility");
  }
  for (mlir::Value out : group.getOuts()) {
    if (!isStaticRankedTensor(out.getType()))
      return FeasibilityResult::failure(
          "group outs must be static ranked tensors for root tile feasibility");
  }
  return FeasibilityResult::success();
}

static FeasibilityResult
checkRootTileFeasibility(wafer::GroupOp group,
                         const RootTileCandidate &candidate) {
  if (FeasibilityResult boundary = checkBoundaryTensors(group);
      !boundary.feasible)
    return boundary;

  mlir::Operation *root = getSingleBodyOp(group);
  if (!root)
    return FeasibilityResult::failure(
        "root tile feasibility requires exactly one root operation");

  if (mlir::isa<mlir::linalg::MatmulOp>(root)) {
    if (candidate.shape.size() != 2)
      return FeasibilityResult::failure(
          "root tile matmul feasibility requires a rank-2 root tile candidate");
    return FeasibilityResult::success();
  }

  auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(root);
  if (generic && matchAttentionGemm(generic))
    return FeasibilityResult::success();
  if (generic && isLimitedBroadcastElementwiseRoot(generic))
    return FeasibilityResult::success();

  auto reduce = mlir::dyn_cast<mlir::linalg::ReduceOp>(root);
  if (reduce && isSupportedReduceRoot(group, reduce))
    return FeasibilityResult::success();

  return FeasibilityResult::failure(
      "root tile feasibility requires a linalg.matmul, supported attention "
      "linalg.generic contraction, limited-broadcast elementwise "
      "linalg.generic, or "
      "supported linalg.reduce root");
}

struct CheckRootTileCandidatesPass
    : public mlir::PassWrapper<CheckRootTileCandidatesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckRootTileCandidatesPass)

  llvm::StringRef getArgument() const final {
    return "wafer-check-root-tile-candidates";
  }

  llvm::StringRef getDescription() const final {
    return "check pass-local root tile candidates for Wafer groups";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, wafer::WaferDialect>();
  }

  void runOnOperation() final {
    bool failed = false;
    getOperation().walk([&](wafer::GroupOp group) {
      std::string reason;
      std::optional<RootTileCandidate> candidate =
          buildRootTileCandidate(group, reason);
      if (!candidate) {
        group.emitOpError("has no feasible root tile candidate: ") << reason;
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      FeasibilityResult feasibility =
          checkRootTileFeasibility(group, *candidate);
      if (!feasibility.feasible) {
        group.emitOpError("has no feasible root tile candidate: ")
            << feasibility.reason;
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      return mlir::WalkResult::advance();
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckRootTileCandidatesPass() {
  return std::make_unique<CheckRootTileCandidatesPass>();
}

} // namespace wafer
