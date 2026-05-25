//===- CheckRootTileCandidates.cpp - Root tile feasibility skeleton -------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
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
  return lhs != mlir::ShapedType::kDynamic && rhs != mlir::ShapedType::kDynamic &&
         lhs != rhs;
}

static bool isSupportedElementwiseKind(mlir::linalg::ElementwiseKind kind) {
  switch (kind) {
  case mlir::linalg::ElementwiseKind::add:
  case mlir::linalg::ElementwiseKind::sub:
  case mlir::linalg::ElementwiseKind::mul:
  case mlir::linalg::ElementwiseKind::div:
  case mlir::linalg::ElementwiseKind::max_signed:
  case mlir::linalg::ElementwiseKind::min_signed:
  case mlir::linalg::ElementwiseKind::negf:
  case mlir::linalg::ElementwiseKind::reciprocal:
  case mlir::linalg::ElementwiseKind::sqrt:
  case mlir::linalg::ElementwiseKind::rsqrt:
  case mlir::linalg::ElementwiseKind::exp:
  case mlir::linalg::ElementwiseKind::tanh:
    return true;
  default:
    return false;
  }
}

static bool isLimitedBroadcastElementwiseRoot(
    mlir::linalg::ElementwiseOp elementwise) {
  if (!isSupportedElementwiseKind(elementwise.getKind()))
    return false;
  if (elementwise->getNumResults() != 1 || elementwise.getOutputs().size() != 1)
    return false;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(elementwise->getResult(0).getType());
  if (!resultType || elementwise.getOutputs()[0].getType() != resultType)
    return false;

  llvm::SmallVector<mlir::AffineMap> maps = elementwise.getIndexingMapsArray();
  if (maps.size() != elementwise.getInputs().size() + elementwise.getOutputs().size())
    return false;
  mlir::AffineMap resultMap = maps.back();
  if (resultMap.getNumDims() != resultType.getRank() ||
      resultMap.getNumSymbols() != 0 || !resultMap.isIdentity())
    return false;

  for (auto [index, input] : llvm::enumerate(elementwise.getInputs())) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType ||
        inputType.getElementType() != resultType.getElementType())
      return false;

    mlir::AffineMap inputMap = maps[index];
    if (inputMap.getNumDims() != resultType.getRank() ||
        inputMap.getNumSymbols() != 0 ||
        inputMap.getNumResults() != inputType.getRank() ||
        !inputMap.isProjectedPermutation())
      return false;

    for (auto [dim, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr || dimExpr.getPosition() >= resultType.getRank())
        return false;
      if (hasStaticMismatch(inputType.getDimSize(dim),
                            resultType.getDimSize(dimExpr.getPosition())))
        return false;
    }
  }
  return true;
}

static std::optional<RootTileCandidate>
buildM0RootTileCandidate(wafer::GroupOp group, std::string &reason) {
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
          "group inputs must be static ranked tensors for M0 feasibility");
  }
  for (mlir::Value out : group.getOuts()) {
    if (!isStaticRankedTensor(out.getType()))
      return FeasibilityResult::failure(
          "group outs must be static ranked tensors for M0 feasibility");
  }
  return FeasibilityResult::success();
}

static FeasibilityResult
checkM0CandidateFeasibility(wafer::GroupOp group,
                            const RootTileCandidate &candidate) {
  if (FeasibilityResult boundary = checkBoundaryTensors(group);
      !boundary.feasible)
    return boundary;

  mlir::Operation *root = getSingleBodyOp(group);
  if (!root)
    return FeasibilityResult::failure(
        "M0 feasibility requires exactly one root operation");

  if (mlir::isa<mlir::linalg::MatmulOp>(root)) {
    if (candidate.shape.size() != 2)
      return FeasibilityResult::failure(
          "M0 matmul feasibility requires a rank-2 root tile candidate");
    return FeasibilityResult::success();
  }

  auto elementwise = mlir::dyn_cast<mlir::linalg::ElementwiseOp>(root);
  if (elementwise && isLimitedBroadcastElementwiseRoot(elementwise))
    return FeasibilityResult::success();

  return FeasibilityResult::failure(
      "M0 feasibility requires a linalg.matmul or limited-broadcast "
      "linalg.elementwise root");
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
          buildM0RootTileCandidate(group, reason);
      if (!candidate) {
        group.emitOpError("has no feasible root tile candidate: ") << reason;
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      FeasibilityResult feasibility =
          checkM0CandidateFeasibility(group, *candidate);
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
