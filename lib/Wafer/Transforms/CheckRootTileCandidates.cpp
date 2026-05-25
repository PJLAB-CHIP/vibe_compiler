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
  if (resultType.getRank() != 2) {
    reason = "only rank-2 root results are supported by the M0 candidate";
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
  if (candidate.shape.size() != 2)
    return FeasibilityResult::failure("root tile candidate must be rank-2");

  if (FeasibilityResult boundary = checkBoundaryTensors(group);
      !boundary.feasible)
    return boundary;

  bool hasMatmulRoot = false;
  group.walk([&](mlir::linalg::MatmulOp) { hasMatmulRoot = true; });
  if (!hasMatmulRoot)
    return FeasibilityResult::failure(
        "M0 feasibility requires a linalg.matmul root");

  return FeasibilityResult::success();
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
