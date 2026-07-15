//===- WaferGroupToTileRegion.cpp - Group conversion entry points ---===//

#include "Internal.h"

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace wafer;
using namespace wafer::group_to_tile_region;

namespace wafer {
#define GEN_PASS_DEF_CONVERTGROUPTOTILEREGIONPASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

wafer::detail::CheckedStaticTileProductStatus
wafer::detail::checkedStaticTileProduct(llvm::ArrayRef<int64_t> ranges,
                                        llvm::ArrayRef<int64_t> tileSizes,
                                        uint64_t &product) {
  product = 1;
  if (ranges.size() != tileSizes.size())
    return CheckedStaticTileProductStatus::InvalidInput;

  for (auto [range, tileSize] : llvm::zip(ranges, tileSizes)) {
    if (range <= 0 || tileSize <= 0 || tileSize > range)
      return CheckedStaticTileProductStatus::InvalidInput;

    uint64_t unsignedRange = static_cast<uint64_t>(range);
    uint64_t unsignedTileSize = static_cast<uint64_t>(tileSize);
    uint64_t tileCount = unsignedRange / unsignedTileSize;
    tileCount += unsignedRange % unsignedTileSize != 0;
    if (product > std::numeric_limits<uint64_t>::max() / tileCount)
      return CheckedStaticTileProductStatus::Overflow;
    product *= tileCount;
  }
  return CheckedStaticTileProductStatus::Success;
}

wafer::detail::CompleteCandidateExpansionStatus
wafer::detail::checkCompleteCandidateExpansionBudget(
    uint64_t outputTileCount, llvm::ArrayRef<uint64_t> reductionChunkCounts,
    uint64_t &materializationCount) {
  materializationCount = 0;
  if (outputTileCount == 0 || reductionChunkCounts.empty() ||
      llvm::is_contained(reductionChunkCounts, uint64_t{0}))
    return CompleteCandidateExpansionStatus::InvalidInput;

  for (uint64_t reductionChunkCount : reductionChunkCounts) {
    if (outputTileCount >
        std::numeric_limits<uint64_t>::max() / reductionChunkCount)
      return CompleteCandidateExpansionStatus::CountOverflow;
    uint64_t rootMaterializationCount = outputTileCount * reductionChunkCount;
    if (materializationCount >
        std::numeric_limits<uint64_t>::max() - rootMaterializationCount)
      return CompleteCandidateExpansionStatus::CountOverflow;
    materializationCount += rootMaterializationCount;
  }

  if (materializationCount > kCompleteCandidateMaterializationBudget)
    return CompleteCandidateExpansionStatus::BudgetExceeded;
  return CompleteCandidateExpansionStatus::WithinBudget;
}

namespace {

static void configureGroupToTileRegionTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::arith::ArithDialect,
                         mlir::bufferization::BufferizationDialect,
                         mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect, wafer::WaferDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalOp<GroupOp, GroupYieldOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

} // namespace

mlir::LogicalResult
wafer::group_to_tile_region::convertGroupToTileRegionModuleInPlace(
    mlir::ModuleOp module, mlir::MLIRContext *context,
    int64_t currentLogicalRank, std::string *failureReason,
    bool suppressDiagnostics, bool verifyResult,
    bool populateFallbackFailureReason) {
  mlir::ConversionTarget target(*context);
  configureGroupToTileRegionTarget(target);

  mlir::RewritePatternSet patterns(context);
  populateGroupToTileRegionPatterns(patterns, failureReason,
                                    currentLogicalRank);

  auto applyConversion = [&]() {
    return mlir::applyFullConversion(module, target, std::move(patterns));
  };
  bool conversionSucceeded;
  if (suppressDiagnostics) {
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionSucceeded = mlir::succeeded(applyConversion());
  } else {
    conversionSucceeded = mlir::succeeded(applyConversion());
  }

  if (!conversionSucceeded) {
    if (populateFallbackFailureReason &&
        (!failureReason || failureReason->empty()))
      setFailureReason(failureReason, "group-to-tile-region lowering failed");
    return mlir::failure();
  }

  if (verifyResult && mlir::failed(mlir::verify(module))) {
    setFailureReason(failureReason,
                     "lowered tile-region module failed verifier");
    return mlir::failure();
  }

  return mlir::success();
}

namespace {

struct ConvertGroupToTileRegionPass
    : public wafer::impl::ConvertGroupToTileRegionPassBase<
          ConvertGroupToTileRegionPass> {
  using wafer::impl::ConvertGroupToTileRegionPassBase<
      ConvertGroupToTileRegionPass>::ConvertGroupToTileRegionPassBase;

  void runOnOperation() final {
    if (logicalRank < 0) {
      getOperation()->emitError()
          << "missing_logical_rank: group-to-tile-region conversion requires "
             "an explicit non-negative logical-rank";
      signalPassFailure();
      return;
    }
    std::string failureReason;
    if (mlir::succeeded(convertGroupToTileRegionModuleInPlace(
            getOperation(), &getContext(), logicalRank, &failureReason,
            /*suppressDiagnostics=*/false, /*verifyResult=*/false,
            /*populateFallbackFailureReason=*/false)))
      return;

    if (!failureReason.empty())
      getOperation().emitError(failureReason);
    else
      getOperation().emitError("group to tile-region conversion failed");
    signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult wafer::lowerGroupToTileRegionModule(
    GroupOp group, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank) {
  if (failureReason)
    failureReason->clear();

  module = detail::cloneGroupToStandaloneModule(group);
  return convertGroupToTileRegionModuleInPlace(
      *module, group.getContext(), currentLogicalRank, failureReason);
}

void wafer::dumpGroupToTileRegionModule(mlir::ModuleOp module,
                                        llvm::StringRef groupLabel,
                                        llvm::raw_ostream &os) {
  os << "wafer.group_to_tile_region group " << groupLabel << "\n";
  module.print(os);
  os << "\n";
}
