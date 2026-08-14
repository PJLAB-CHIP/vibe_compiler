//===- TileRegionSPMCapacityEvaluation.cpp - Region SPM evaluation -----===//

#include "TileRegionSPMCapacityEvaluation.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

namespace wafer::compiler::detail {
namespace {

static TileRegionSPMCapacityEvaluation classify(
    TileRegionSPMCapacityStatus status, TileRegionSPMCapacityPhase phase,
    llvm::StringRef detail, SPMMemoryPlanningFailure failure = {}) {
  TileRegionSPMCapacityEvaluation outcome;
  outcome.status = status;
  outcome.phase = phase;
  outcome.detail = detail.str();
  outcome.planningFailure = std::move(failure);
  return outcome;
}

class TileRegionEvaluationScope {
public:
  static mlir::FailureOr<std::unique_ptr<TileRegionEvaluationScope>>
  create(TileRegionOp source, std::string &detail) {
    auto evaluation = std::make_unique<TileRegionEvaluationScope>();
    auto *scratchBlock = new mlir::Block();
    evaluation->scratchRegion.push_back(scratchBlock);

    llvm::SmallVector<mlir::Location, 4> argumentLocations(
        source.getInputs().size(), source.getLoc());
    llvm::SmallVector<mlir::Type, 4> argumentTypes;
    argumentTypes.reserve(source.getInputs().size());
    for (mlir::Value input : source.getInputs())
      argumentTypes.push_back(input.getType());
    scratchBlock->addArguments(argumentTypes, argumentLocations);
    mlir::IRMapping mapping;
    for (auto [input, scratchArgument] :
         llvm::zip_equal(source.getInputs(), scratchBlock->getArguments()))
      mapping.map(input, scratchArgument);

    mlir::OpBuilder builder(source.getContext());
    builder.setInsertionPointToEnd(scratchBlock);
    evaluation->region = mlir::cast<TileRegionOp>(
        builder.clone(*source.getOperation(), mapping));
    if (mlir::failed(mlir::verify(evaluation->region))) {
      detail = "cloned TileRegion is not locally verifier-legal";
      return mlir::failure();
    }
    return evaluation;
  }

  TileRegionOp getRegion() const { return region; }

private:
  mlir::Region scratchRegion;
  TileRegionOp region;
};

} // namespace

llvm::StringRef
TileRegionSPMCapacityEvaluation::getPhaseDiagnosticLabel() const {
  switch (phase) {
  case TileRegionSPMCapacityPhase::None:
    return "none";
  case TileRegionSPMCapacityPhase::InputValidation:
    return "input-validation";
  case TileRegionSPMCapacityPhase::InstructionLowering:
    return "instruction-lowering";
  case TileRegionSPMCapacityPhase::StaticPacking:
    return "static-spm-packing";
  }
  llvm_unreachable("unknown TileRegion SPM capacity evaluation phase");
}

TileRegionSPMCapacityEvaluation evaluateTileRegionSPMCapacity(
    TileRegionOp region, TileRegionToInstrLoweringSession &loweringSession,
    llvm::raw_ostream &diagnostics) {
  wafer::support::ScopedCompileTimingSpan timing(
      "analysis", "tile-region-spm-capacity-evaluation",
      "tile-region-spm-capacity-evaluation");
  auto reportNonCapacityResult = [&](TileRegionSPMCapacityEvaluation outcome) {
    timing.markFailed();
    diagnostics << "wafer-compile: tile-region-spm-capacity outcome="
                << (outcome.capacityExceeded()
                        ? "capacity-exceeded"
                        : outcome.requiresFunctionScope()
                              ? "requires-function-scope"
                              : "analysis-failure")
                << " phase=" << outcome.getPhaseDiagnosticLabel()
                << " detail=" << outcome.detail << '\n';
    return outcome;
  };

  std::string detail;
  if (!region)
    return reportNonCapacityResult(classify(
        TileRegionSPMCapacityStatus::AnalysisFailure,
        TileRegionSPMCapacityPhase::InputValidation,
        "SPM capacity evaluation requires one materialized TileRegion"));
  bool hasCall = false;
  region.walk([&](mlir::CallOpInterface) { hasCall = true; });
  if (hasCall)
    return reportNonCapacityResult(classify(
        TileRegionSPMCapacityStatus::RequiresFunctionScope,
        TileRegionSPMCapacityPhase::InputValidation,
        "TileRegion contains a call and requires function-scoped SPM planning"));

  auto evaluation = TileRegionEvaluationScope::create(region, detail);
  if (mlir::failed(evaluation))
    return reportNonCapacityResult(classify(
        TileRegionSPMCapacityStatus::AnalysisFailure,
        TileRegionSPMCapacityPhase::InputValidation, detail));
  TileRegionOp isolatedRegion = (*evaluation)->getRegion();
  if (mlir::failed(convertTileRegionToInstr(isolatedRegion, loweringSession)) ||
      containsTileDataflowOperations(isolatedRegion.getOperation()) ||
      mlir::failed(rebuildRequiredNCCJoinsForIsolatedTileRegion(
          isolatedRegion)) ||
      mlir::failed(mlir::verify(isolatedRegion)))
    return reportNonCapacityResult(classify(
        TileRegionSPMCapacityStatus::AnalysisFailure,
        TileRegionSPMCapacityPhase::InstructionLowering,
        "TileRegion-to-Instr conversion or isolated completion analysis "
        "failed"));

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  SPMMemoryPlanningFailure spmFailure;
  if (mlir::failed(checkTileRegionSPMCapacity(
          isolatedRegion, memory.spmBase, memory.spmLimit, memory.spmAlignment,
          &spmFailure))) {
    const bool exact =
        spmFailure.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
    const bool requiresFunctionScope =
        spmFailure.kind == SPMMemoryPlanningFailureKind::UnsupportedLifetime;
    return reportNonCapacityResult(classify(
        exact ? TileRegionSPMCapacityStatus::CapacityExceeded
              : requiresFunctionScope
                    ? TileRegionSPMCapacityStatus::RequiresFunctionScope
                    : TileRegionSPMCapacityStatus::AnalysisFailure,
        TileRegionSPMCapacityPhase::StaticPacking,
        exact ? "static SPM packing proved a TileRegion capacity overflow"
              : requiresFunctionScope
                    ? "TileRegion lifetime requires function-scoped SPM planning"
                    : "TileRegion Instr SPM capacity analysis did not produce a proof",
        std::move(spmFailure)));
  }

  TileRegionSPMCapacityEvaluation outcome;
  outcome.status = TileRegionSPMCapacityStatus::Fits;
  outcome.phase = TileRegionSPMCapacityPhase::None;
  return outcome;
}

} // namespace wafer::compiler::detail
