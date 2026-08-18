//===- SPMCapacityEvaluation.cpp - Scoped fixed-capacity SPM probes ----===//

#include "SPMCapacityEvaluation.h"

#include "CompilationInternal.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Pipelines/Pipelines.h"
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

static TileRegionSPMCapacityEvaluation
classifyRegion(TileRegionSPMCapacityStatus status,
               TileRegionSPMCapacityPhase phase, llvm::StringRef detail,
               TileMemoryPlanningFailure failure = {}) {
  TileRegionSPMCapacityEvaluation outcome;
  outcome.status = status;
  outcome.phase = phase;
  outcome.detail = detail.str();
  outcome.planningFailure = std::move(failure);
  return outcome;
}

static TileFunctionSPMCapacityEvaluation
classifyFunction(TileFunctionSPMCapacityStatus status,
                 TileFunctionSPMCapacityPhase phase, llvm::StringRef detail,
                 TileMemoryPlanningFailure failure = {}) {
  TileFunctionSPMCapacityEvaluation outcome;
  outcome.status = status;
  outcome.phase = phase;
  outcome.detail = detail.str();
  outcome.planningFailure = std::move(failure);
  return outcome;
}

/// Private evaluation clone for one TileRegion. Inputs are rebound to scratch
/// block arguments; every cloned operation result is recorded in the mapping
/// so caller relations can be remapped into the clone and stay current across
/// the subsequent Instr conversion.
class TileRegionEvaluationScope {
public:
  static mlir::FailureOr<std::unique_ptr<TileRegionEvaluationScope>> create(
      TileRegionOp source, const StructuredMaterializationRelations *relations,
      std::string &detail) {
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
    if (relations) {
      mlir::FailureOr<StructuredMaterializationRelations> remapped =
          remapStructuredBufferRelationsComplete(*relations, mapping);
      if (mlir::failed(remapped)) {
        detail =
            "region probe relation remap is incomplete: a relation buffer "
            "was not mapped into the scratch clone";
        return mlir::failure();
      }
      evaluation->relations = std::move(*remapped);
      evaluation->relationsProvided = true;
    }
    evaluation->listener =
        std::make_unique<StructuredBufferReplacementListener>(
            evaluation->relations);
    return evaluation;
  }

  TileRegionOp getRegion() const { return region; }

  StructuredBufferReplacementListener *getListener() const {
    return listener.get();
  }

  bool relationsCurrent() {
    return !relationsProvided ||
           mlir::succeeded(checkStructuredBufferRelationsCurrent(
               region.getOperation(), relations));
  }

  const StructuredMaterializationRelations &getRelations() const {
    return relations;
  }

private:
  mlir::Region scratchRegion;
  TileRegionOp region;
  StructuredMaterializationRelations relations;
  bool relationsProvided = false;
  std::unique_ptr<StructuredBufferReplacementListener> listener;
};

static void reportRegionOutcome(const TileRegionSPMCapacityEvaluation &outcome,
                                llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-compile: tile-region-spm-capacity outcome="
              << (outcome.capacityExceeded() ? "capacity-exceeded"
                  : outcome.requiresFunctionScope()
                      ? "requires-function-scope"
                      : "analysis-failure")
              << " phase=" << outcome.getPhaseDiagnosticLabel()
              << " detail=" << outcome.detail << '\n';
}

static void reportFunctionOutcome(
    const TileFunctionSPMCapacityEvaluation &outcome,
    llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-compile: tile-function-spm-capacity outcome="
              << (outcome.capacityExceeded() ? "capacity-exceeded"
                  : outcome.unsupportedLifetime() ? "unsupported-lifetime"
                                                  : "analysis-failure")
              << " phase=" << outcome.getPhaseDiagnosticLabel()
              << " detail=" << outcome.detail << '\n';
}

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

llvm::StringRef
TileFunctionSPMCapacityEvaluation::getPhaseDiagnosticLabel() const {
  switch (phase) {
  case TileFunctionSPMCapacityPhase::None:
    return "none";
  case TileFunctionSPMCapacityPhase::InputValidation:
    return "input-validation";
  case TileFunctionSPMCapacityPhase::InstructionLowering:
    return "instruction-lowering";
  case TileFunctionSPMCapacityPhase::PlanningPreparation:
    return "planning-preparation";
  case TileFunctionSPMCapacityPhase::StaticPacking:
    return "static-spm-packing";
  }
  llvm_unreachable("unknown Tile function SPM capacity evaluation phase");
}

TileRegionSPMCapacityEvaluation
evaluateTileRegionSPMCapacity(TileRegionOp region,
                              TileRegionToInstrLoweringSession &loweringSession,
                              llvm::raw_ostream &diagnostics,
                              const StructuredMaterializationRelations
                                  *relations) {
  wafer::support::ScopedCompileTimingSpan timing(
      "analysis", "tile-region-spm-capacity-evaluation",
      "tile-region-spm-capacity-evaluation");
  auto report = [&](TileRegionSPMCapacityEvaluation outcome) {
    if (outcome.status != TileRegionSPMCapacityStatus::Fits)
      timing.markFailed();
    reportRegionOutcome(outcome, diagnostics);
    return outcome;
  };

  std::string detail;
  if (!region)
    return report(classifyRegion(
        TileRegionSPMCapacityStatus::AnalysisFailure,
        TileRegionSPMCapacityPhase::InputValidation,
        "SPM capacity evaluation requires one materialized TileRegion"));
  bool hasCall = false;
  region.walk([&](mlir::CallOpInterface) { hasCall = true; });
  if (hasCall)
    return report(classifyRegion(
        TileRegionSPMCapacityStatus::RequiresFunctionScope,
        TileRegionSPMCapacityPhase::InputValidation,
        "TileRegion contains a call and requires function-scoped SPM "
        "planning"));

  auto evaluation = TileRegionEvaluationScope::create(region, relations,
                                                      detail);
  if (mlir::failed(evaluation))
    return report(classifyRegion(
        TileRegionSPMCapacityStatus::AnalysisFailure,
        TileRegionSPMCapacityPhase::InputValidation, detail));
  TileRegionOp isolatedRegion = (*evaluation)->getRegion();
  std::string conversionDiagnostics;
  {
    llvm::raw_string_ostream stream(conversionDiagnostics);
    mlir::ScopedDiagnosticHandler captureHandler(
        isolatedRegion.getContext(),
        [&](mlir::Diagnostic &diag) {
          diag.print(stream);
          stream << "\n";
        });
    if (mlir::failed(convertTileRegionToInstr(
            isolatedRegion, loweringSession, (*evaluation)->getListener())))
      return report(classifyRegion(
          TileRegionSPMCapacityStatus::AnalysisFailure,
          TileRegionSPMCapacityPhase::InstructionLowering,
          "TileRegion-to-Instr conversion failed: " + conversionDiagnostics));
  }
  {
    std::string stage;
    if (containsTileDataflowOperations(isolatedRegion.getOperation()))
      stage = "residual tile dataflow after conversion";
    else if (!(*evaluation)->relationsCurrent()) {
      stage = "relations left the current IR";
    } else if (mlir::failed(rebuildRequiredNCCJoinsForIsolatedTileRegion(
                   isolatedRegion)))
      stage = "isolated required-join rebuild failed";
    else if (!(*evaluation)->relationsCurrent())
      stage = "required-join rebuild changed the relation domain";
    else if (mlir::failed(mlir::verify(isolatedRegion)))
      stage = "isolated region verify failed";
    if (!stage.empty())
      return report(classifyRegion(
          TileRegionSPMCapacityStatus::AnalysisFailure,
          TileRegionSPMCapacityPhase::InstructionLowering, stage));
  }

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  SPMMemoryPlanningFailure spmFailure;
  if (mlir::failed(checkTileRegionSPMCapacity(
          isolatedRegion, memory.spmBase, memory.spmLimit, memory.spmAlignment,
          &spmFailure))) {
    const bool exact =
        spmFailure.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
    const bool requiresFunctionScope =
        spmFailure.kind == SPMMemoryPlanningFailureKind::UnsupportedLifetime;
    TileMemoryPlanningFailure attributed = convertSPMMemoryPlanningFailure(
        spmFailure, relations ? (*evaluation)->getRelations()
                              : StructuredMaterializationRelations{});
    return report(classifyRegion(
        exact ? TileRegionSPMCapacityStatus::CapacityExceeded
        : requiresFunctionScope
            ? TileRegionSPMCapacityStatus::RequiresFunctionScope
            : TileRegionSPMCapacityStatus::AnalysisFailure,
        TileRegionSPMCapacityPhase::StaticPacking,
        exact ? "static SPM packing proved a TileRegion capacity overflow"
        : requiresFunctionScope
            ? "TileRegion lifetime requires function-scoped SPM planning"
            : "TileRegion Instr SPM capacity analysis did not produce a proof",
        std::move(attributed)));
  }

  TileRegionSPMCapacityEvaluation outcome;
  outcome.status = TileRegionSPMCapacityStatus::Fits;
  outcome.phase = TileRegionSPMCapacityPhase::None;
  return outcome;
}

TileFunctionSPMCapacityEvaluation evaluateTileFunctionSPMCapacity(
    mlir::OwningOpRef<mlir::ModuleOp> scratch,
    StructuredMaterializationRelations cloneRelations,
    llvm::raw_ostream &diagnostics) {
  wafer::support::ScopedCompileTimingSpan timing(
      "analysis", "tile-function-spm-capacity-evaluation",
      "tile-function-spm-capacity-evaluation");
  auto report = [&](TileFunctionSPMCapacityEvaluation outcome) {
    if (outcome.status != TileFunctionSPMCapacityStatus::Fits)
      timing.markFailed();
    reportFunctionOutcome(outcome, diagnostics);
    return outcome;
  };

  if (!scratch) {
    return report(classifyFunction(
        TileFunctionSPMCapacityStatus::AnalysisFailure,
        TileFunctionSPMCapacityPhase::InputValidation,
        "function-scoped SPM capacity evaluation requires one owned Tile "
        "entry module"));
  }

  mlir::func::FuncOp function;
  for (mlir::func::FuncOp candidate : scratch->getOps<mlir::func::FuncOp>()) {
    if (candidate.isExternal())
      continue;
    if (function) {
      return report(classifyFunction(
          TileFunctionSPMCapacityStatus::AnalysisFailure,
          TileFunctionSPMCapacityPhase::InputValidation,
          "function-scoped SPM capacity evaluation requires exactly one "
          "defined Tile entry"));
    }
    function = candidate;
  }
  if (!function) {
    return report(classifyFunction(
        TileFunctionSPMCapacityStatus::AnalysisFailure,
        TileFunctionSPMCapacityPhase::InputValidation,
        "function-scoped SPM capacity evaluation requires exactly one "
        "defined Tile entry"));
  }
  if (mlir::failed(mlir::verify(*scratch))) {
    return report(
        classifyFunction(TileFunctionSPMCapacityStatus::AnalysisFailure,
                         TileFunctionSPMCapacityPhase::InputValidation,
                         "owned Tile entry module is not verifier-legal"));
  }
  if (mlir::failed(checkStructuredBufferRelationsCurrent(
          function.getOperation(), cloneRelations))) {
    return report(classifyFunction(
        TileFunctionSPMCapacityStatus::AnalysisFailure,
        TileFunctionSPMCapacityPhase::InputValidation,
        "function-scoped probe relations are outside the owned Tile entry"));
  }

  // The exact per-Tile sequence of the final gate: TileRegion-to-Instr with
  // relation-preserving replacement, function-level required NCC join
  // placement, memory-planning preparation and SPM offset assignment.
  llvm::SmallVector<TileRegionOp, 4> regions;
  scratch->walk([&](TileRegionOp region) { regions.push_back(region); });
  TileRegionToInstrLoweringSession loweringSession(*scratch->getContext());
  StructuredBufferReplacementListener replacementListener(cloneRelations);
  for (TileRegionOp region : regions)
    if (mlir::failed(convertTileRegionToInstr(region, loweringSession,
                                              &replacementListener))) {
      return report(classifyFunction(
          TileFunctionSPMCapacityStatus::AnalysisFailure,
          TileFunctionSPMCapacityPhase::InstructionLowering,
          "TileRegion-to-Instr conversion failed in the function-scoped "
          "probe"));
    }
  if (!replacementListener.preservedAllRelations() ||
      mlir::failed(checkStructuredBufferRelationsCurrent(
          scratch->getOperation(), cloneRelations))) {
    return report(classifyFunction(
        TileFunctionSPMCapacityStatus::AnalysisFailure,
        TileFunctionSPMCapacityPhase::InstructionLowering,
        "TileRegion-to-Instr conversion did not preserve every structured "
        "buffer relation in the function-scoped probe"));
  }
  if (mlir::failed(runPassPipeline(*scratch, "required-ncc-join-placement",
                                   [](mlir::OpPassManager &manager) {
                                     wafer::addRequiredNCCJoinPlacementPass(
                                         manager.nest<mlir::func::FuncOp>());
                                   })) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(
          scratch->getOperation(), cloneRelations))) {
    return report(classifyFunction(
        TileFunctionSPMCapacityStatus::AnalysisFailure,
        TileFunctionSPMCapacityPhase::InstructionLowering,
        "function-level required NCC join placement failed in the "
        "function-scoped probe"));
  }

  if (mlir::failed(runPassPipeline(
          *scratch, "instr-memory-planning-preparation",
          wafer::buildPrepareInstrForMemoryPlanningPipeline)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(
          scratch->getOperation(), cloneRelations)) ||
      mlir::failed(mlir::verify(*scratch))) {
    return report(classifyFunction(
        TileFunctionSPMCapacityStatus::AnalysisFailure,
        TileFunctionSPMCapacityPhase::PlanningPreparation,
        "memory-planning preparation failed in the function-scoped probe"));
  }

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  SPMMemoryPlanningFailure spmFailure;
  PlanSPMMemoryPassOptions spmOptions;
  spmOptions.spmBase = memory.spmBase;
  spmOptions.spmLimit = memory.spmLimit;
  spmOptions.spmAlignment = memory.spmAlignment;
  if (mlir::failed(runPassPipeline(
          *scratch, "tile-spm-planning",
          [&](mlir::OpPassManager &manager) {
            wafer::addAssignSPMOffsetsPass(manager, spmOptions, &spmFailure);
          }))) {
    TileMemoryPlanningFailure attributed = convertSPMMemoryPlanningFailure(
        spmFailure, cloneRelations);
    const bool exact =
        spmFailure.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
    const bool unsupported =
        spmFailure.kind == SPMMemoryPlanningFailureKind::UnsupportedLifetime;
    return report(classifyFunction(
        exact ? TileFunctionSPMCapacityStatus::CapacityExceeded
        : unsupported ? TileFunctionSPMCapacityStatus::UnsupportedLifetime
                      : TileFunctionSPMCapacityStatus::AnalysisFailure,
        TileFunctionSPMCapacityPhase::StaticPacking,
        exact ? "function-scoped SPM planning proved a capacity overflow"
        : unsupported
            ? "function-scoped SPM planning cannot express the Tile lifetime"
            : "function-scoped SPM planning did not produce a proof",
        std::move(attributed)));
  }

  TileFunctionSPMCapacityEvaluation outcome;
  outcome.status = TileFunctionSPMCapacityStatus::Fits;
  outcome.phase = TileFunctionSPMCapacityPhase::None;
  return outcome;
}

} // namespace wafer::compiler::detail
