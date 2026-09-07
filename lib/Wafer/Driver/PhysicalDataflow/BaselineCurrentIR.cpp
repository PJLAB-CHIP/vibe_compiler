//===- BaselineCurrentIR.cpp - Deterministic current-IR baseline -------===//

#include "BaselineCurrentIR.h"

#include "PhysicalDataflowInstrumentation.h"
#include "StructuredProgramAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Transforms/Linalg/CommunicationRegionClosure.h"
#include "Wafer/Transforms/Linalg/OnlineAttentionDecomposition.h"
#include "Wafer/Transforms/Linalg/SpatialRegionMaterialization.h"
#include "Wafer/Transforms/Linalg/StructuredGraphNormalization.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace wafer::compiler::detail {
namespace {

static ExecutableCompilationResult fail(ExecutableCompilationStatus status,
                                        llvm::StringRef gate,
                                        llvm::StringRef detail) {
  ExecutableCompilationResult result;
  result.status = status;
  result.gate = gate.str();
  result.detail = detail.str();
  return result;
}

static mlir::FailureOr<mlir::func::FuncOp>
getProgramFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp function;
  for (mlir::func::FuncOp candidate : module.getOps<mlir::func::FuncOp>()) {
    if (candidate.isExternal())
      continue;
    if (function)
      return mlir::failure();
    function = candidate;
  }
  return function ? mlir::FailureOr<mlir::func::FuncOp>(function)
                  : mlir::FailureOr<mlir::func::FuncOp>(mlir::failure());
}

static std::string
demandFailureDetail(const analysis::ExactDemandOutcome &outcome) {
  return std::visit(
      [](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
          return {};
        else
          return value.detail;
      },
      outcome);
}

static std::string
rootWorkFailureDetail(const RootWorkCollectionOutcome &outcome) {
  return std::visit(
      [](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, RootWorkCollection>)
          return {};
        else
          return value.detail;
      },
      outcome);
}

struct FixedTemporalChoice {
  TemporalChoice choice;
  bool refined = false;
};

static mlir::FailureOr<FixedTemporalChoice>
buildFixedTemporalChoice(const TemporalDomain &domain, uint32_t refinement,
                         std::string &detail) {
  if (refinement == 0) {
    TemporalSuccessor first = domain.getFirstChoice();
    if (first.getKind() != TemporalSuccessorKind::Choice ||
        !first.getChoice()) {
      detail = first.getDetail().str();
      return mlir::failure();
    }
    return FixedTemporalChoice{*first.getChoice(), false};
  }

  auto construct =
      [&](TemporalTraversalKind kind) -> mlir::FailureOr<FixedTemporalChoice> {
    FixedTemporalChoice result;
    result.choice.kind = kind;
    for (const TemporalScopeDescriptor &descriptor :
         domain.getScopeDescriptors(kind)) {
      TemporalScopeChoice scope;
      scope.operation = descriptor.operation;
      scope.iteratorTileSizes = descriptor.iterationExtents;
      for (auto [dimension, capability] :
           llvm::enumerate(descriptor.iteratorCapabilities)) {
        if (capability != IteratorTilingCapability::Tileable)
          continue;
        int64_t &size = scope.iteratorTileSizes[dimension];
        for (uint32_t step = 0; step < refinement && size > 1; ++step)
          size = (size + 1) / 2;
        result.refined |= size < descriptor.iterationExtents[dimension];
      }
      mlir::FailureOr<llvm::SmallVector<uint32_t, 4>> order =
          buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                      scope.iteratorTileSizes,
                                      descriptor.precedence, &detail);
      if (mlir::failed(order))
        return mlir::failure();
      scope.loopOrder = std::move(*order);
      result.choice.scopes.push_back(std::move(scope));
    }
    if (!domain.contains(result.choice))
      return mlir::failure();
    return result;
  };

  mlir::FailureOr<FixedTemporalChoice> joint =
      construct(TemporalTraversalKind::Joint);
  if (mlir::succeeded(joint))
    return joint;
  mlir::FailureOr<FixedTemporalChoice> independent =
      construct(TemporalTraversalKind::Independent);
  if (mlir::failed(independent) && detail.empty())
    detail = "fixed temporal refinement is outside the current domain";
  return independent;
}

static ExecutableCompilationStatus
classifyLayoutFailure(ExactPBQPStatus status) {
  switch (status) {
  case ExactPBQPStatus::Optimal:
  case ExactPBQPStatus::Feasible:
    return ExecutableCompilationStatus::Accepted;
  case ExactPBQPStatus::NoSolution:
    return ExecutableCompilationStatus::UnsupportedFailure;
  case ExactPBQPStatus::Indeterminate:
    return ExecutableCompilationStatus::IndeterminateFailure;
  case ExactPBQPStatus::BrokenContract:
    return ExecutableCompilationStatus::CompilerFailure;
  }
  return ExecutableCompilationStatus::CompilerFailure;
}

static void attachCapacityRejectionDetail(ExecutableCompilationResult &result,
                                          uint32_t attempts) {
  auto capacity = llvm::find_if(
      result.tileFailures, [](const ExecutableTileFailure &failure) {
        return failure.memoryPlanning.spmCapacityOverflow;
      });
  if (capacity == result.tileFailures.end())
    return;
  std::string detail;
  llvm::raw_string_ostream stream(detail);
  stream << "actual SPM capacity rejected baseline after " << attempts
         << " attempt(s): tile=" << capacity->tileId.getValue()
         << ", demands=" << capacity->memoryPlanning.spmDemandCount
         << ", largest=" << capacity->memoryPlanning.spmLargestDemandBytes
         << " bytes";
  if (capacity->memoryPlanning.spmLargestDemandType)
    stream << ", type=" << capacity->memoryPlanning.spmLargestDemandType;
  if (!capacity->memoryPlanning.spmCapacityConflictDemands.empty()) {
    stream << ", conflict=[";
    llvm::interleaveComma(
        capacity->memoryPlanning.spmCapacityConflictDemands, stream,
        [&](const TileMemoryPlanningFailure::SPMDemandEvidence &demand) {
          stream << demand.bytes << " bytes " << demand.type << " users=[";
          llvm::interleaveComma(
              demand.userOperationNames, stream,
              [&](mlir::OperationName name) { stream << name.getStringRef(); });
          stream << "] outputs=[";
          llvm::interleaveComma(demand.outputIndices, stream);
          stream << "]";
        });
    stream << "]";
  }
  result.detail = std::move(detail);
}

} // namespace

ExecutableCompilationResult compileBaselineCurrentIR(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const BaselineCurrentIROptions &options,
    BaselineCurrentIRStatistics *statistics,
    ExecutableLoweringStatistics *executableStatistics) {
  if (!tensorProgram || options.maximumCapacityAttempts == 0 ||
      options.layoutWorkLimit == 0)
    return fail(ExecutableCompilationStatus::CompilerFailure, "baseline-input",
                "baseline requires current TensorProgram and positive work "
                "limits");
  mlir::FailureOr<mlir::func::FuncOp> function =
      getProgramFunction(tensorProgram);
  if (mlir::failed(function))
    return fail(ExecutableCompilationStatus::CompilerFailure, "baseline-input",
                "baseline requires exactly one defined TensorProgram");
  if (mlir::failed(closeStructuredProgramOutputs(*function)))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "baseline-output-closure",
                "baseline cannot close current TensorProgram outputs");
  if (mlir::failed(mlir::verify(tensorProgram)))
    return fail(ExecutableCompilationStatus::CompilerFailure, "baseline-input",
                "baseline requires verifier-valid normalized TensorProgram");

  auto structured = analyzeStructuredProgram(tensorProgram, program,
                                             executionConfig, diagnostics);
  if (mlir::failed(structured))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "structured-analysis",
                "cannot derive baseline structured program facts");
  std::string detail;
  auto spatial = buildCanonicalSpatialAssignment(
      (*structured)->dag, (*structured)->availableTileIds, &detail);
  if (mlir::failed(spatial))
    return fail(ExecutableCompilationStatus::UnsupportedFailure,
                "baseline-spatial", detail);

  auto demandSession =
      DemandPlanningSession::create((*structured)->dag, {}, &detail);
  if (mlir::failed(demandSession))
    return fail(ExecutableCompilationStatus::CompilerFailure, "baseline-demand",
                detail);
  analysis::ExactDemandOutcome demand =
      demandSession->query(spatial->assignment);
  const analysis::ExactDemandProof *proof =
      analysis::getExactDemandProof(demand);
  if (!proof) {
    demandSession->close();
    return fail(ExecutableCompilationStatus::UnsupportedFailure,
                "baseline-demand", demandFailureDetail(demand));
  }
  auto rootDomain =
      RootWorkDomain::create((*structured)->dag, spatial->assignment, *proof,
                             (*structured)->availableTileIds, &detail);
  demandSession->close();
  if (mlir::failed(rootDomain))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "baseline-root-work", detail);
  RootWorkCollectionOutcome rootWork = collectRootWorks(*rootDomain);
  const RootWorkCollection *rootWorks = getRootWorkCollection(rootWork);
  if (!rootWorks) {
    const bool indeterminate =
        std::holds_alternative<analysis::RootRegionWorkLimitReached>(rootWork);
    return fail(indeterminate
                    ? ExecutableCompilationStatus::IndeterminateFailure
                    : ExecutableCompilationStatus::CompilerFailure,
                "baseline-root-work", rootWorkFailureDetail(rootWork));
  }
  auto regionDomain = RegionDomain::create(rootWorks->works, &detail);
  if (mlir::failed(regionDomain))
    return fail(ExecutableCompilationStatus::CompilerFailure,
                "baseline-region-domain", detail);
  std::vector<RegionPlan> regionProposals = regionDomain->getProposals(1);
  if (regionProposals.empty())
    return fail(ExecutableCompilationStatus::CompilerFailure, "baseline-region",
                "baseline Region domain has no coherent fixed proposal");
  auto regionPlanIt =
      llvm::find_if(regionProposals, [](const RegionPlan &proposal) {
        return llvm::all_of(proposal.groups, [](const RegionGroupPlan &group) {
          return group.mandatoryRoots.size() == 1 && group.replicas.empty() &&
                 group.localBindings.empty();
        });
      });
  if (regionPlanIt == regionProposals.end())
    return fail(ExecutableCompilationStatus::CompilerFailure, "baseline-region",
                "baseline Region domain has no singleton fixed proposal");
  const RegionPlan &regionPlan = *regionPlanIt;

  std::optional<ExecutableCompilationResult> lastCapacityRejection;
  constexpr CardId cardId(0);
  for (uint32_t attempt = 0; attempt < options.maximumCapacityAttempts;
       ++attempt) {
    wafer::support::addCompileCounter("baseline", "attempts", 1);
    if (statistics)
      ++statistics->attempts;
    SpatialRegionMaterializationFailure spatialFailure;
    auto candidate = materializeSpatialRegions(
        tensorProgram, cardId, (*structured)->availableTileIds,
        (*structured)->operationNodes, rootWorks->works, regionPlan,
        &spatialFailure);
    if (mlir::failed(candidate))
      return fail(spatialFailure.kind ==
                          SpatialRegionMaterializationFailureKind::Unsupported
                      ? ExecutableCompilationStatus::UnsupportedFailure
                      : ExecutableCompilationStatus::CompilerFailure,
                  "baseline-spatial-materialization", spatialFailure.detail);
    if (statistics)
      ++statistics->spatialMaterializations;

    llvm::SmallVector<TileRegionOp, 32> regions;
    candidate->module->walk(
        [&](TileRegionOp region) { regions.push_back(region); });
    bool refined = false;
    for (TileRegionOp region : regions) {
      TemporalDomainResult domain = buildTemporalDomain(region);
      if (!domain.succeeded())
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "baseline-temporal-domain", domain.failure->detail);
      mlir::FailureOr<FixedTemporalChoice> choice =
          buildFixedTemporalChoice(*domain.domain, attempt, detail);
      if (mlir::failed(choice))
        return fail(ExecutableCompilationStatus::UnsupportedFailure,
                    "baseline-temporal-choice", detail);
      refined |= choice->refined;
      TemporalTilingFailure temporalFailure;
      auto tiled = applyTemporalTiling(*domain.domain, choice->choice,
                                       candidate->relations, &temporalFailure);
      if (mlir::failed(tiled))
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "baseline-temporal-apply", temporalFailure.detail);
      if (statistics)
        ++statistics->temporalApplications;
    }
    if (attempt != 0 && !refined && lastCapacityRejection) {
      attachCapacityRejectionDetail(*lastCapacityRejection, attempt);
      return std::move(*lastCapacityRejection);
    }

    if (options.qualification && options.qualification->mergeRegions) {
      SpatialRegionMaterializationFailure failure;
      CommunicationRegionClosureStatistics closure;
      if (mlir::failed(closeCrossTileCommunicationRegions(
              *candidate->module, candidate->relations, &closure, &failure)))
        return fail(ExecutableCompilationStatus::CompilerFailure,
                    "communication-region-closure", failure.detail);
      if (statistics)
        statistics->communicationRegionClosures +=
            closure.closedExchangeComponents;
    }

    OnlineAttentionDecompositionFailure attentionFailure;
    if (mlir::failed(decomposeOnlineAttention(
            *candidate->module, candidate->relations, &attentionFailure)))
      return fail(
          attentionFailure.kind ==
                  OnlineAttentionDecompositionFailureKind::UnsupportedSemantics
              ? ExecutableCompilationStatus::UnsupportedFailure
              : ExecutableCompilationStatus::CompilerFailure,
          "baseline-attention-decomposition", attentionFailure.detail);

    LayoutOptimizationResult layout = resolveCurrentLayoutsAndBufferize(
        *candidate->module, candidate->relations, options.layoutWorkLimit);
    recordLayoutInstrumentation(layout.statistics);
    if (statistics) {
      ++statistics->layoutInvocations;
      statistics->layoutFeasibleFallbacks +=
          layout.status == ExactPBQPStatus::Feasible;
    }
    if (!layout.succeeded())
      return fail(classifyLayoutFailure(layout.status), "baseline-layout",
                  layout.detail);
    StructuredToTileResult compute =
        lowerStructuredComputeToTile(*candidate->module, candidate->relations);
    if (!compute.succeeded())
      return fail(compute.failure == StructuredToTileFailureKind::Unsupported
                      ? ExecutableCompilationStatus::UnsupportedFailure
                      : ExecutableCompilationStatus::CompilerFailure,
                  "baseline-structured-to-tile", compute.detail);
    BoundaryMovementResult movement = materializeTileBoundaryMovement(
        *candidate->module, candidate->relations,
        options.qualification ? options.qualification->movement
                              : BoundaryMovementOptions{});
    recordMovementInstrumentation(movement.statistics);
    if (!movement.succeeded())
      return fail(movement.failure == BoundaryMovementFailureKind::Unsupported
                      ? ExecutableCompilationStatus::UnsupportedFailure
                      : ExecutableCompilationStatus::CompilerFailure,
                  "baseline-boundary-movement", movement.detail);
    if (statistics)
      statistics->boundaryMovement = movement.statistics;

    CurrentIRDownstreamStatistics downstream;
    ExecutableCompilationResult result = compileCurrentIRCandidateToExecutable(
        std::move(candidate->module), std::move(candidate->relations), cardId,
        (*structured)->availableTileIds, program, executionConfig, diagnostics,
        programData, options.downstream, &downstream, executableStatistics);
    if (statistics)
      statistics->downstream = downstream;
    if (!result.isProvenExactRejection())
      return result;
    lastCapacityRejection = std::move(result);
    wafer::support::addCompileCounter("baseline", "capacity-refinements", 1);
    if (statistics)
      ++statistics->capacityRefinements;
  }
  if (lastCapacityRejection) {
    attachCapacityRejectionDetail(*lastCapacityRejection,
                                  options.maximumCapacityAttempts);
    return std::move(*lastCapacityRejection);
  }
  return fail(ExecutableCompilationStatus::CompilerFailure,
              "baseline-controller",
              "baseline exhausted attempts without a typed result");
}

} // namespace wafer::compiler::detail
