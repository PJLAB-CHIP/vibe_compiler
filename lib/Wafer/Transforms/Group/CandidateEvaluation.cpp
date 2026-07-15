//===- CandidateEvaluation.cpp - Tile candidate implementation
//-----------------===//

#include "Group/SelectGroupTileInternal.h"
#include "Wafer/Transforms/Passes.h"

namespace wafer::group_tile_selection {

static std::string
takeDiagnostics(mlir::MLIRContext *context,
                llvm::function_ref<mlir::LogicalResult()> callback,
                mlir::LogicalResult &result) {
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  result = callback();
  return diagnostics;
}

static std::string joinFailure(llvm::StringRef gate, llvm::StringRef reason,
                               llvm::StringRef diagnostics) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << gate << ": ";
  if (!reason.empty())
    os << reason;
  else if (!diagnostics.empty())
    os << diagnostics.trim();
  else
    os << "failed";
  return os.str();
}

static std::string joinInstructionFailure(llvm::StringRef gate,
                                          llvm::StringRef reason,
                                          llvm::StringRef diagnostics) {
  llvm::StringRef detail = reason.empty() ? diagnostics.trim() : reason.trim();
  constexpr llvm::StringLiteral targetNarrowingKey = "target_abi_narrowing:";
  size_t keyOffset = detail.find(targetNarrowingKey);
  if (keyOffset != llvm::StringRef::npos)
    return detail.drop_front(keyOffset).split('\n').first.str();
  return joinFailure(gate, reason, diagnostics);
}

static CandidateEvaluation
finishCandidateEvaluation(CandidateEvaluation evaluation,
                          const SelectionConfig &config) {
  mlir::MLIRContext *context = evaluation.module->getContext();
  std::string failureReason;
  mlir::LogicalResult result = mlir::success();
  failureReason.clear();
  std::string diagnostics = takeDiagnostics(
      context,
      [&]() {
        return convertTileRegionToInstrModule(*evaluation.module,
                                              &failureReason);
      },
      result);
  if (mlir::failed(result)) {
    evaluation.failureReason =
        joinInstructionFailure("instr-lowering", failureReason, diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context,
      [&]() {
        return planSPMMemoryModule(*evaluation.module, config.spmBase,
                                   config.spmLimit, config.spmAlignment);
      },
      result);
  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("spm-offsets", "", diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context,
      [&]() {
        return planDDRMemoryModule(*evaluation.module, config.ddrAlignmentBytes,
                                   config.ddrCapacityBytes,
                                   config.ddrLargestContiguousBytes,
                                   config.ddrBandwidthLimitBytes);
      },
      result);
  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("ddr-offsets", "", diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context, [&]() { return mlir::verify(*evaluation.module); }, result);
  if (mlir::failed(result)) {
    evaluation.failureReason =
        joinInstructionFailure("verifier", "", diagnostics);
    return evaluation;
  }

  evaluation.stats = estimateStats(*evaluation.module);
  return evaluation;
}

static CandidateEvaluation
evaluateTileInstance(GroupOp group, llvm::ArrayRef<int64_t> traversalShape,
                     const CandidateSpec &candidate, const TileInstance &tile,
                     const SelectionConfig &config) {
  CandidateEvaluation evaluation;
  mlir::MLIRContext *context = group.getContext();

  std::string failureReason;
  mlir::LogicalResult result = mlir::success();
  std::string diagnostics = takeDiagnostics(
      context,
      [&]() {
        return lowerCandidateGroupToTileRegionModule(
            group, tile.offsets, tile.sizes, candidate.reductionSplitSizes,
            evaluation.module, &failureReason, config.logicalRank);
      },
      result);

  bool usedFullGroupFallback = false;
  if (mlir::failed(result) && candidate.reductionSplitSizes.empty() &&
      isFullFirstTile(traversalShape, tile)) {
    failureReason.clear();
    diagnostics = takeDiagnostics(
        context,
        [&]() {
          return lowerGroupToTileRegionModule(
              group, evaluation.module, &failureReason, config.logicalRank);
        },
        result);
    usedFullGroupFallback = mlir::succeeded(result);
  }

  if (mlir::failed(result)) {
    evaluation.failureReason =
        joinFailure("tile-region", failureReason, diagnostics);
    return evaluation;
  }

  if (isFullFirstTile(traversalShape, tile)) {
    evaluation.artifactSource =
        usedFullGroupFallback ? CandidateArtifactSource::FullGroupFallback
                              : CandidateArtifactSource::CompleteTileInstance;
  }
  return finishCandidateEvaluation(std::move(evaluation), config);
}

static bool canUseFullGroupFallback(const CandidateSpec &candidate,
                                    llvm::ArrayRef<int64_t> traversalShape) {
  return candidate.reductionSplitSizes.empty() &&
         candidate.tileSizes.size() == traversalShape.size() &&
         std::equal(candidate.tileSizes.begin(), candidate.tileSizes.end(),
                    traversalShape.begin(), traversalShape.end());
}

CandidateEvaluation
evaluateCompleteCandidate(GroupOp group, llvm::ArrayRef<int64_t> traversalShape,
                          const CandidateSpec &candidate,
                          const SelectionConfig &config) {
  CandidateEvaluation evaluation;
  mlir::MLIRContext *context = group.getContext();

  std::string failureReason;
  mlir::LogicalResult result = mlir::success();
  std::string diagnostics = takeDiagnostics(
      context,
      [&]() {
        return lowerCompleteCandidateGroupToTileRegionModule(
            group, candidate.tileSizes, candidate.reductionSplitSizes,
            evaluation.module, &failureReason, config.logicalRank);
      },
      result);
  evaluation.artifactSource = CandidateArtifactSource::CompleteTraversalAPI;

  if (mlir::failed(result) &&
      canUseFullGroupFallback(candidate, traversalShape)) {
    failureReason.clear();
    diagnostics = takeDiagnostics(
        context,
        [&]() {
          return lowerGroupToTileRegionModule(
              group, evaluation.module, &failureReason, config.logicalRank);
        },
        result);
    if (mlir::succeeded(result))
      evaluation.artifactSource = CandidateArtifactSource::FullGroupFallback;
  }

  if (mlir::failed(result)) {
    evaluation.failureReason =
        joinFailure("complete-tile-region", failureReason, diagnostics);
    return evaluation;
  }

  return finishCandidateEvaluation(std::move(evaluation), config);
}

static void
registerSelectionEvaluationDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
}

static mlir::OwningOpRef<mlir::ModuleOp>
parseStandaloneGroupModule(llvm::StringRef standaloneGroupModuleText,
                           mlir::MLIRContext &context,
                           std::string &failureReason) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::LogicalResult parseResult = mlir::success();
  std::string diagnostics = takeDiagnostics(
      &context,
      [&]() {
        module = mlir::parseSourceString<mlir::ModuleOp>(
            standaloneGroupModuleText, &context);
        return module ? mlir::success() : mlir::failure();
      },
      parseResult);
  if (mlir::failed(parseResult) || !module) {
    failureReason = joinFailure("parse-standalone", "", diagnostics);
    return nullptr;
  }
  return module;
}

static CandidateCheckResult
evaluateCandidate(GroupOp group, llvm::ArrayRef<int64_t> traversalShape,
                  const CandidateSpec &candidate, const SelectionConfig &config,
                  bool retainAcceptedModule) {
  CandidateCheckResult result;
  result.spec = candidate;
  llvm::SmallVector<TileInstance, 8> reps =
      buildRepresentativeTiles(traversalShape, candidate.tileSizes);
  result.representativeCount = static_cast<int64_t>(reps.size());

  CandidateEvaluation acceptedEvaluation;
  for (auto [repIndex, rep] : llvm::enumerate(reps)) {
    CandidateEvaluation evaluation =
        evaluateTileInstance(group, traversalShape, candidate, rep, config);
    if (!evaluation.failureReason.empty()) {
      result.failureReason = evaluation.failureReason;
      return result;
    }
    if (repIndex == 0)
      result.stats = evaluation.stats;
    if (isCompleteArtifactSource(evaluation.artifactSource))
      acceptedEvaluation = std::move(evaluation);
  }

  if (!acceptedEvaluation.module) {
    acceptedEvaluation =
        evaluateCompleteCandidate(group, traversalShape, candidate, config);
    if (!acceptedEvaluation.failureReason.empty()) {
      result.failureReason = acceptedEvaluation.failureReason;
      return result;
    }
  }

  if (!isCompleteArtifactSource(acceptedEvaluation.artifactSource)) {
    result.failureReason =
        "complete-artifact: accepted candidate has representative-only "
        "provenance";
    return result;
  }
  result.artifactSource = acceptedEvaluation.artifactSource;
  if (retainAcceptedModule)
    result.module = std::move(acceptedEvaluation.module);
  return result;
}

CandidateCheckResult evaluateCandidateOnOriginalGroup(
    GroupOp group, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config) {
  return evaluateCandidate(group, traversalShape, candidate, config,
                           /*retainAcceptedModule=*/true);
}

CandidateCheckResult
evaluateCandidateOnStandaloneText(llvm::StringRef standaloneGroupModuleText,
                                  llvm::ArrayRef<int64_t> traversalShape,
                                  const CandidateSpec &candidate,
                                  const SelectionConfig &config) {
  CandidateCheckResult result;
  result.spec = candidate;

  mlir::DialectRegistry registry;
  registerSelectionEvaluationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string parseFailure;
  mlir::OwningOpRef<mlir::ModuleOp> module = parseStandaloneGroupModule(
      standaloneGroupModuleText, context, parseFailure);
  if (!module) {
    result.failureReason = parseFailure;
    return result;
  }

  GroupOp parsedGroup = findSingleSelectionGroup(*module);
  if (!parsedGroup) {
    result.failureReason =
        "parse-standalone: standalone module has no wafer.group";
    return result;
  }

  return evaluateCandidate(parsedGroup, traversalShape, candidate, config,
                           /*retainAcceptedModule=*/false);
}

} // namespace wafer::group_tile_selection
