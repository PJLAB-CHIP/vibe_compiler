//===- CandidateEvaluation.cpp - Tile candidate implementation
//-----------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace wafer::tensor_program_scheduling {

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
                          const SelectionConfig &config,
                          const TileRegionToInstrOptions &options) {
  mlir::MLIRContext *context = evaluation.module->getContext();
  std::string failureReason;
  mlir::LogicalResult result = mlir::success();
  failureReason.clear();
  std::string diagnostics = takeDiagnostics(
      context,
      [&]() {
        return tile_region_to_instr::convertTileRegionToInstrModule(
            *evaluation.module, options, &failureReason);
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

  evaluation.stats =
      estimateStats(*evaluation.module, config.scheduleCostPolicy);
  return evaluation;
}

static TileRegionToInstrOptions
getCommunicationOptions(CommunicationAlternative alternative) {
  switch (alternative) {
  case CommunicationAlternative::Ring:
    return {};
  case CommunicationAlternative::DirectAllGather:
    return {/*allGatherSchedule=*/AllGatherSchedule::Direct,
            /*allReduceSchedule=*/AllReduceSchedule::Ring,
            /*reduceScatterSchedule=*/ReduceScatterSchedule::Direct};
  case CommunicationAlternative::RingReduceScatter:
    return {/*allGatherSchedule=*/AllGatherSchedule::Ring,
            /*allReduceSchedule=*/AllReduceSchedule::Auto,
            /*reduceScatterSchedule=*/ReduceScatterSchedule::Ring};
  case CommunicationAlternative::TreeAllReduce:
    return {/*allGatherSchedule=*/AllGatherSchedule::Ring,
            /*allReduceSchedule=*/AllReduceSchedule::Tree,
            /*reduceScatterSchedule=*/ReduceScatterSchedule::Direct};
  case CommunicationAlternative::DirectAllGatherTreeAllReduce:
    return {/*allGatherSchedule=*/AllGatherSchedule::Direct,
            /*allReduceSchedule=*/AllReduceSchedule::Tree,
            /*reduceScatterSchedule=*/ReduceScatterSchedule::Direct};
  }
  llvm_unreachable("unknown communication alternative");
}

static bool
canUseFullTraversalFallback(const CandidateSpec &candidate,
                            llvm::ArrayRef<int64_t> traversalShape) {
  return candidate.traversalKind ==
             CandidateTileTraversalKind::ResultDriven &&
         candidate.reductionSplitSizes.empty() &&
         candidate.tileSizes.size() == traversalShape.size() &&
         std::equal(candidate.tileSizes.begin(), candidate.tileSizes.end(),
                    traversalShape.begin(), traversalShape.end());
}

CandidateEvaluation evaluateCompleteCandidate(
    mlir::func::FuncOp task, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config) {
  CandidateEvaluation evaluation;
  std::string failureReason;
  mlir::LogicalResult result = mlir::success();
  const bool canUseFullTraversal =
      canUseFullTraversalFallback(candidate, traversalShape);
  // A full-shape direct-boundary route must be materialized from the original
  // task boundary.  Building a one-trip complete traversal first introduces
  // extract/insert slices whose identity is only visible after lowering, so
  // it would silently collapse this distinct route back to Tensor staging.
  const bool preferFullTraversal =
      config.useDirectMappedBoundaryTransfer && canUseFullTraversal;
  std::string diagnostics;
  if (preferFullTraversal) {
    diagnostics = takeDiagnostics(
        task.getContext(),
        [&]() {
          return lowerTensorProgramToTileRegionModule(
              task, evaluation.module, &failureReason, config.logicalRank,
              candidate.selectedImplementationAlternative,
              config.useDirectMappedBoundaryTransfer);
        },
        result);
    evaluation.artifactSource = CandidateArtifactSource::FullTraversalFallback;
  } else {
    diagnostics = takeDiagnostics(
        task.getContext(),
        [&]() {
          return lowerCompleteCandidateTensorProgramToTileRegionModule(
              task, candidate.tileSizes, candidate.reductionSplitSizes,
              evaluation.module, &failureReason, config.logicalRank,
              candidate.selectedImplementationAlternative,
              config.useDirectMappedBoundaryTransfer,
              candidate.traversalKind);
        },
        result);
    evaluation.artifactSource = CandidateArtifactSource::CompleteTraversalAPI;
  }
  if (mlir::failed(result) && canUseFullTraversal && !preferFullTraversal) {
    failureReason.clear();
    diagnostics = takeDiagnostics(
        task.getContext(),
        [&]() {
          return lowerTensorProgramToTileRegionModule(
              task, evaluation.module, &failureReason, config.logicalRank,
              candidate.selectedImplementationAlternative,
              config.useDirectMappedBoundaryTransfer);
        },
        result);
    if (mlir::succeeded(result))
      evaluation.artifactSource =
          CandidateArtifactSource::FullTraversalFallback;
  } else if (mlir::failed(result) && preferFullTraversal) {
    failureReason.clear();
    diagnostics = takeDiagnostics(
        task.getContext(),
        [&]() {
          return lowerCompleteCandidateTensorProgramToTileRegionModule(
              task, candidate.tileSizes, candidate.reductionSplitSizes,
              evaluation.module, &failureReason, config.logicalRank,
              candidate.selectedImplementationAlternative,
              config.useDirectMappedBoundaryTransfer,
              candidate.traversalKind);
        },
        result);
    if (mlir::succeeded(result))
      evaluation.artifactSource = CandidateArtifactSource::CompleteTraversalAPI;
  }
  if (mlir::failed(result)) {
    evaluation.failureReason =
        joinFailure("complete-tile-region", failureReason, diagnostics);
    return evaluation;
  }
  return finishCandidateEvaluation(
      std::move(evaluation), config,
      getCommunicationOptions(config.communicationAlternative));
}

static void
registerSelectionEvaluationDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
}

static mlir::OwningOpRef<mlir::ModuleOp>
parseStandaloneTaskModule(llvm::StringRef standaloneTaskModuleText,
                          mlir::MLIRContext &context,
                          std::string &failureReason) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::LogicalResult parseResult = mlir::success();
  std::string diagnostics = takeDiagnostics(
      &context,
      [&]() {
        module = mlir::parseSourceString<mlir::ModuleOp>(
            standaloneTaskModuleText, &context);
        return module ? mlir::success() : mlir::failure();
      },
      parseResult);
  if (mlir::failed(parseResult) || !module) {
    failureReason = joinFailure("parse-standalone", "", diagnostics);
    return nullptr;
  }
  return module;
}

enum class AcceptedModuleTransfer {
  RetainInCurrentContext,
  SerializeForOwnerImport,
};

static CandidateCheckResult evaluateTaskCandidate(
    mlir::func::FuncOp task, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config,
    AcceptedModuleTransfer transfer) {
  CandidateCheckResult result;
  result.spec = candidate;
  llvm::SmallVector<TileInstance, 8> reps =
      buildRepresentativeTiles(traversalShape, candidate.tileSizes);
  result.representativeCount = static_cast<int64_t>(reps.size());

  // The compact complete traversal contains the main tile, every static tail
  // and every tail corner in the same artifact that can be committed.  Running
  // each representative as an independent function is neither an additional
  // coverage proof nor a sound resource bound: a partial function must
  // reconstruct its full output destination and can therefore allocate a
  // full-shape SPM buffer that does not exist in the complete traversal.
  // Keep the representative count for deterministic diagnostics, but run all
  // instruction, SPM, DDR and cost gates exactly once on the all-and-only
  // complete artifact.
  CandidateEvaluation acceptedEvaluation =
      evaluateCompleteCandidate(task, traversalShape, candidate, config);
  if (!acceptedEvaluation.failureReason.empty()) {
    result.failureReason = acceptedEvaluation.failureReason;
    return result;
  }
  if (!isCompleteArtifactSource(acceptedEvaluation.artifactSource)) {
    result.failureReason =
        "complete-artifact: accepted candidate has representative-only "
        "provenance";
    return result;
  }
  result.stats = acceptedEvaluation.stats;
  result.artifactSource = acceptedEvaluation.artifactSource;
  if (transfer == AcceptedModuleTransfer::RetainInCurrentContext) {
    result.module = std::move(acceptedEvaluation.module);
  } else {
    llvm::raw_string_ostream os(result.acceptedModuleText);
    acceptedEvaluation.module->print(os);
  }
  return result;
}

class CandidateEvaluationExecutor::Impl {
public:
  explicit Impl(unsigned workerCount)
      : queueCapacity(std::clamp(workerCount, 1u, kMaximumWorkerCount)) {}

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    workAvailable.notify_all();
    queueSpaceAvailable.notify_all();
    for (std::thread &worker : workers)
      worker.join();
  }

  std::future<CandidateCheckResult>
  submit(std::shared_ptr<const std::string> standaloneTaskModuleText,
         llvm::ArrayRef<int64_t> traversalShape, const CandidateSpec &candidate,
         const SelectionConfig &config) {
    auto item = std::make_unique<WorkItem>(std::move(standaloneTaskModuleText),
                                           traversalShape, candidate, config);
    std::future<CandidateCheckResult> future = item->promise.get_future();
    {
      std::unique_lock<std::mutex> lock(mutex);
      if (!stopping)
        startWorkersLocked();
      queueSpaceAvailable.wait(
          lock, [&]() { return stopping || queue.size() < queueCapacity; });
      if (stopping) {
        CandidateCheckResult result;
        result.spec = candidate;
        result.failureReason = "candidate-executor: executor is stopping";
        item->promise.set_value(std::move(result));
        return future;
      }
      queue.push_back(std::move(item));
    }
    workAvailable.notify_one();
    return future;
  }

  unsigned getWorkerCount() const {
    return static_cast<unsigned>(queueCapacity);
  }

  unsigned getWorkerConstructionCount() const {
    std::lock_guard<std::mutex> lock(mutex);
    return static_cast<unsigned>(workers.size());
  }

  unsigned getContextConstructionCount() const {
    return contextConstructionCount.load(std::memory_order_relaxed);
  }

  unsigned getTaskParseCount() const {
    return taskParseCount.load(std::memory_order_relaxed);
  }

private:
  static constexpr unsigned kMaximumWorkerCount = 64;

  struct WorkItem {
    WorkItem(std::shared_ptr<const std::string> standaloneTaskModuleText,
             llvm::ArrayRef<int64_t> traversalShape,
             const CandidateSpec &candidate, const SelectionConfig &config)
        : standaloneTaskModuleText(std::move(standaloneTaskModuleText)),
          traversalShape(traversalShape.begin(), traversalShape.end()),
          candidate(candidate), config(config) {
      this->config.evaluationExecutor = nullptr;
    }

    std::shared_ptr<const std::string> standaloneTaskModuleText;
    llvm::SmallVector<int64_t, 4> traversalShape;
    CandidateSpec candidate;
    SelectionConfig config;
    std::promise<CandidateCheckResult> promise;
  };

  void startWorkersLocked() {
    if (!workers.empty())
      return;
    workers.reserve(queueCapacity);
    for (unsigned workerIndex = 0; workerIndex < queueCapacity; ++workerIndex)
      workers.emplace_back([this]() { workerLoop(); });
  }

  void workerLoop() {
    std::unique_ptr<mlir::MLIRContext> context;
    std::string parseFailure;
    std::string parsedTaskModuleText;
    bool hasParsedTask = false;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp task;

    while (true) {
      std::unique_ptr<WorkItem> item;
      {
        std::unique_lock<std::mutex> lock(mutex);
        workAvailable.wait(lock, [&]() { return stopping || !queue.empty(); });
        if (queue.empty()) {
          if (stopping)
            return;
          continue;
        }
        item = std::move(queue.front());
        queue.pop_front();
      }
      queueSpaceAvailable.notify_one();

      if (!context) {
        mlir::DialectRegistry registry;
        registerSelectionEvaluationDialects(registry);
        context = std::make_unique<mlir::MLIRContext>(registry);
        context->disableMultithreading();
        context->loadAllAvailableDialects();
        contextConstructionCount.fetch_add(1, std::memory_order_relaxed);
      }

      if (!hasParsedTask ||
          parsedTaskModuleText != *item->standaloneTaskModuleText) {
        parsedTaskModuleText = *item->standaloneTaskModuleText;
        hasParsedTask = true;
        parseFailure.clear();
        task = {};
        module = parseStandaloneTaskModule(parsedTaskModuleText, *context,
                                           parseFailure);
        if (module)
          task = findSingleSelectionTask(*module);
        if (module && !task)
          parseFailure =
              "parse-standalone: standalone module has no scheduling task";
        taskParseCount.fetch_add(1, std::memory_order_relaxed);
      }

      CandidateCheckResult result;
      if (!parseFailure.empty()) {
        result.spec = item->candidate;
        result.failureReason = parseFailure;
      } else {
        result = evaluateTaskCandidate(
            task, item->traversalShape, item->candidate, item->config,
            AcceptedModuleTransfer::SerializeForOwnerImport);
      }
      item->promise.set_value(std::move(result));
    }
  }

  size_t queueCapacity;
  std::atomic<unsigned> contextConstructionCount{0};
  std::atomic<unsigned> taskParseCount{0};
  mutable std::mutex mutex;
  std::condition_variable workAvailable;
  std::condition_variable queueSpaceAvailable;
  std::deque<std::unique_ptr<WorkItem>> queue;
  bool stopping = false;
  std::vector<std::thread> workers;
};

CandidateEvaluationExecutor::CandidateEvaluationExecutor(unsigned workerCount)
    : impl(std::make_unique<Impl>(workerCount)) {}

CandidateEvaluationExecutor::~CandidateEvaluationExecutor() = default;

std::future<CandidateCheckResult> CandidateEvaluationExecutor::submit(
    std::shared_ptr<const std::string> standaloneTaskModuleText,
    llvm::ArrayRef<int64_t> traversalShape, const CandidateSpec &candidate,
    const SelectionConfig &config) {
  return impl->submit(std::move(standaloneTaskModuleText), traversalShape,
                      candidate, config);
}

unsigned CandidateEvaluationExecutor::getWorkerCount() const {
  return impl->getWorkerCount();
}

unsigned CandidateEvaluationExecutor::getWorkerConstructionCount() const {
  return impl->getWorkerConstructionCount();
}

unsigned CandidateEvaluationExecutor::getContextConstructionCount() const {
  return impl->getContextConstructionCount();
}

unsigned CandidateEvaluationExecutor::getTaskParseCount() const {
  return impl->getTaskParseCount();
}

CandidateCheckResult evaluateCandidateOnOriginalTask(
    mlir::func::FuncOp task, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config) {
  return evaluateTaskCandidate(task, traversalShape, candidate, config,
                               AcceptedModuleTransfer::RetainInCurrentContext);
}

CandidateCheckResult
evaluateCandidateOnStandaloneTaskText(llvm::StringRef standaloneTaskModuleText,
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
  mlir::OwningOpRef<mlir::ModuleOp> module = parseStandaloneTaskModule(
      standaloneTaskModuleText, context, parseFailure);
  if (!module) {
    result.failureReason = parseFailure;
    return result;
  }
  mlir::func::FuncOp task = findSingleSelectionTask(*module);
  if (!task) {
    result.failureReason =
        "parse-standalone: standalone module has no scheduling task";
    return result;
  }
  return evaluateTaskCandidate(task, traversalShape, candidate, config,
                               AcceptedModuleTransfer::SerializeForOwnerImport);
}

mlir::LogicalResult
importAcceptedCandidateModule(CandidateCheckResult &result,
                              mlir::MLIRContext &ownerContext,
                              const analysis::TargetScheduleCostPolicy
                                  &scheduleCostPolicy) {
  if (!result.failureReason.empty())
    return mlir::failure();
  if (result.acceptedModuleText.empty()) {
    result.failureReason =
        "owner-import: accepted worker candidate has no module transport";
    return mlir::failure();
  }

  mlir::OwningOpRef<mlir::ModuleOp> imported;
  mlir::LogicalResult parseResult = mlir::success();
  std::string diagnostics = takeDiagnostics(
      &ownerContext,
      [&]() {
        imported = mlir::parseSourceString<mlir::ModuleOp>(
            result.acceptedModuleText, &ownerContext);
        return imported ? mlir::success() : mlir::failure();
      },
      parseResult);
  if (mlir::failed(parseResult) || !imported) {
    result.failureReason = joinFailure("owner-import", "", diagnostics);
    return mlir::failure();
  }

  mlir::LogicalResult verifyResult = mlir::success();
  diagnostics = takeDiagnostics(
      &ownerContext, [&]() { return mlir::verify(*imported); }, verifyResult);
  if (mlir::failed(verifyResult)) {
    result.failureReason =
        joinInstructionFailure("owner-verifier", "", diagnostics);
    return mlir::failure();
  }

  result.stats = estimateStats(*imported, scheduleCostPolicy);
  result.module = std::move(imported);
  result.acceptedModuleText.clear();
  return mlir::success();
}

} // namespace wafer::tensor_program_scheduling
