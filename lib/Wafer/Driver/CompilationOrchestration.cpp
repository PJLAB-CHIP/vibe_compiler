//===- CompilationOrchestration.cpp - Compiler transaction orchestration ===//

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CompilationStatistics.h"
#include "Wafer/Frontend/StableHLO/ProgramIngestion.h"
#include "Wafer/Package/Writer/PackageInternal.h"

#include "Wafer/Conversion/StableHLOToLinalg/Pipelines.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Support/PassPipeline.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace wafer::compiler::detail {

static void printOptimizationConfig(OptimizationConfig config,
                                    llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-compile: optimization-policy="
              << (config.isSearch() ? "search" : "none");
  if (std::optional<SearchLimits> limits = config.getSearchLimits())
    diagnostics << " search-width=" << limits->width
                << " search-trials=" << limits->trials;
  diagnostics << '\n';
}

mlir::LogicalResult runCompilationTransaction(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    CompilationOptions options, std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<DeviceExecutable> *retainedDeviceExecutable,
    std::optional<TargetLLVMModules> *retainedTargetLLVMModules,
    std::optional<CompilationIRTrace> *retainedIRTrace,
    std::optional<ExecutablePackage> *retainedPackage,
    std::optional<ProfileInstrumentationProduct> *retainedProfileProduct,
    CompilationStage *failureStage,
    CommitFailureInjection commitFailureInjection,
    const CommunicationCandidateSelection *qualification) {
#if !defined(WAFER_ENABLE_STABLEHLO) || !defined(WAFER_ENABLE_SHARDY)
  (void)request;
  (void)outputDirectory;
  (void)xlaSpmdPartitionerHelper;
  (void)targetToolchain;
  (void)options;
  (void)failAfterLaunchSlot;
  (void)failAfterTargetLaunchSlot;
  (void)failAfterPackageLaunchSlot;
  (void)retainedDeviceExecutable;
  (void)retainedTargetLLVMModules;
  (void)retainedIRTrace;
  (void)retainedPackage;
  (void)retainedProfileProduct;
  (void)commitFailureInjection;
  (void)qualification;
  if (failureStage)
    *failureStage = CompilationStage::SourceVerification;
  reject(diagnostics,
         "StableHLO and SPMD partitioner dependencies are required");
  return mlir::failure();
#else
  CompilationStageTracker stages;
  // The stage is written on every exit; callers consume it only when the
  // transaction failed.
  auto failureStageReport = llvm::make_scope_exit([&] {
    if (failureStage)
      *failureStage = stages.current;
  });
  const CompileClock::time_point transactionStart = CompileClock::now();
  std::shared_ptr<wafer::support::CompileTimingSession> timingSession;
  if (options.shouldReportDetailedTiming())
    timingSession =
        std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
  wafer::support::ScopedCompileTimingActivation timingActivation(timingSession);
  auto timingReport = llvm::make_scope_exit([&] {
    if (timingSession)
      timingSession->finishAndPrintSummary();
  });
  std::shared_ptr<wafer::support::CompileWorkStatisticsSession>
      compileWorkSession;
  if (options.shouldReportDetailedTiming())
    compileWorkSession =
        std::make_shared<wafer::support::CompileWorkStatisticsSession>();
  wafer::support::ScopedCompileWorkStatisticsActivation compileWorkActivation(
      compileWorkSession);
  auto compileWorkReport = llvm::make_scope_exit([&] {
    if (!compileWorkSession)
      return;
    wafer::support::CompileWorkStatistics work = compileWorkSession->snapshot();
    diagnostics << "wafer-compile: compile-work"
                << " tile_memory_planning_invocations="
                << work.tileMemoryPlanningInvocations
                << " tile_to_instr_lowerings="
                << work.tileToInstructionLowerings
                << " relation_descriptor_plannings="
                << work.relationDescriptorPlannings
                << " spm_planning_invocations=" << work.spmPlanningInvocations
                << " ddr_planning_invocations=" << work.ddrPlanningInvocations
                << " target_abi_module_clones=" << work.targetABIModuleClones
                << " target_lowering_invocations="
                << work.targetLoweringInvocations
                << " target_translation_invocations="
                << work.targetTranslationInvocations << "\n";
  });
  wafer::support::ScopedCompileTimingSpan transactionTiming(
      "stage", "source-to-package", "compile-transaction");
  auto sourceTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "stage", "source-to-package", "source-to-tensor-program");
  if (outputDirectory.empty()) {
    reject(diagnostics, "output directory must not be empty");
    return mlir::failure();
  }
  if (xlaSpmdPartitionerHelper.empty()) {
    reject(diagnostics, "XLA SPMD partitioner helper path must not be empty");
    return mlir::failure();
  }
  if (options.shouldProduceProfileInstrumentation() &&
      request.getExecutionConfig().getTileCount() !=
          ExecutionConfig::kSingleCardTileCount) {
    reject(diagnostics, "profile compilation requires all Tiles on the card");
    return mlir::failure();
  }
  printOptimizationConfig(options.getOptimizationConfig(), diagnostics);

  llvm::SmallString<256> canonicalSource;
  if (std::error_code error = llvm::sys::fs::real_path(
          request.getSourceProgramDirectory(), canonicalSource)) {
    reject(diagnostics, "failed to resolve source program directory '" +
                            request.getSourceProgramDirectory().str() +
                            "': " + error.message());
    return mlir::failure();
  }
  if (!isDirectory(canonicalSource)) {
    reject(diagnostics, "source program directory is not a directory: '" +
                            canonicalSource.str().str() + "'");
    return mlir::failure();
  }

  llvm::SmallString<256> absoluteOutput;
  if (makeAbsoluteNormalizedPath(outputDirectory, absoluteOutput, diagnostics))
    return mlir::failure();
  llvm::StringRef outputName = llvm::sys::path::filename(absoluteOutput);
  if (outputName.empty()) {
    reject(diagnostics, "output directory must name a directory");
    return mlir::failure();
  }
  if (pathEntryExists(absoluteOutput)) {
    reject(diagnostics, "refusing to replace existing output directory: '" +
                            absoluteOutput.str().str() + "'");
    return mlir::failure();
  }

  llvm::SmallString<256> outputParent(absoluteOutput);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";

  llvm::SmallString<256> prospectiveCanonicalOutputParent;
  if (resolveThroughExistingAncestor(
          outputParent, prospectiveCanonicalOutputParent, diagnostics))
    return mlir::failure();
  llvm::SmallString<256> prospectiveCanonicalOutput(
      prospectiveCanonicalOutputParent);
  llvm::sys::path::append(prospectiveCanonicalOutput, outputName);
  if (pathIsWithin(prospectiveCanonicalOutput, canonicalSource)) {
    reject(diagnostics,
           "output directory must not equal or be nested under the "
           "source program directory");
    return mlir::failure();
  }

  if (createDirectory(outputParent, diagnostics))
    return mlir::failure();

  llvm::SmallString<256> canonicalOutputParent;
  if (std::error_code error =
          llvm::sys::fs::real_path(outputParent, canonicalOutputParent)) {
    reject(diagnostics, "failed to resolve output parent directory '" +
                            outputParent.str().str() + "': " + error.message());
    return mlir::failure();
  }
  llvm::SmallString<256> canonicalOutput(canonicalOutputParent);
  llvm::sys::path::append(canonicalOutput, outputName);
  if (pathIsWithin(canonicalOutput, canonicalSource)) {
    reject(diagnostics,
           "output directory must not equal or be nested under the "
           "source program directory");
    return mlir::failure();
  }
  if (pathEntryExists(canonicalOutput)) {
    reject(diagnostics, "refusing to replace existing output directory: '" +
                            canonicalOutput.str().str() + "'");
    return mlir::failure();
  }
  llvm::SmallString<256> stagingPrefix(canonicalOutputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-compile-staging");
  llvm::SmallString<256> transactionRoot;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          stagingPrefix, transactionRoot)) {
    reject(diagnostics, "failed to create compilation staging directory: " +
                            error.message());
    return mlir::failure();
  }
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(transactionRoot); });

  // Program data ownership: discover payload members, snapshot only IR/
  // metadata/directory structure, open and verify each payload exactly once,
  // and pass content-stable files to the external SPMD helper boundary.
  // Program-data storage is a sibling of compilation staging and is owned by
  // the handoff itself. A returned DeviceExecutable therefore remains readable
  // after the transaction root is removed.
  ProgramDataHandoff programData(canonicalOutputParent.str().str(),
                                 options.shouldReportDetailedTiming());
  std::string sourceMetaPath =
      programFile(canonicalSource, {llvm::StringRef("functions"),
                                    llvm::StringRef("forward.meta")});
  llvm::Expected<std::vector<frontend::ProgramInputLocator>> inputLocators =
      frontend::readProgramInputLocators(sourceMetaPath);
  if (!inputLocators) {
    reject(diagnostics, "failed to read source program input locators: " +
                            llvm::toString(inputLocators.takeError()));
    return mlir::failure();
  }
  llvm::SmallVector<std::string, 8> payloadMembers;
  llvm::SmallVector<llvm::StringRef, 8> payloadMemberRefs;
  for (const frontend::ProgramInputLocator &locator : *inputLocators) {
    if (locator.type == "parameter") {
      payloadMembers.push_back("data/" + locator.name);
    } else if (locator.type == "constant") {
      payloadMembers.push_back("constants/" + std::to_string(locator.position));
    }
  }
  for (const std::string &member : payloadMembers)
    payloadMemberRefs.push_back(member);

  llvm::SmallString<256> sourceSnapshot(transactionRoot);
  llvm::sys::path::append(sourceSnapshot, "source");
  if (copyDirectory(canonicalSource, sourceSnapshot, diagnostics,
                    payloadMemberRefs))
    return mlir::failure();

  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        diagnostic.print(diagnostics);
        diagnostics << "\n";
        return mlir::success();
      });

  auto sourceArtifact = frontend::deserializeStableHLOProgramDirectory(
      sourceSnapshot, context, diagnostics);
  if (mlir::failed(sourceArtifact))
    return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> sourceModule = std::move(*sourceArtifact);
  if (hasPostSpmdMarker(*sourceModule, sourceSnapshot)) {
    reject(diagnostics,
           "source program is already SPMD-partitioned; typed compilation "
           "requires an exporter program");
    return mlir::failure();
  }
  if (containsDialectSemantics(*sourceModule, "sdy")) {
    reject(diagnostics,
           "source program contains Shardy semantics that the external XLA "
           "SPMD helper boundary cannot consume");
    return mlir::failure();
  }
  if (mlir::failed(
          frontend::verifyStableHLOSourceModule(*sourceModule, diagnostics)))
    return mlir::failure();
  if (mlir::failed(materializeOrVerifyExactExecutionConfig(
          *sourceModule, request.getExecutionConfig())))
    return mlir::failure();
  if (timingSession) {
    wafer::support::CompileIRInventory inventory;
    inventory.record(sourceModule->getOperation());
    inventory.print("source-stablehlo", diagnostics);
  }

  // The resolver opens every payload from the canonical source exactly once
  // at owner establishment; program-directory verification then reads
  // header and extent facts from the owned pinned content instead of
  // reopening paths.
  struct TransactionPayloadResolver final
      : public frontend::ProgramPayloadResolver {
    ProgramDataHandoff &handoff;
    llvm::StringRef canonicalSource;
    mutable llvm::StringMap<SourceDataId> resolved;
    mutable std::optional<ProgramDataFailure> failure;

    TransactionPayloadResolver(ProgramDataHandoff &handoff,
                               llvm::StringRef canonicalSource)
        : handoff(handoff), canonicalSource(canonicalSource) {}

    const frontend::ProgramPayloadSource *
    resolve(llvm::StringRef locator) const override {
      auto existing = resolved.find(locator);
      if (existing != resolved.end()) {
        if (handoff.hasIOStatistics())
          ++handoff.getIOStatistics().headerReads;
        return &handoff.getSource(existing->second);
      }
      llvm::SmallString<256> path(canonicalSource);
      llvm::sys::path::append(path, locator);
      ProgramDataFailure establishmentFailure;
      llvm::Expected<SourceDataId> id =
          handoff.establishSource(path, locator, &establishmentFailure);
      if (!id) {
        llvm::consumeError(id.takeError());
        if (!failure)
          failure = std::move(establishmentFailure);
        return nullptr;
      }
      resolved[locator] = *id;
      if (handoff.hasIOStatistics())
        ++handoff.getIOStatistics().headerReads;
      return &handoff.getSource(*id);
    }
  };
  TransactionPayloadResolver resolver(programData, canonicalSource);
  if (mlir::failed(verifyProgramDirectoryMetadata(
          *sourceModule, sourceSnapshot, diagnostics, nullptr, &resolver))) {
    if (resolver.failure) {
      const ProgramDataFailure &payloadFailure = *resolver.failure;
      diagnostics << "wafer-compile: program-data-failure kind="
                  << stringifyProgramDataFailureKind(payloadFailure.kind)
                  << " locator=" << payloadFailure.locator
                  << " detail=" << payloadFailure.detail << "\n";
    }
    return mlir::failure();
  }

  // The helper input view carries IR/metadata plus the all-and-only payload
  // files the helper consumes, materialized once from owned content and
  // verified by digest readback. No payload bytes are copied for the
  // snapshot or propagated directories themselves.
  llvm::SmallString<256> helperInput(transactionRoot);
  llvm::sys::path::append(helperInput, "helper-input");
  if (copyDirectory(sourceSnapshot, helperInput, diagnostics))
    return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> helperModule = sourceModule->clone();
  eraseTargetTopologyAndExecutionMesh(*helperModule);
  if (mlir::failed(mlir::verify(*helperModule)))
    return mlir::failure();
  if (writeProgramModule(*helperModule, helperInput, diagnostics))
    return mlir::failure();
  if (std::error_code error = llvm::sys::fs::remove(programFile(
          helperInput, {llvm::StringRef("functions"),
                        llvm::StringRef("forward.stablehlo.bc")}))) {
    reject(diagnostics,
           "failed to remove portable source artifact from helper input: " +
               error.message());
    return mlir::failure();
  }
  for (const auto &entry : resolver.resolved) {
    ProgramDataFailure materializationFailure;
    if (llvm::Error error = programData.materializeSourceToFile(
            entry.getValue(), programFile(helperInput, {entry.getKey()}),
            &materializationFailure)) {
      diagnostics << "wafer-compile: program-data-failure kind="
                  << stringifyProgramDataFailureKind(
                         materializationFailure.kind)
                  << " locator=" << materializationFailure.locator
                  << " detail=" << materializationFailure.detail << "\n";
      llvm::consumeError(std::move(error));
      return mlir::failure();
    }
  }

  llvm::SmallString<256> tensorProgram(transactionRoot);
  // Post-SPMD payload verification resolves helper output (shard files)
  // through the tensor resolver: each shard is read back and established
  // exactly once as a handoff candidate, and owned content answers every
  // later verification. Constants resolve to the pre-SPMD owned source;
  // only their on-disk presence is re-checked.
  struct TensorPayloadResolver final : public frontend::ProgramPayloadResolver {
    ProgramDataHandoff &handoff;
    llvm::StringRef tensorProgram;
    mutable std::optional<ProgramDataFailure> failure;

    TensorPayloadResolver(ProgramDataHandoff &handoff,
                          llvm::StringRef tensorProgram)
        : handoff(handoff), tensorProgram(tensorProgram) {}

    const frontend::ProgramPayloadSource *
    resolve(llvm::StringRef locator) const override {
      if (const ProgramDataSource *existing = handoff.findByLocator(locator)) {
        // Constant copies in the helper output are dead data after
        // ownership, but the tensor program directory must still carry the
        // member the metadata lists.
        if (locator.starts_with("constants/")) {
          llvm::SmallString<256> member(tensorProgram);
          llvm::sys::path::append(member, locator);
          llvm::sys::fs::file_status status;
          std::error_code error = llvm::sys::fs::status(member, status);
          if (error || !llvm::sys::fs::is_regular_file(status)) {
            if (!failure)
              failure = ProgramDataFailure{
                  ProgramDataFailureKind::MissingPayload, locator.str(),
                  "tensor program directory is missing the constant member"};
            return nullptr;
          }
        }
        if (handoff.hasIOStatistics())
          ++handoff.getIOStatistics().headerReads;
        return existing;
      }
      llvm::SmallString<256> path(tensorProgram);
      llvm::sys::path::append(path, locator);
      ProgramDataFailure establishmentFailure;
      llvm::Expected<const ProgramDataSource *> candidate =
          handoff.establishHelperOutput(path, locator, &establishmentFailure);
      if (!candidate) {
        llvm::consumeError(candidate.takeError());
        if (!failure)
          failure = std::move(establishmentFailure);
        return nullptr;
      }
      if (handoff.hasIOStatistics())
        ++handoff.getIOStatistics().headerReads;
      return *candidate;
    }
  };
  llvm::sys::path::append(tensorProgram, "tensor-program");
  // Constructed after the path append: the resolver holds a StringRef into
  // the tensor-program path buffer.
  TensorPayloadResolver tensorResolver(programData, tensorProgram);
  stages.enter(CompilationStage::SpmdPartitioning);
  if (mlir::failed(runSpmdHelper(xlaSpmdPartitionerHelper, helperInput,
                                 tensorProgram, request.getExecutionConfig(),
                                 diagnostics)))
    return mlir::failure();

  stages.enter(CompilationStage::TensorProgramPreparation);
  if (validateRegularDirectoryTree(tensorProgram, diagnostics))
    return mlir::failure();
  for (llvm::StringRef requiredMember :
       {llvm::StringRef("forward.mlir"), llvm::StringRef("forward.meta")}) {
    if (!isRegularFile(programFile(
            tensorProgram, {llvm::StringRef("functions"), requiredMember}))) {
      reject(diagnostics,
             "XLA SPMD partitioner output is missing required regular member "
             "'functions/" +
                 requiredMember.str() + "'");
      return mlir::failure();
    }
  }
  if (!isRegularFile(programFile(
          tensorProgram, {llvm::StringRef("functions"),
                          llvm::StringRef("forward.parameter_shards.json")}))) {
    reject(diagnostics,
           "XLA SPMD partitioner output is missing its partition marker");
    return mlir::failure();
  }
  // The source snapshot carries no payload members, so this merge only
  // restores structure and exporter members such as auxiliary data files.
  if (mergeMissingProgramMembers(sourceSnapshot, tensorProgram, diagnostics) ||
      validateRegularDirectoryTree(tensorProgram, diagnostics))
    return mlir::failure();
  // The helper output does not carry constant payloads; restore them from
  // owned content so the tensor program directory stays complete. These
  // writes are counted by optional I/O statistics with digest readback.
  for (const auto &entry : resolver.resolved) {
    if (!entry.getKey().starts_with("constants/"))
      continue;
    ProgramDataFailure constantFailure;
    std::string restored = programFile(tensorProgram, {entry.getKey()});
    if (llvm::Error error = programData.materializeSourceToFile(
            entry.getValue(), restored, &constantFailure)) {
      diagnostics << "wafer-compile: program-data-failure kind="
                  << stringifyProgramDataFailureKind(constantFailure.kind)
                  << " locator=" << constantFailure.locator
                  << " detail=" << constantFailure.detail << "\n";
      llvm::consumeError(std::move(error));
      return mlir::failure();
    }
  }

  mlir::OwningOpRef<mlir::ModuleOp> tensorModule =
      parseProgramDirectoryModule(tensorProgram, context);
  if (!tensorModule)
    return mlir::failure();
  if (mlir::failed(materializeOrVerifyExactExecutionConfig(
          *tensorModule, request.getExecutionConfig())))
    return mlir::failure();
  if (!hasPostSpmdMarker(*tensorModule, tensorProgram)) {
    reject(diagnostics,
           "XLA SPMD partitioner output is missing its partition marker");
    return mlir::failure();
  }
  if (containsDialectSemantics(*tensorModule, "sdy")) {
    reject(diagnostics,
           "XLA SPMD partitioner output still contains Shardy semantics");
    return mlir::failure();
  }
  if (mlir::failed(verifyStablehloStageOperations(*tensorModule)))
    return mlir::failure();
  if (timingSession) {
    wafer::support::CompileIRInventory inventory;
    inventory.record(tensorModule->getOperation());
    inventory.print("post-spmd-stablehlo", diagnostics);
  }
  frontend::FrontendProgramVerificationResult verifiedTensorFacts;
  if (mlir::failed(verifyProgramDirectoryMetadata(
          *tensorModule, tensorProgram, diagnostics, &verifiedTensorFacts,
          &tensorResolver))) {
    if (tensorResolver.failure) {
      const ProgramDataFailure &payloadFailure = *tensorResolver.failure;
      diagnostics << "wafer-compile: program-data-failure kind="
                  << stringifyProgramDataFailureKind(payloadFailure.kind)
                  << " locator=" << payloadFailure.locator
                  << " detail=" << payloadFailure.detail << "\n";
    }
    return mlir::failure();
  }

  // Helper shard readback: each shard was established exactly once during
  // tensor verification. Prove the shard payload bytes against the owned
  // pre-SPMD source region; byte-identical shards keep referencing the
  // original source, real partitions are adopted as new owned sources.
  auto reportProgramDataFailure = [&](const ProgramDataFailure &failure) {
    diagnostics << "wafer-compile: program-data-failure kind="
                << stringifyProgramDataFailureKind(failure.kind)
                << " locator=" << failure.locator
                << " detail=" << failure.detail << "\n";
  };
  constexpr int64_t kSingleCardPartitionId = 0;
  auto establishRange =
      [&](ProgramTensorId tensorId, ProgramElementType dtype,
          llvm::ArrayRef<int64_t> globalShape,
          llvm::ArrayRef<int64_t> localShape,
          frontend::ProgramDistributionKind distribution,
          ProgramDataRangeOrigin origin, llvm::ArrayRef<int64_t> offsets,
          llvm::ArrayRef<int64_t> sizes, SourceDataId sourceId) -> bool {
    ProgramDataFailure rangeFailure;
    llvm::Expected<ProgramDataRange> range = ProgramDataRange::create(
        tensorId, dtype, globalShape, localShape, distribution, origin, offsets,
        sizes, llvm::ArrayRef<int64_t>(std::vector<int64_t>(offsets.size(), 1)),
        sourceId, programData.getSource(sourceId), &rangeFailure);
    if (!range) {
      llvm::consumeError(range.takeError());
      reportProgramDataFailure(rangeFailure);
      return true;
    }
    if (llvm::Error error = programData.addRange(std::move(*range))) {
      reject(diagnostics, llvm::toString(std::move(error)));
      return true;
    }
    return false;
  };
  for (const frontend::ProgramParameterBinding &parameter :
       verifiedTensorFacts.parameters) {
    const frontend::ProgramPartitionSlice *partitionSlice = nullptr;
    for (const frontend::ProgramPartitionSlice &slice :
         parameter.partitionSlices) {
      if (slice.partitionId != kSingleCardPartitionId)
        continue;
      if (partitionSlice) {
        reject(diagnostics,
               "parameter shard metadata contains duplicate partition "
               "slices: " +
                   parameter.name);
        return mlir::failure();
      }
      partitionSlice = &slice;
    }
    if (!partitionSlice) {
      reject(diagnostics,
             "parameter shard metadata is missing the card partition slice: " +
                 parameter.name);
      return mlir::failure();
    }
    auto original = resolver.resolved.find("data/" + parameter.name);
    if (original == resolver.resolved.end()) {
      reject(diagnostics,
             "parameter payload has no established transaction source: " +
                 parameter.name);
      return mlir::failure();
    }
    const ProgramDataSource *shardSource =
        programData.findByLocator(partitionSlice->payloadPath);
    if (!shardSource) {
      reject(diagnostics,
             "parameter shard payload has no established transaction "
             "candidate: " +
                 partitionSlice->payloadPath);
      return mlir::failure();
    }
    ProgramDataFailure shardFailure;
    llvm::Expected<bool> byteIdentical = programData.verifyShardAgainstSource(
        *shardSource, original->second, partitionSlice->offsets,
        partitionSlice->sizes, &shardFailure);
    if (!byteIdentical) {
      llvm::consumeError(byteIdentical.takeError());
      reportProgramDataFailure(shardFailure);
      return mlir::failure();
    }
    const ProgramTensorId tensorId{ProgramResourceRole::Parameter,
                                   parameter.argumentIndex};
    if (*byteIdentical) {
      // Replication or contiguous slice: reference the original source.
      if (establishRange(tensorId, parameter.dtype, parameter.globalShape,
                         parameter.localShape, parameter.distribution,
                         ProgramDataRangeOrigin::OriginalSource,
                         partitionSlice->offsets, partitionSlice->sizes,
                         original->second))
        return mlir::failure();
    } else {
      SourceDataId shardId = programData.adoptCandidate(shardSource);
      if (establishRange(
              tensorId, parameter.dtype, parameter.globalShape,
              parameter.localShape, parameter.distribution,
              ProgramDataRangeOrigin::MaterializedShard,
              /*offsets=*/std::vector<int64_t>(parameter.localShape.size(), 0),
              parameter.localShape, shardId))
        return mlir::failure();
    }
  }
  for (const frontend::ProgramConstantBinding &constant :
       verifiedTensorFacts.constants) {
    std::string locator = "constants/" + std::to_string(constant.position);
    auto established = resolver.resolved.find(locator);
    if (established == resolver.resolved.end()) {
      reject(diagnostics,
             "captured constant payload has no established transaction "
             "source: " +
                 locator);
      return mlir::failure();
    }
    if (establishRange({ProgramResourceRole::Constant, constant.position},
                       constant.dtype, constant.shape, constant.shape,
                       frontend::ProgramDistributionKind::Replicated,
                       ProgramDataRangeOrigin::OriginalSource,
                       std::vector<int64_t>(constant.shape.size(), 0),
                       constant.shape, established->second))
      return mlir::failure();
  }

  if (mlir::failed(wafer::support::runPassPipeline(
          *tensorModule, "stablehlo-to-linalg",
          wafer::buildStablehloToLinalgPipeline)))
    return mlir::failure();
  if (containsDialectSemantics(*tensorModule, "stablehlo") ||
      containsDialectSemantics(*tensorModule, "sdy")) {
    reject(diagnostics,
           "tensor program still contains frontend or sharding operations");
    return mlir::failure();
  }
  if (mlir::failed(verifyTensorProgramStageOperations(*tensorModule)))
    return mlir::failure();
  if (writeProgramModule(*tensorModule, tensorProgram, diagnostics))
    return mlir::failure();

  mlir::OwningOpRef<mlir::ModuleOp> verifiedTensorModule =
      parseProgramDirectoryModule(tensorProgram, context);
  if (!verifiedTensorModule)
    return mlir::failure();
  if (mlir::failed(verifyExactExecutionConfigInternal(
          *verifiedTensorModule, request.getExecutionConfig())))
    return mlir::failure();
  if (!hasPostSpmdMarker(*verifiedTensorModule, tensorProgram)) {
    reject(diagnostics,
           "tensor-program readback is missing its partition marker");
    return mlir::failure();
  }
  if (containsDialectSemantics(*verifiedTensorModule, "stablehlo") ||
      containsDialectSemantics(*verifiedTensorModule, "sdy")) {
    reject(diagnostics,
           "tensor-program readback contains frontend or sharding semantics");
    return mlir::failure();
  }
  if (mlir::failed(verifyTensorProgramStageOperations(*verifiedTensorModule)) ||
      mlir::failed(verifyProgramDirectoryMetadata(*verifiedTensorModule,
                                                  tensorProgram, diagnostics,
                                                  nullptr, &tensorResolver)))
    return mlir::failure();
  if (timingSession) {
    wafer::support::CompileIRInventory inventory;
    inventory.record(verifiedTensorModule->getOperation());
    inventory.print("tensor-linalg", diagnostics);
  }

  sourceTiming.reset();
  if (timingSession)
    diagnostics << "wafer-compile: compile-stats stage=source-to-tensor-program"
                << " wall_ms=" << elapsedCompileMilliseconds(transactionStart)
                << " peak_rss_kib=" << getCompilePeakRSSKiB() << "\n";

  const CompileClock::time_point targetCodeGenStart = CompileClock::now();
  auto targetCodeGenTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "source-to-package", "target-codegen");
  std::optional<DeviceExecutable> deviceExecutable;
  std::optional<TargetLLVMModules> targetLLVMModules;
  CompilationIRTrace irTrace;
  ProfileInstrumentationIdentity profileIdentity;
  if (options.shouldProduceProfileInstrumentation()) {
    if (mlir::failed(stageProfileTargetPackages(
            tensorProgram, transactionRoot, request.getExecutionConfig(),
            options.getOptimizationConfig(), targetToolchain, diagnostics,
            failAfterLaunchSlot, failAfterTargetLaunchSlot,
            failAfterPackageLaunchSlot, deviceExecutable, targetLLVMModules,
            programData, tensorResolver, irTrace, stages, profileIdentity)))
      return mlir::failure();
  } else if (mlir::failed(stageTargetPackage(
                 tensorProgram, transactionRoot, request.getExecutionConfig(),
                 options.getOptimizationConfig(), targetToolchain, diagnostics,
                 failAfterLaunchSlot, failAfterTargetLaunchSlot,
                 failAfterPackageLaunchSlot, deviceExecutable,
                 targetLLVMModules, programData, tensorResolver, irTrace,
                 stages, qualification))) {
    return mlir::failure();
  }
  if (timingSession) {
    diagnostics << "wafer-compile: program-data-io ";
    deviceExecutable->getProgramDataHandoff().getIOStatistics().print(
        diagnostics);
    diagnostics << "\n";
  }
  targetCodeGenTiming.reset();
  if (timingSession)
    diagnostics << "wafer-compile: compile-stats stage=target-codegen"
                << " wall_ms=" << elapsedCompileMilliseconds(targetCodeGenStart)
                << " peak_rss_kib=" << getCompilePeakRSSKiB() << " profile="
                << (options.shouldProduceProfileInstrumentation() ? "true"
                                                                  : "false")
                << "\n";

  const CompileClock::time_point outputRenameStart = CompileClock::now();
  auto outputRenameTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "source-to-package", "output-rename");
  stages.enter(CompilationStage::PackageCommit);

  // Single-visibility-point commit. Every fallible step closes here, before
  // the one no-replace rename that publishes the output:
  // 1. the staged package is freshly loaded and verified;
  // 2. its all-and-only members are opened and bound by exact size and digest
  //    against the verified manifest;
  // 3. an explicitly requested profile instrumentation is bound by digest
  //    against the same staged inodes.
  // The ordinary case publishes the staged package as the output directory.
  // The profile case publishes the common delivery root `<output>`
  // containing exactly `package/` and `package.profile/`, so the package and
  // the instrumentation become visible in one rename and the runtime sibling
  // rule `<package-root>.profile` is preserved. After the rename no step can
  // fail, so a failure return always leaves this invocation's output
  // invisible.
  const bool profileRequested = options.shouldProduceProfileInstrumentation();
  llvm::SmallString<256> stagedPackage(transactionRoot);
  if (profileRequested) {
    llvm::sys::path::append(stagedPackage, "delivery", "package");
  } else {
    llvm::sys::path::append(stagedPackage, "package");
  }
  if (commitFailureInjection ==
      CommitFailureInjection::CorruptPackageProgramData) {
    llvm::SmallString<256> programDataPath(stagedPackage);
    llvm::sys::path::append(programDataPath, "data", "program-data.bin");
    std::error_code error;
    llvm::raw_fd_ostream stream(programDataPath, error,
                                llvm::sys::fs::OF_Append);
    if (error) {
      reject(diagnostics,
             "test-only staged package corruption failed: " + error.message());
      return mlir::failure();
    }
    stream << 'x';
    stream.close();
    if (stream.has_error()) {
      reject(diagnostics,
             "test-only staged package corruption failed while writing");
      return mlir::failure();
    }
  }
  llvm::Expected<runtime::detail::BoundExecutablePackage> boundPackage =
      runtime::detail::bindExecutablePackage(stagedPackage);
  if (!boundPackage) {
    reject(diagnostics, "staged package manifest verification failed: " +
                            llvm::toString(boundPackage.takeError()));
    return mlir::failure();
  }
  llvm::SmallString<256> publishedPackageRoot(canonicalOutput);
  if (profileRequested)
    llvm::sys::path::append(publishedPackageRoot, "package");
  std::string publishedPackageRootStorage = publishedPackageRoot.str().str();
  std::optional<BoundProfileInstrumentation> boundProfileInstrumentation;
  std::string publishedInstrumentationRootStorage;
  if (profileRequested) {
    llvm::SmallString<256> stagedInstrumentation(transactionRoot);
    llvm::sys::path::append(stagedInstrumentation, "delivery",
                            "package.profile");
    if (commitFailureInjection == CommitFailureInjection::CorruptProfilePlan) {
      llvm::SmallString<256> planPath(stagedInstrumentation);
      llvm::sys::path::append(planPath, "plan.json");
      std::error_code error;
      llvm::raw_fd_ostream stream(planPath, error, llvm::sys::fs::OF_Append);
      if (error) {
        reject(diagnostics, "test-only staged profile corruption failed: " +
                                error.message());
        return mlir::failure();
      }
      stream << ' ';
      stream.close();
      if (stream.has_error()) {
        reject(diagnostics,
               "test-only staged profile corruption failed while writing");
        return mlir::failure();
      }
    }
    // The co-commit verification is unconditional on the requested product
    // set, never on whether a caller retains the result member.
    llvm::Expected<BoundProfileInstrumentation> boundInstrumentation =
        bindProfileInstrumentation(*boundPackage, stagedInstrumentation,
                                   profileIdentity);
    if (!boundInstrumentation) {
      reject(diagnostics, "staged profile instrumentation binding failed: " +
                              llvm::toString(boundInstrumentation.takeError()));
      return mlir::failure();
    }
    boundProfileInstrumentation.emplace(std::move(*boundInstrumentation));
    if (retainedProfileProduct) {
      llvm::SmallString<256> publishedInstrumentationRoot(canonicalOutput);
      llvm::sys::path::append(publishedInstrumentationRoot, "package.profile");
      publishedInstrumentationRootStorage =
          publishedInstrumentationRoot.str().str();
    }
  }

  if (commitFailureInjection == CommitFailureInjection::FailAfterVerification) {
    reject(diagnostics, "test-only commit verification failure injected before "
                        "publication");
    return mlir::failure();
  }

  llvm::SmallString<256> publicationRoot(transactionRoot);
  if (profileRequested) {
    llvm::sys::path::append(publicationRoot, "delivery");
  } else {
    llvm::sys::path::append(publicationRoot, "package");
  }
  // The single visibility point. Everything above was fallible; everything
  // below only moves already-verified owners into the result.
  if (renameDirectoryNoReplace(publicationRoot, canonicalOutput, diagnostics))
    return mlir::failure();
  if (retainedPackage)
    retainedPackage->emplace(runtime::detail::ExecutablePackageFactory::make(
        std::move(publishedPackageRootStorage), std::move(*boundPackage)));
  if (profileRequested && retainedProfileProduct) {
    if (!boundProfileInstrumentation)
      llvm_unreachable("profile transaction lost its bound resources");
    retainedProfileProduct->emplace(ProfileInstrumentationProductBuilder::make(
        std::move(publishedInstrumentationRootStorage),
        std::move(profileIdentity), std::move(*boundProfileInstrumentation)));
  }
  if (retainedDeviceExecutable)
    retainedDeviceExecutable->emplace(std::move(*deviceExecutable));
  if (retainedTargetLLVMModules)
    retainedTargetLLVMModules->emplace(std::move(*targetLLVMModules));
  if (retainedIRTrace)
    retainedIRTrace->emplace(std::move(irTrace));
  outputRenameTiming.reset();
  if (timingSession) {
    diagnostics << "wafer-compile: compile-stats stage=output-rename"
                << " wall_ms=" << elapsedCompileMilliseconds(outputRenameStart)
                << " peak_rss_kib=" << getCompilePeakRSSKiB() << "\n";
    diagnostics << "wafer-compile: compile-stats stage=compile-transaction"
                << " wall_ms=" << elapsedCompileMilliseconds(transactionStart)
                << " peak_rss_kib=" << getCompilePeakRSSKiB() << "\n";
  }
  return mlir::success();
#endif
}

} // namespace wafer::compiler::detail
