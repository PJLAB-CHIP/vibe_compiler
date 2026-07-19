//===- wafer-qualify-optimizations.cpp - Isolated qualification runner --===//

#include "Compiler/OptimizationQualificationCompiler.h"

#include "Wafer/Compiler/TargetArtifact.h"
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
#include "Wafer/Model/TargetModelExpectedOutputGate.h"
#endif
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
#include "Wafer/Model/TargetBulkModel.h"
#endif
#include "Wafer/Support/OptimizationArtifactDigest.h"
#include "Wafer/Support/OptimizationInvocation.h"
#include "Wafer/Support/OptimizationQualificationArchiveStore.h"
#include "Wafer/Support/OptimizationQualificationExecution.h"
#include "Wafer/Target/TargetProfile.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef WAFER_XLA_SPMD_PARTITIONER_HELPER
#define WAFER_XLA_SPMD_PARTITIONER_HELPER ""
#endif
#ifndef WAFER_PYTHON_EXECUTABLE
#define WAFER_PYTHON_EXECUTABLE ""
#endif
#ifndef WAFER_DEVICE_LINKER_SCRIPT
#define WAFER_DEVICE_LINKER_SCRIPT ""
#endif

namespace {

struct ModelBinding {
  int64_t index = -1;
  std::string path;
};

struct Options {
  bool worker = false;
  bool runTargetModel = false;
  std::string archiveRoot;
  std::string workDirectory;
  std::string sourceProgramDirectory;
  std::string sourceInput;
  std::string sourceExpected;
  std::string llamaProgramDirectory;
  std::string llamaInput;
  std::string llamaExpected;
  double sourceAtol = 1e-6;
  double sourceRtol = 1e-5;
  double llamaAtol = 0.02;
  double llamaRtol = 0.01;

  std::string inputProgramDirectory;
  std::string outputProgramDirectory;
  std::string resultFile;
  uint32_t rankCount = 0;
  uint32_t corpusId = 0;
  uint64_t executionOrdinal = 0;
  uint32_t normalizationRepetitions = 1;
  std::string configurationSpelling = "all-on";
  wafer::EquivalentInputVariantV1 inputVariant =
      wafer::EquivalentInputVariantV1::Original;
  wafer::AdoptionDigest qualificationRunDigest{};
  wafer::AdoptionDigest inputSnapshotDigest{};
  std::vector<ModelBinding> modelInputs;
  std::vector<ModelBinding> modelExpected;
  double modelAtol = 0.0;
  double modelRtol = 0.0;
};

struct QualificationCasePaths {
  wafer::QualificationCaseKeyV1 originalCase;
  std::string programDirectory;
  std::vector<ModelBinding> modelInputs;
  std::vector<ModelBinding> modelExpected;
  double atol = 0.0;
  double rtol = 0.0;
};

enum class ExecutionPurpose {
  Equivalent,
  NormalizationOnce,
  NormalizationTwice,
  Production,
};

struct ExecutionPlan {
  ExecutionPurpose purpose = ExecutionPurpose::Equivalent;
  size_t caseIndex = 0;
  wafer::QualificationCaseKeyV1 caseKey;
  wafer::OptimizationConfigurationV1 configuration;
  uint32_t normalizationRepetitions = 1;
  bool runTargetModel = false;
  uint64_t executionOrdinal = 0;
};

struct CompletedExecution {
  ExecutionPlan plan;
  wafer::OptimizationQualificationExecutionResultV1 result;
  wafer::AdoptionDigest resultDigest{};
  uint64_t wallNs = 0;
  uint64_t peakRssBytes = 0;
};

static bool fail(llvm::Twine message) {
  llvm::errs() << "wafer-qualify-optimizations: " << message << "\n";
  return false;
}

static bool isZero(const wafer::AdoptionDigest &digest) {
  return llvm::all_of(digest, [](uint8_t byte) { return byte == 0; });
}

static bool parseU64(llvm::StringRef text, uint64_t &value) {
  return !text.empty() && !text.getAsInteger(10, value);
}

static bool parseDouble(llvm::StringRef text, double &value) {
  std::string storage = text.str();
  char *end = nullptr;
  errno = 0;
  value = std::strtod(storage.c_str(), &end);
  return errno == 0 && end == storage.c_str() + storage.size() && value >= 0.0;
}

static bool parseDigest(llvm::StringRef text, wafer::AdoptionDigest &digest) {
  if (text.size() != digest.size() * 2)
    return false;
  auto nibble = [](char character) -> std::optional<uint8_t> {
    if (character >= '0' && character <= '9')
      return static_cast<uint8_t>(character - '0');
    if (character >= 'a' && character <= 'f')
      return static_cast<uint8_t>(character - 'a' + 10);
    return std::nullopt;
  };
  for (size_t index = 0; index < digest.size(); ++index) {
    std::optional<uint8_t> high = nibble(text[index * 2]);
    std::optional<uint8_t> low = nibble(text[index * 2 + 1]);
    if (!high || !low)
      return false;
    digest[index] = static_cast<uint8_t>((*high << 4) | *low);
  }
  return true;
}

static bool parseBinding(llvm::StringRef text, ModelBinding &binding) {
  auto parts = text.split('=');
  uint64_t index = 0;
  if (parts.first.empty() || parts.second.empty() ||
      !parseU64(parts.first, index) ||
      index > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  binding = {static_cast<int64_t>(index), parts.second.str()};
  return true;
}

static bool parseOptions(int argc, char **argv, Options &options) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef argument(argv[index]);
    if (argument == "--worker") {
      options.worker = true;
      continue;
    }
    if (argument == "--run-target-model") {
      options.runTargetModel = true;
      continue;
    }
    auto take = [&](llvm::StringRef name) -> std::optional<llvm::StringRef> {
      if (argument != name || index + 1 == argc)
        return std::nullopt;
      return llvm::StringRef(argv[++index]);
    };
    if (auto value = take("--archive-root"))
      options.archiveRoot = value->str();
    else if (auto value = take("--work-dir"))
      options.workDirectory = value->str();
    else if (auto value = take("--source-program-dir"))
      options.sourceProgramDirectory = value->str();
    else if (auto value = take("--source-input"))
      options.sourceInput = value->str();
    else if (auto value = take("--source-expected"))
      options.sourceExpected = value->str();
    else if (auto value = take("--llama-program-dir"))
      options.llamaProgramDirectory = value->str();
    else if (auto value = take("--llama-input"))
      options.llamaInput = value->str();
    else if (auto value = take("--llama-expected"))
      options.llamaExpected = value->str();
    else if (auto value = take("--source-atol")) {
      if (!parseDouble(*value, options.sourceAtol))
        return false;
    } else if (auto value = take("--source-rtol")) {
      if (!parseDouble(*value, options.sourceRtol))
        return false;
    } else if (auto value = take("--llama-atol")) {
      if (!parseDouble(*value, options.llamaAtol))
        return false;
    } else if (auto value = take("--llama-rtol")) {
      if (!parseDouble(*value, options.llamaRtol))
        return false;
    } else if (auto value = take("--input-program-dir"))
      options.inputProgramDirectory = value->str();
    else if (auto value = take("--output-program-dir"))
      options.outputProgramDirectory = value->str();
    else if (auto value = take("--result-file"))
      options.resultFile = value->str();
    else if (auto value = take("--execution-ranks")) {
      uint64_t parsed = 0;
      if (!parseU64(*value, parsed) || parsed > UINT32_MAX)
        return false;
      options.rankCount = static_cast<uint32_t>(parsed);
    } else if (auto value = take("--corpus-id")) {
      uint64_t parsed = 0;
      if (!parseU64(*value, parsed) || parsed > UINT32_MAX)
        return false;
      options.corpusId = static_cast<uint32_t>(parsed);
    } else if (auto value = take("--execution-ordinal")) {
      if (!parseU64(*value, options.executionOrdinal))
        return false;
    } else if (auto value = take("--normalization-repetitions")) {
      uint64_t parsed = 0;
      if (!parseU64(*value, parsed) || parsed > UINT32_MAX)
        return false;
      options.normalizationRepetitions = static_cast<uint32_t>(parsed);
    } else if (auto value = take("--configuration")) {
      options.configurationSpelling = value->str();
    } else if (auto value = take("--input-variant")) {
      if (*value == "original")
        options.inputVariant = wafer::EquivalentInputVariantV1::Original;
      else if (*value == "metamorphic")
        options.inputVariant = wafer::EquivalentInputVariantV1::Metamorphic;
      else
        return false;
    } else if (auto value = take("--qualification-run-digest")) {
      if (!parseDigest(*value, options.qualificationRunDigest))
        return false;
    } else if (auto value = take("--input-snapshot-digest")) {
      if (!parseDigest(*value, options.inputSnapshotDigest))
        return false;
    } else if (auto value = take("--model-input")) {
      ModelBinding binding;
      if (!parseBinding(*value, binding))
        return false;
      options.modelInputs.push_back(std::move(binding));
    } else if (auto value = take("--model-expected")) {
      ModelBinding binding;
      if (!parseBinding(*value, binding))
        return false;
      options.modelExpected.push_back(std::move(binding));
    } else if (auto value = take("--model-atol")) {
      if (!parseDouble(*value, options.modelAtol))
        return false;
    } else if (auto value = take("--model-rtol")) {
      if (!parseDouble(*value, options.modelRtol))
        return false;
    } else {
      return false;
    }
  }
  if (options.worker)
    return !options.inputProgramDirectory.empty() &&
           !options.outputProgramDirectory.empty() &&
           !options.resultFile.empty() && options.rankCount != 0 &&
           options.corpusId != 0 && !isZero(options.qualificationRunDigest) &&
           !isZero(options.inputSnapshotDigest) &&
           (!options.runTargetModel ||
            (!options.modelInputs.empty() && !options.modelExpected.empty()));
  return !options.archiveRoot.empty() && !options.workDirectory.empty() &&
         !options.sourceProgramDirectory.empty() &&
         !options.sourceInput.empty() && !options.sourceExpected.empty() &&
         !options.llamaProgramDirectory.empty() &&
         !options.llamaInput.empty() && !options.llamaExpected.empty();
}

static void printUsage() {
  llvm::errs()
      << "usage: wafer-qualify-optimizations --archive-root <dir> "
         "--work-dir <dir> --source-program-dir <dir> --source-input <npy> "
         "--source-expected <npy> --llama-program-dir <dir> "
         "--llama-input <npy> --llama-expected <npy> "
         "[--source-atol <n>] [--source-rtol <n>] "
         "[--llama-atol <n>] [--llama-rtol <n>]\n";
}

static wafer::OptimizationConfigurationV1 allOnEvidenceConfiguration() {
  return {wafer::OptimizationGroupSelectionV1::AllOn, std::nullopt,
          wafer::OptimizationGroupSelectionV1::AllOn, std::nullopt};
}

static wafer::OptimizationConfigurationV1 allOffEvidenceConfiguration() {
  return {wafer::OptimizationGroupSelectionV1::AllOff, std::nullopt,
          wafer::OptimizationGroupSelectionV1::AllOff, std::nullopt};
}

static wafer::OptimizationConfigurationV1
equivalentEvidenceConfiguration(bool cleanupEnabled) {
  return {wafer::OptimizationGroupSelectionV1::AllOff, std::nullopt,
          cleanupEnabled ? wafer::OptimizationGroupSelectionV1::AllOn
                         : wafer::OptimizationGroupSelectionV1::AllOff,
          std::nullopt};
}

static std::optional<wafer::OptimizationConfiguration>
toCompilerConfiguration(const wafer::OptimizationConfigurationV1 &configuration,
                        std::string &diagnostic) {
  auto convert = [](wafer::OptimizationGroupSelectionV1 value) {
    switch (value) {
    case wafer::OptimizationGroupSelectionV1::AllOn:
      return wafer::OptimizationGroupSelectionKind::AllOn;
    case wafer::OptimizationGroupSelectionV1::AllOff:
      return wafer::OptimizationGroupSelectionKind::AllOff;
    case wafer::OptimizationGroupSelectionV1::DisableOne:
      return wafer::OptimizationGroupSelectionKind::DisableOne;
    }
    return wafer::OptimizationGroupSelectionKind::AllOn;
  };
  wafer::OptimizationConfiguration result{
      {convert(configuration.fixedSelection), configuration.fixedDisabledKey},
      {convert(configuration.cleanupSelection),
       configuration.cleanupDisabledKey}};
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  if (!wafer::validateOptimizationConfiguration(proposal, result, &diagnostic))
    return std::nullopt;
  return result;
}

static std::optional<wafer::OptimizationConfigurationV1>
parseEvidenceConfiguration(llvm::StringRef spelling) {
  if (spelling == "all-on")
    return allOnEvidenceConfiguration();
  if (spelling == "all-off")
    return allOffEvidenceConfiguration();
  if (spelling == "fixed-off-cleanup-off")
    return equivalentEvidenceConfiguration(false);
  if (spelling == "fixed-off-cleanup-on")
    return equivalentEvidenceConfiguration(true);
  return std::nullopt;
}

static std::string
configurationSpelling(const wafer::OptimizationConfigurationV1 &configuration) {
  if (configuration.fixedSelection ==
          wafer::OptimizationGroupSelectionV1::AllOn &&
      configuration.cleanupSelection ==
          wafer::OptimizationGroupSelectionV1::AllOn)
    return "all-on";
  if (configuration.fixedSelection ==
          wafer::OptimizationGroupSelectionV1::AllOff &&
      configuration.cleanupSelection ==
          wafer::OptimizationGroupSelectionV1::AllOff)
    return "fixed-off-cleanup-off";
  return "fixed-off-cleanup-on";
}

static std::optional<wafer::QualificationCaseKeyV1>
findCase(uint32_t corpusId, uint32_t rankCount,
         wafer::EquivalentInputVariantV1 variant) {
  for (const wafer::QualificationCaseKeyV1 &candidate :
       wafer::getCurrentMandatoryQualificationCasesV1())
    if (candidate.corpus.id == corpusId && candidate.rankCount == rankCount) {
      wafer::QualificationCaseKeyV1 result = candidate;
      result.inputVariant = variant;
      return result;
    }
  return std::nullopt;
}

static bool writeBytes(llvm::StringRef path, llvm::ArrayRef<uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    return fail("cannot create result file: " + error.message());
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  return !output.has_error() || fail("cannot write result file");
}

static bool readBytes(llvm::StringRef path, std::vector<uint8_t> &bytes) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return fail("cannot read result file: " + buffer.getError().message());
  llvm::StringRef contents = (*buffer)->getBuffer();
  bytes.assign(contents.bytes_begin(), contents.bytes_end());
  return true;
}

static void appendU64(std::vector<uint8_t> &bytes, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static void appendString(std::vector<uint8_t> &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

static wafer::AdoptionDigest
combineDigests(llvm::StringRef domain,
               llvm::ArrayRef<wafer::AdoptionDigest> digests,
               llvm::ArrayRef<std::string> strings = {}) {
  std::vector<uint8_t> bytes;
  for (const wafer::AdoptionDigest &digest : digests)
    bytes.insert(bytes.end(), digest.begin(), digest.end());
  for (const std::string &value : strings)
    appendString(bytes, value);
  return wafer::digestOptimizationBytesV1(domain, bytes);
}

#ifdef WAFER_ENABLE_SYSTEMC_MODEL
static wafer::AdoptionDigest
digestTargetModelResult(const wafer::model::TargetModelResult &result) {
  std::vector<uint8_t> bytes;
  for (uint64_t value :
       {static_cast<uint64_t>(result.completedRankCount),
        result.issuedTransactionCount, result.systemCThreadProcessCount,
        result.finalDeltaCount, result.formalNumericCommandCount,
        result.managedReferenceNumericCommandCount,
        result.managedReferenceScalarEvaluationCount,
        result.bulkNumericCommandCount, result.bulkMatmulInvocationCount,
        result.bulkReorderInvocationCount,
        result.bulkFormalFusedMultiplyAddCount})
    appendU64(bytes, value);
  bytes.push_back(result.numericFlags.invalid);
  bytes.push_back(result.numericFlags.divByZero);
  bytes.push_back(result.numericFlags.overflow);
  bytes.push_back(result.numericFlags.underflow);
  bytes.push_back(result.numericFlags.inexact);
  for (const std::string &value :
       {result.systemCVersion, result.schedulerIdentity})
    appendString(bytes, value);
  for (const auto *values : {&result.bulkAdmissionRecordDigests,
                             &result.bulkManagedReferenceEnvironmentDigests,
                             &result.managedReferenceTensorEnvironmentDigests,
                             &result.managedReferenceTensorImplementations}) {
    appendU64(bytes, values->size());
    for (const std::string &value : *values)
      appendString(bytes, value);
  }
  appendU64(bytes, result.outputs.size());
  for (const wafer::model::TargetModelOutput &output : result.outputs) {
    appendU64(bytes, static_cast<uint64_t>(output.logicalRank));
    appendU64(bytes, static_cast<uint64_t>(output.slotOrdinal));
    appendU64(bytes, static_cast<uint64_t>(output.resourceIndex));
    wafer::AdoptionDigest outputDigest = wafer::digestOptimizationBytesV1(
        "wafer.optimization-qualification-model-output-v1", output.bytes);
    bytes.insert(bytes.end(), outputDigest.begin(), outputDigest.end());
  }
  return wafer::digestOptimizationBytesV1(
      "wafer.optimization-qualification-target-model-v1", bytes);
}
#endif

static int runWorker(const Options &options) {
  std::optional<wafer::QualificationCaseKeyV1> caseKey =
      findCase(options.corpusId, options.rankCount, options.inputVariant);
  std::optional<wafer::OptimizationConfigurationV1> evidenceConfiguration =
      parseEvidenceConfiguration(options.configurationSpelling);
  if (!caseKey || !evidenceConfiguration) {
    fail("worker case or configuration is invalid");
    return 2;
  }
  std::string diagnostic;
  std::optional<wafer::OptimizationConfiguration> configuration =
      toCompilerConfiguration(*evidenceConfiguration, diagnostic);
  if (!configuration) {
    fail(diagnostic);
    return 2;
  }
  if (options.executionOrdinal >=
      std::numeric_limits<uint64_t>::max() /
          wafer::kOptimizationInvocationLocalOrdinalLimit) {
    fail("worker execution ordinal is out of range");
    return 2;
  }
  wafer::OptimizationInvocationScopeContextV1 scope;
  scope.scopeKind = wafer::InvocationScopeKindV1::QualificationRun;
  scope.scopeDigest = options.qualificationRunDigest;
  scope.qualificationCase = *caseKey;
  scope.invocationOrdinalBase = options.executionOrdinal *
                                wafer::kOptimizationInvocationLocalOrdinalLimit;
  auto journal = std::make_shared<wafer::OptimizationInvocationJournalV1>();
  wafer::ScopedOptimizationInvocationRecorder recorder(journal, scope);
  if (!recorder.installed()) {
    fail("worker cannot install invocation recorder");
    return 1;
  }
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      options.rankCount, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  auto request = executionConfig
                     ? wafer::compiler::CompilationRequest::create(
                           options.inputProgramDirectory, *executionConfig)
                     : llvm::Expected<wafer::compiler::CompilationRequest>(
                           executionConfig.takeError());
  auto toolchain = wafer::compiler::TargetToolchain::create(
      WAFER_PYTHON_EXECUTABLE, WAFER_DEVICE_LINKER_SCRIPT);
  if (!request || !toolchain) {
    if (!request)
      llvm::consumeError(request.takeError());
    if (!toolchain)
      llvm::consumeError(toolchain.takeError());
    fail("worker compiler inputs are invalid");
    return 2;
  }
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  auto product = wafer::compiler::qualification_internal::
      compileProgramWithOptimizationConfiguration(
          std::move(*request), options.outputProgramDirectory,
          WAFER_XLA_SPMD_PARTITIONER_HELPER, *toolchain, proposal,
          *configuration, options.inputVariant,
          options.normalizationRepetitions, llvm::errs());
  if (mlir::failed(product))
    return 1;
  if (!journal->sealScope(scope.scopeKind, scope.scopeDigest, &diagnostic)) {
    fail("worker cannot seal invocation scope: " + diagnostic);
    return 1;
  }
  auto artifactDigest = wafer::digestOptimizationArtifactDirectoryV1(
      options.outputProgramDirectory,
      "wafer.optimization-qualification-output-v1");
  if (!artifactDigest) {
    fail("worker cannot digest published output artifact");
    return 1;
  }

  std::optional<wafer::AdoptionDigest> modelDigest;
  if (options.runTargetModel) {
#if defined(WAFER_ENABLE_SYSTEMC_MODEL) &&                                     \
    defined(WAFER_ENABLE_TARGET_BULK_MODEL)
    constexpr uint64_t largeBudget = 1000000000000000ULL;
    auto backend = wafer::model::ManagedReferenceTargetModelBackend::create(
        wafer::BulkNumericWorkBudget::create(largeBudget, largeBudget,
                                             largeBudget));
    if (!backend) {
      fail("worker cannot create managed model backend: " +
           llvm::toString(backend.takeError()));
      return 1;
    }
    std::vector<wafer::model::TargetModelTensorFileBinding> inputs;
    std::vector<wafer::model::TargetModelTensorFileBinding> expected;
    for (const ModelBinding &binding : options.modelInputs)
      inputs.push_back({binding.index, binding.path});
    for (const ModelBinding &binding : options.modelExpected)
      expected.push_back({binding.index, binding.path});
    auto result = wafer::model::executeAndCompareTargetModelExpectedOutputs(
        product->executableBundle, product->targetLLVMModuleBundle,
        options.outputProgramDirectory, inputs, expected, options.modelAtol,
        options.modelRtol,
        wafer::model::TargetModelKernelBudget::create(
            wafer::FormalNumericWorkBudget::create(largeBudget, largeBudget),
            largeBudget, largeBudget),
        wafer::model::TargetModelExecutionPolicy::managedReference(**backend,
                                                                   **backend),
        &llvm::outs());
    if (!result) {
      fail("worker target model gate failed: " +
           llvm::toString(result.takeError()));
      return 1;
    }
    modelDigest = digestTargetModelResult(*result);
    llvm::outs() << "target-model ranks=" << result->completedRankCount
                 << " transactions=" << result->issuedTransactionCount
                 << " formal=" << result->formalNumericCommandCount
                 << " managed=" << result->managedReferenceNumericCommandCount
                 << " bulk=" << result->bulkNumericCommandCount << "\n";
#else
    fail("worker target model support is not configured");
    return 1;
#endif
  }

  wafer::OptimizationQualificationExecutionResultV1 result;
  result.qualificationRunDigest = options.qualificationRunDigest;
  result.inputSnapshotDigest = options.inputSnapshotDigest;
  result.caseKey = *caseKey;
  result.configuration = *evidenceConfiguration;
  result.requiredTensorNormalizationRepetitions =
      options.normalizationRepetitions;
  result.outputArtifactDigest = *artifactDigest;
  result.staticMetrics = std::move(product->staticMetrics);
  result.targetModelEvidenceDigest = modelDigest;
  result.invocationTerminals = journal->committedTerminals();
  std::sort(result.invocationTerminals.begin(),
            result.invocationTerminals.end(),
            [](const auto &lhs, const auto &rhs) {
              return wafer::digestInvocationIdentityV1(lhs.identity) <
                     wafer::digestInvocationIdentityV1(rhs.identity);
            });
  std::vector<uint8_t> bytes =
      wafer::encodeOptimizationQualificationExecutionResultV1(result);
  if (bytes.empty() || !writeBytes(options.resultFile, bytes))
    return 1;
  llvm::outs() << "isolated execution passed case=" << options.corpusId << ":"
               << options.rankCount
               << " variant=" << static_cast<uint32_t>(options.inputVariant)
               << " terminals=" << result.invocationTerminals.size() << "\n";
  return 0;
}

static std::optional<wafer::AdoptionDigest> digestFile(llvm::StringRef path,
                                                       llvm::StringRef domain) {
  return wafer::digestOptimizationArtifactFileV1(path, domain);
}

static std::optional<std::string> readTextFile(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFileAsStream(path);
  if (!buffer)
    return std::nullopt;
  return (*buffer)->getBuffer().str();
}

static std::optional<wafer::AdoptionDigest> computeHostEnvironmentDigest() {
  std::vector<std::string> facts;
  for (llvm::StringRef path :
       {"/proc/sys/kernel/osrelease", "/proc/self/status"}) {
    std::optional<std::string> contents = readTextFile(path);
    if (!contents)
      return std::nullopt;
    if (path == "/proc/self/status") {
      llvm::StringRef status(*contents);
      llvm::SmallVector<llvm::StringRef, 64> lines;
      status.split(lines, '\n');
      auto line = llvm::find_if(lines, [](llvm::StringRef candidate) {
        return candidate.starts_with("Cpus_allowed_list:");
      });
      if (line == lines.end())
        return std::nullopt;
      facts.push_back(line->str());
    } else {
      facts.push_back(*contents);
    }
  }
  std::optional<std::string> governor =
      readTextFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  if (!governor)
    return std::nullopt;
  facts.push_back(*governor);
  return combineDigests("wafer.optimization-qualification-host-v1", {}, facts);
}

static std::optional<wafer::AdoptionDigest> computeToolchainDigest() {
  std::vector<wafer::AdoptionDigest> digests;
  for (auto [path, domain] :
       {std::pair<llvm::StringRef, llvm::StringRef>{
            WAFER_PYTHON_EXECUTABLE,
            "wafer.optimization-qualification-python-v1"},
        {WAFER_DEVICE_LINKER_SCRIPT,
         "wafer.optimization-qualification-linker-script-v1"},
        {WAFER_XLA_SPMD_PARTITIONER_HELPER,
         "wafer.optimization-qualification-spmd-helper-v1"}}) {
    std::optional<wafer::AdoptionDigest> digest = digestFile(path, domain);
    if (!digest)
      return std::nullopt;
    digests.push_back(*digest);
  }
  return combineDigests("wafer.optimization-qualification-toolchain-v1",
                        digests);
}

static std::optional<wafer::AdoptionDigest>
computeCorpusDigest(llvm::ArrayRef<QualificationCasePaths> cases) {
  std::vector<wafer::AdoptionDigest> digests;
  std::set<std::string> seenPrograms;
  std::set<std::string> seenFiles;
  for (const QualificationCasePaths &item : cases) {
    if (seenPrograms.insert(item.programDirectory).second) {
      auto digest = wafer::digestOptimizationArtifactDirectoryV1(
          item.programDirectory,
          "wafer.optimization-qualification-corpus-program-v1");
      if (!digest)
        return std::nullopt;
      digests.push_back(*digest);
    }
    for (const ModelBinding *binding :
         {&item.modelInputs.front(), &item.modelExpected.front()})
      if (seenFiles.insert(binding->path).second) {
        auto digest =
            digestFile(binding->path,
                       "wafer.optimization-qualification-corpus-reference-v1");
        if (!digest)
          return std::nullopt;
        digests.push_back(*digest);
      }
  }
  return combineDigests("wafer.optimization-qualification-corpus-v1", digests);
}

static wafer::AdoptionDigest
computeInputSnapshotDigest(const QualificationCasePaths &paths,
                           wafer::EquivalentInputVariantV1 variant) {
  auto program = wafer::digestOptimizationArtifactDirectoryV1(
      paths.programDirectory,
      "wafer.optimization-qualification-input-program-v1");
  std::vector<wafer::AdoptionDigest> digests;
  if (program)
    digests.push_back(*program);
  std::vector<std::string> facts = {
      std::to_string(paths.originalCase.corpus.id),
      std::to_string(paths.originalCase.rankCount),
      std::to_string(static_cast<uint32_t>(variant))};
  return combineDigests("wafer.optimization-qualification-input-case-v1",
                        digests, facts);
}

static std::vector<ExecutionPlan>
buildExecutionPlan(llvm::ArrayRef<QualificationCasePaths> cases) {
  std::vector<ExecutionPlan> plan;
  uint64_t ordinal = 0;
  for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
    for (bool cleanupEnabled : {false, true})
      for (wafer::EquivalentInputVariantV1 variant :
           {wafer::EquivalentInputVariantV1::Original,
            wafer::EquivalentInputVariantV1::Metamorphic}) {
        wafer::QualificationCaseKeyV1 key = cases[caseIndex].originalCase;
        key.inputVariant = variant;
        plan.push_back({ExecutionPurpose::Equivalent, caseIndex, key,
                        equivalentEvidenceConfiguration(cleanupEnabled), 1,
                        false, ordinal++});
      }
    plan.push_back({ExecutionPurpose::NormalizationOnce, caseIndex,
                    cases[caseIndex].originalCase, allOnEvidenceConfiguration(),
                    1, false, ordinal++});
    plan.push_back({ExecutionPurpose::NormalizationTwice, caseIndex,
                    cases[caseIndex].originalCase, allOnEvidenceConfiguration(),
                    2, false, ordinal++});
    plan.push_back({ExecutionPurpose::Production, caseIndex,
                    cases[caseIndex].originalCase, allOnEvidenceConfiguration(),
                    1, true, ordinal++});
  }
  return plan;
}

static llvm::SmallString<256> child(llvm::StringRef parent,
                                    llvm::StringRef name) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, name);
  return path;
}

static bool
runIsolatedExecution(llvm::StringRef executable, llvm::StringRef workRoot,
                     const QualificationCasePaths &paths,
                     const ExecutionPlan &plan,
                     const wafer::AdoptionDigest &runDigest,
                     const wafer::AdoptionDigest &inputSnapshotDigest,
                     CompletedExecution &completed) {
  llvm::SmallString<256> executionDirectory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          child(workRoot, "execution"), executionDirectory))
    return fail("cannot create isolated execution directory: " +
                error.message());
  llvm::SmallString<256> outputDirectory =
      child(executionDirectory, "compiled-program");
  llvm::SmallString<256> resultFile = child(executionDirectory, "result.bin");
  llvm::SmallString<256> stdoutFile = child(executionDirectory, "stdout.log");
  llvm::SmallString<256> stderrFile = child(executionDirectory, "stderr.log");

  std::vector<std::string> storage = {
      executable.str(),
      "--worker",
      "--input-program-dir",
      paths.programDirectory,
      "--output-program-dir",
      outputDirectory.str().str(),
      "--result-file",
      resultFile.str().str(),
      "--execution-ranks",
      std::to_string(plan.caseKey.rankCount),
      "--corpus-id",
      std::to_string(plan.caseKey.corpus.id),
      "--execution-ordinal",
      std::to_string(plan.executionOrdinal),
      "--normalization-repetitions",
      std::to_string(plan.normalizationRepetitions),
      "--configuration",
      configurationSpelling(plan.configuration),
      "--input-variant",
      plan.caseKey.inputVariant == wafer::EquivalentInputVariantV1::Original
          ? "original"
          : "metamorphic",
      "--qualification-run-digest",
      wafer::toHex(runDigest),
      "--input-snapshot-digest",
      wafer::toHex(inputSnapshotDigest),
  };
  if (plan.runTargetModel) {
    storage.push_back("--run-target-model");
    for (const ModelBinding &binding : paths.modelInputs) {
      storage.push_back("--model-input");
      storage.push_back(std::to_string(binding.index) + "=" + binding.path);
    }
    for (const ModelBinding &binding : paths.modelExpected) {
      storage.push_back("--model-expected");
      storage.push_back(std::to_string(binding.index) + "=" + binding.path);
    }
    storage.push_back("--model-atol");
    storage.push_back(std::to_string(paths.atol));
    storage.push_back("--model-rtol");
    storage.push_back(std::to_string(paths.rtol));
  }
  std::vector<llvm::StringRef> arguments;
  arguments.reserve(storage.size());
  for (const std::string &argument : storage)
    arguments.push_back(argument);
  std::optional<llvm::sys::ProcessStatistics> statistics;
  std::string error;
  bool executionFailed = false;
  std::array<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(""), stdoutFile.str(), stderrFile.str()};
  auto begin = std::chrono::steady_clock::now();
  int exitCode =
      llvm::sys::ExecuteAndWait(executable, arguments, std::nullopt, redirects,
                                0, 0, &error, &executionFailed, &statistics);
  auto elapsed = std::chrono::steady_clock::now() - begin;
  if (executionFailed || exitCode != 0) {
    std::optional<std::string> stderrText = readTextFile(stderrFile);
    return fail("isolated execution failed ordinal=" +
                llvm::Twine(plan.executionOrdinal) + " exit=" +
                llvm::Twine(exitCode) + " error=" + error + " stderr=" +
                (stderrText ? *stderrText : std::string("<unavailable>")));
  }
  std::vector<uint8_t> bytes;
  std::string diagnostic;
  wafer::OptimizationQualificationExecutionResultV1 result;
  if (!readBytes(resultFile, bytes) ||
      !wafer::decodeCanonicalOptimizationQualificationExecutionResultV1(
          bytes, result, &diagnostic))
    return fail("isolated execution result rejected: " + diagnostic);
  if (result.qualificationRunDigest != runDigest ||
      result.inputSnapshotDigest != inputSnapshotDigest ||
      result.requiredTensorNormalizationRepetitions !=
          plan.normalizationRepetitions ||
      result.caseKey.corpus.id != plan.caseKey.corpus.id ||
      result.caseKey.rankCount != plan.caseKey.rankCount ||
      result.caseKey.inputVariant != plan.caseKey.inputVariant ||
      result.configuration.fixedSelection !=
          plan.configuration.fixedSelection ||
      result.configuration.cleanupSelection !=
          plan.configuration.cleanupSelection ||
      result.targetModelEvidenceDigest.has_value() != plan.runTargetModel)
    return fail("isolated execution result disagrees with its launch plan");
  completed.plan = plan;
  completed.result = std::move(result);
  completed.resultDigest =
      wafer::digestOptimizationQualificationExecutionResultV1(completed.result);
  completed.wallNs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  completed.peakRssBytes = statistics ? statistics->PeakMemory * 1024 : 0;
  llvm::outs() << "completed isolated execution ordinal="
               << plan.executionOrdinal << " case=" << plan.caseKey.corpus.id
               << ":" << plan.caseKey.rankCount << " variant="
               << static_cast<uint32_t>(plan.caseKey.inputVariant)
               << " purpose=" << static_cast<uint32_t>(plan.purpose)
               << " wall_ns=" << completed.wallNs
               << " peak_rss_bytes=" << completed.peakRssBytes << "\n";
  llvm::outs().flush();
  return true;
}

static bool sameStaticMetrics(const wafer::ExactStaticVectorEvidenceV1 &lhs,
                              const wafer::ExactStaticVectorEvidenceV1 &rhs) {
  if (lhs.registrySchema != rhs.registrySchema ||
      lhs.registryDigest != rhs.registryDigest ||
      lhs.orderedMetrics.size() != rhs.orderedMetrics.size())
    return false;
  for (size_t index = 0; index < lhs.orderedMetrics.size(); ++index)
    if (lhs.orderedMetrics[index].metricId !=
            rhs.orderedMetrics[index].metricId ||
        lhs.orderedMetrics[index].value != rhs.orderedMetrics[index].value)
      return false;
  return true;
}

static const CompletedExecution *findExecution(
    llvm::ArrayRef<CompletedExecution> executions, size_t caseIndex,
    ExecutionPurpose purpose,
    std::optional<wafer::EquivalentInputVariantV1> variant = std::nullopt,
    std::optional<bool> cleanupEnabled = std::nullopt) {
  auto found = llvm::find_if(executions, [&](const CompletedExecution &item) {
    if (item.plan.caseIndex != caseIndex || item.plan.purpose != purpose ||
        (variant && item.plan.caseKey.inputVariant != *variant))
      return false;
    if (!cleanupEnabled)
      return true;
    bool actual = item.plan.configuration.cleanupSelection ==
                  wafer::OptimizationGroupSelectionV1::AllOn;
    return actual == *cleanupEnabled;
  });
  return found == executions.end() ? nullptr : &*found;
}

static bool
verifyExecutionGates(llvm::ArrayRef<QualificationCasePaths> cases,
                     llvm::ArrayRef<CompletedExecution> executions) {
  for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
    for (bool cleanupEnabled : {false, true}) {
      const CompletedExecution *original = findExecution(
          executions, caseIndex, ExecutionPurpose::Equivalent,
          wafer::EquivalentInputVariantV1::Original, cleanupEnabled);
      const CompletedExecution *metamorphic = findExecution(
          executions, caseIndex, ExecutionPurpose::Equivalent,
          wafer::EquivalentInputVariantV1::Metamorphic, cleanupEnabled);
      if (!original || !metamorphic ||
          original->result.outputArtifactDigest !=
              metamorphic->result.outputArtifactDigest ||
          !sameStaticMetrics(original->result.staticMetrics,
                             metamorphic->result.staticMetrics))
        return fail("Equivalent-IR output/static gate failed for case " +
                    llvm::Twine(cases[caseIndex].originalCase.corpus.id) + ":" +
                    llvm::Twine(cases[caseIndex].originalCase.rankCount));
    }
    const CompletedExecution *once = findExecution(
        executions, caseIndex, ExecutionPurpose::NormalizationOnce);
    const CompletedExecution *twice = findExecution(
        executions, caseIndex, ExecutionPurpose::NormalizationTwice);
    const CompletedExecution *production =
        findExecution(executions, caseIndex, ExecutionPurpose::Production);
    if (!once || !twice || !production ||
        once->result.outputArtifactDigest !=
            twice->result.outputArtifactDigest ||
        !sameStaticMetrics(once->result.staticMetrics,
                           twice->result.staticMetrics) ||
        !production->result.targetModelEvidenceDigest)
      return fail("normalizer/production gate failed for mandatory case");
    bool secondNoChange = llvm::any_of(
        twice->result.invocationTerminals,
        [](const wafer::InvocationTelemetryV1 &terminal) {
          return terminal.identity.mechanismKey ==
                     wafer::mechanism::RequiredTensorNormalization &&
                 terminal.identity.invocationOrdinal %
                         wafer::kOptimizationInvocationLocalOrdinalLimit ==
                     1 &&
                 terminal.outcome == wafer::InvocationOutcome::NoChange &&
                 terminal.rewriteCount == 0;
        });
    if (!secondNoChange)
      return fail("second required tensor normalization was not a no-change");
  }
  return true;
}

static bool gateLess(const wafer::GateEvidenceV1 &lhs,
                     const wafer::GateEvidenceV1 &rhs) {
  auto optional = [](const std::optional<wafer::MechanismKey> &value) {
    return std::pair<bool, uint32_t>{value.has_value(),
                                     value ? value->semanticId : 0};
  };
  return std::make_tuple(
             lhs.gate.id, lhs.gate.registrySchema, lhs.gate.registryDigest,
             lhs.caseKey.corpus.id, lhs.caseKey.corpus.registrySchema,
             lhs.caseKey.corpus.registryDigest, lhs.caseKey.rankCount,
             lhs.caseKey.inputVariant, lhs.configuration.fixedSelection,
             optional(lhs.configuration.fixedDisabledKey),
             lhs.configuration.cleanupSelection,
             optional(lhs.configuration.cleanupDisabledKey)) <
         std::make_tuple(
             rhs.gate.id, rhs.gate.registrySchema, rhs.gate.registryDigest,
             rhs.caseKey.corpus.id, rhs.caseKey.corpus.registrySchema,
             rhs.caseKey.corpus.registryDigest, rhs.caseKey.rankCount,
             rhs.caseKey.inputVariant, rhs.configuration.fixedSelection,
             optional(rhs.configuration.fixedDisabledKey),
             rhs.configuration.cleanupSelection,
             optional(rhs.configuration.cleanupDisabledKey));
}

static wafer::GateEvidenceBundleV1
buildEquivalentGateEvidence(llvm::ArrayRef<QualificationCasePaths> cases,
                            llvm::ArrayRef<CompletedExecution> executions) {
  wafer::GateEvidenceBundleV1 bundle;
  for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex)
    for (bool cleanupEnabled : {false, true}) {
      const CompletedExecution *original = findExecution(
          executions, caseIndex, ExecutionPurpose::Equivalent,
          wafer::EquivalentInputVariantV1::Original, cleanupEnabled);
      const CompletedExecution *metamorphic = findExecution(
          executions, caseIndex, ExecutionPurpose::Equivalent,
          wafer::EquivalentInputVariantV1::Metamorphic, cleanupEnabled);
      wafer::AdoptionDigest pairDigest =
          combineDigests("wafer.optimization-qualification-equivalent-pair-v1",
                         {original->resultDigest, metamorphic->resultDigest});
      for (wafer::EquivalentInputVariantV1 variant :
           {wafer::EquivalentInputVariantV1::Original,
            wafer::EquivalentInputVariantV1::Metamorphic}) {
        wafer::QualificationCaseKeyV1 key = cases[caseIndex].originalCase;
        key.inputVariant = variant;
        bundle.orderedGateResults.push_back(
            {wafer::getCurrentEquivalentIRGateRefV1(), key,
             equivalentEvidenceConfiguration(cleanupEnabled),
             wafer::GateStatusV1::Passed,
             combineDigests(
                 "wafer.optimization-qualification-equivalent-row-v1",
                 {pairDigest},
                 {std::to_string(static_cast<uint32_t>(variant)),
                  std::to_string(cleanupEnabled)}),
             std::nullopt});
      }
    }
  std::sort(bundle.orderedGateResults.begin(), bundle.orderedGateResults.end(),
            gateLess);
  return bundle;
}

static wafer::GateEvidenceBundleV1
buildProductionGateEvidence(llvm::ArrayRef<QualificationCasePaths> cases,
                            llvm::ArrayRef<CompletedExecution> executions) {
  wafer::GateEvidenceBundleV1 bundle;
  for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
    const CompletedExecution *production =
        findExecution(executions, caseIndex, ExecutionPurpose::Production);
    bundle.orderedGateResults.push_back(
        {wafer::getCurrentProductionAllOnGateRefV1(),
         cases[caseIndex].originalCase, allOnEvidenceConfiguration(),
         wafer::GateStatusV1::Passed, production->resultDigest, std::nullopt});
  }
  std::sort(bundle.orderedGateResults.begin(), bundle.orderedGateResults.end(),
            gateLess);
  return bundle;
}

struct InvocationGroupKey {
  wafer::RegistryRefV1 site;
  wafer::OptimizationCutPoint cut;
  wafer::QualificationCaseKeyV1 caseKey;
};

static bool operator<(const InvocationGroupKey &lhs,
                      const InvocationGroupKey &rhs) {
  return std::tie(lhs.site.id, lhs.site.registrySchema, lhs.site.registryDigest,
                  lhs.cut, lhs.caseKey.corpus.id,
                  lhs.caseKey.corpus.registrySchema,
                  lhs.caseKey.corpus.registryDigest, lhs.caseKey.rankCount,
                  lhs.caseKey.inputVariant) <
         std::tie(rhs.site.id, rhs.site.registrySchema, rhs.site.registryDigest,
                  rhs.cut, rhs.caseKey.corpus.id,
                  rhs.caseKey.corpus.registrySchema,
                  rhs.caseKey.corpus.registryDigest, rhs.caseKey.rankCount,
                  rhs.caseKey.inputVariant);
}

static std::vector<wafer::QualificationObservationV1>
buildObservations(const wafer::AdoptionQualificationRunV1 &run,
                  const wafer::AdoptionDigest &runDigest,
                  const wafer::AdoptionDigest &attemptDigest,
                  llvm::ArrayRef<wafer::InvocationTelemetryV1> terminals,
                  const wafer::GateEvidenceBundleV1 &productionGates) {
  std::vector<wafer::QualificationObservationV1> observations;
  wafer::AdoptionDigest proposalDigest =
      wafer::digestOptimizationQualificationProposalV1(
          wafer::getCurrentOptimizationQualificationProposal());
  for (const wafer::AdoptionSpec &spec : wafer::getAllAdoptionSpecs()) {
    wafer::QualificationObservationV1 observation;
    observation.mechanismKey = spec.mechanismKey;
    observation.specDigest = wafer::digestAdoptionSpecV1(spec);
    observation.qualificationIdentity = run.qualificationIdentity;
    observation.qualificationPolicy = run.qualificationPolicy;
    observation.compileWorkSummary.workPolicyDigest =
        wafer::digestAdoptionWorkPolicyV1(spec.workPolicyKind);
    observation.qualificationRunDigest = runDigest;
    bool optionalOptimization =
        spec.adoptionMode == wafer::AdoptionMode::FixedOptimization ||
        spec.adoptionMode == wafer::AdoptionMode::BestEffortCleanup;
    if (optionalOptimization) {
      observation.optimizationProposalDigest = proposalDigest;
      observation.optimizationPublicationAttemptDigest = attemptDigest;
    }

    std::vector<const wafer::InvocationTelemetryV1 *> selected;
    for (const wafer::InvocationTelemetryV1 &terminal : terminals)
      if (terminal.identity.mechanismKey == spec.mechanismKey)
        selected.push_back(&terminal);
    std::map<std::pair<wafer::InvocationOutcome, uint32_t>, uint64_t>
        outcomeCounts;
    std::map<InvocationGroupKey, wafer::InvocationEvidenceV1> groups;
    std::map<uint32_t, uint64_t> work;
    uint64_t rewriteTotal = 0;
    uint64_t actionTotal = 0;
    bool actionsSucceeded = true;
    for (const wafer::InvocationTelemetryV1 *terminal : selected) {
      uint32_t reasonId =
          terminal->terminalReason ? terminal->terminalReason->reason.id : 0;
      ++outcomeCounts[{terminal->outcome, reasonId}];
      InvocationGroupKey key{terminal->identity.invocationSite,
                             terminal->identity.cutPoint,
                             *terminal->qualificationCase};
      wafer::InvocationEvidenceV1 &group = groups[key];
      group.invocationSite = key.site;
      group.cutPoint = key.cut;
      group.caseKey = key.caseKey;
      ++group.invocationCount;
      group.rewriteCount += terminal->rewriteCount;
      group.backendActions.insert(group.backendActions.end(),
                                  terminal->backendActions.begin(),
                                  terminal->backendActions.end());
      rewriteTotal += terminal->rewriteCount;
      for (const wafer::BackendActionEvidenceV1 &action :
           terminal->backendActions) {
        ++actionTotal;
        actionsSucceeded &=
            action.terminalStatus == wafer::BackendActionStatusV1::Success;
      }
      for (const wafer::WorkCounterV1 &counter :
           terminal->workSummary.orderedCounters)
        work[counter.counterId] += counter.value;
    }
    for (const auto &[key, count] : outcomeCounts) {
      std::optional<wafer::ClosedReasonV1> reason;
      if (key.second != 0) {
        auto terminal = llvm::find_if(selected, [&](const auto *candidate) {
          return candidate->outcome == key.first && candidate->terminalReason &&
                 candidate->terminalReason->reason.id == key.second;
        });
        reason = (*terminal)->terminalReason;
      }
      observation.outcomeCounts.push_back({key.first, reason, count});
    }
    for (auto &[key, group] : groups) {
      std::sort(group.backendActions.begin(), group.backendActions.end(),
                [](const auto &lhs, const auto &rhs) {
                  return std::tie(lhs.invocationId, lhs.actionOrdinal) <
                         std::tie(rhs.invocationId, rhs.actionOrdinal);
                });
      observation.invocationEvidence.push_back(std::move(group));
    }
    for (const auto &[counterId, value] : work)
      observation.compileWorkSummary.orderedCounters.push_back(
          {counterId, value});

    bool eligible = spec.availability == wafer::Availability::Resolved &&
                    spec.adoptionMode != wafer::AdoptionMode::None &&
                    !selected.empty();
    bool evidenceComplete = false;
    switch (spec.evidenceKind) {
    case wafer::InvocationEvidenceKind::Rewrite:
      evidenceComplete = rewriteTotal != 0;
      break;
    case wafer::InvocationEvidenceKind::BackendAction:
      evidenceComplete = actionTotal != 0 && actionsSucceeded;
      break;
    case wafer::InvocationEvidenceKind::InvocationOnly:
      evidenceComplete = true;
      break;
    }
    if (eligible) {
      observation.mechanismGateResults = productionGates;
      if (evidenceComplete) {
        observation.qualificationStatus =
            wafer::QualificationStatusV1::Qualified;
      } else if (spec.evidenceKind == wafer::InvocationEvidenceKind::Rewrite &&
                 rewriteTotal == 0) {
        observation.qualificationStatus =
            wafer::QualificationStatusV1::NoOpObserved;
        observation.closedReason = wafer::ClosedReasonV1{
            wafer::getGlobalClosedReasonRefV1(
                wafer::GlobalClosedReasonV1::NoDeterministicBenefit),
            std::nullopt};
      } else {
        observation.qualificationStatus =
            wafer::QualificationStatusV1::Rejected;
        observation.closedReason = wafer::ClosedReasonV1{
            wafer::getGlobalClosedReasonRefV1(
                wafer::GlobalClosedReasonV1::QualificationEvidenceRejected),
            std::nullopt};
      }
    } else {
      observation.qualificationStatus = wafer::QualificationStatusV1::Rejected;
      observation.closedReason = wafer::ClosedReasonV1{
          wafer::getGlobalClosedReasonRefV1(
              wafer::GlobalClosedReasonV1::QualificationEvidenceRejected),
          std::nullopt};
    }
    observations.push_back(std::move(observation));
  }
  return observations;
}

static int runCoordinator(llvm::StringRef executable, const Options &options) {
#if !defined(WAFER_ENABLE_SYSTEMC_MODEL) ||                                    \
    !defined(WAFER_ENABLE_TARGET_BULK_MODEL)
  fail("qualification requires a SystemC and managed bulk enabled build");
  return 1;
#else
  if (std::error_code error =
          llvm::sys::fs::create_directories(options.workDirectory)) {
    fail("cannot create qualification work directory: " + error.message());
    return 1;
  }
  std::vector<wafer::QualificationCaseKeyV1> mandatory =
      wafer::getCurrentMandatoryQualificationCasesV1();
  std::vector<QualificationCasePaths> cases = {
      {mandatory[0],
       options.sourceProgramDirectory,
       {{0, options.sourceInput}},
       {{0, options.sourceExpected}},
       options.sourceAtol,
       options.sourceRtol},
      {mandatory[1],
       options.sourceProgramDirectory,
       {{0, options.sourceInput}},
       {{0, options.sourceExpected}},
       options.sourceAtol,
       options.sourceRtol},
      {mandatory[2],
       options.llamaProgramDirectory,
       {{0, options.llamaInput}},
       {{0, options.llamaExpected}},
       options.llamaAtol,
       options.llamaRtol},
  };

  auto hostBefore = computeHostEnvironmentDigest();
  auto toolchain = computeToolchainDigest();
  auto corpus = computeCorpusDigest(cases);
  auto build =
      digestFile(executable, "wafer.optimization-qualification-build-v1");
  std::vector<std::string> featureFacts = {
      "systemc=1", "managed-reference=1",
      "target=wafer-tx81-single-card-kernel-v1"};
  wafer::AdoptionDigest feature = combineDigests(
      "wafer.optimization-qualification-features-v1", {}, featureFacts);
  if (!hostBefore || !toolchain || !corpus || !build) {
    fail("cannot construct qualification identity from actual environment: " +
         llvm::Twine(!hostBefore ? "host " : "") +
         llvm::Twine(!toolchain ? "toolchain " : "") +
         llvm::Twine(!corpus ? "corpus " : "") +
         llvm::Twine(!build ? "build" : ""));
    return 1;
  }

  wafer::OptimizationQualificationArchiveStoreV1 store(options.archiveRoot);
  std::string diagnostic;
  std::optional<uint64_t> runSeries =
      store.allocateRunSeriesOrdinal(&diagnostic);
  if (!runSeries) {
    fail("cannot allocate qualification run series: " + diagnostic);
    return 1;
  }
  wafer::CompletedQualificationEvidenceV1 chain;
  for (const wafer::AdoptionSpec &spec : wafer::getAllAdoptionSpecs())
    chain.input.specBindings.push_back(
        {spec.mechanismKey, wafer::digestAdoptionSpecV1(spec)});
  std::vector<std::array<wafer::AdoptionDigest, 2>> snapshots;
  for (const QualificationCasePaths &paths : cases) {
    std::array<wafer::AdoptionDigest, 2> pair = {
        computeInputSnapshotDigest(paths,
                                   wafer::EquivalentInputVariantV1::Original),
        computeInputSnapshotDigest(
            paths, wafer::EquivalentInputVariantV1::Metamorphic)};
    snapshots.push_back(pair);
    for (size_t variant = 0; variant < pair.size(); ++variant) {
      wafer::QualificationCaseKeyV1 key = paths.originalCase;
      key.inputVariant = static_cast<wafer::EquivalentInputVariantV1>(variant);
      chain.input.qualificationCases.push_back({key, pair[variant]});
    }
  }
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  chain.input.optimizationProposalDigest =
      wafer::digestOptimizationQualificationProposalV1(proposal);
  chain.run.qualificationInputDigest =
      wafer::digestAdoptionQualificationInputV1(chain.input);
  chain.run.qualificationIdentity = {*build, *toolchain, *hostBefore, *corpus,
                                     feature};
  chain.run.qualificationPolicy =
      wafer::getCurrentOptimizationSetQualificationPolicyRefV1();
  chain.run.runSeriesOrdinal = *runSeries;
  chain.run.attemptOrdinal = 0;
  wafer::AdoptionDigest runDigest =
      wafer::digestAdoptionQualificationRunV1(chain.run);
  chain.publicationAttempt.emplace();
  chain.publicationAttempt->qualificationRunDigest = runDigest;
  chain.publicationAttempt->proposalDigest =
      *chain.input.optimizationProposalDigest;
  llvm::SmallString<256> activePath(options.archiveRoot);
  llvm::sys::path::append(activePath, "active-ref.bin");
  if (llvm::sys::fs::exists(activePath)) {
    wafer::ActiveQualifiedOptimizationSelectionV1 active;
    if (!store.loadActiveQualifiedOptimizationSet(active, &diagnostic)) {
      fail("existing active qualification reference is invalid: " + diagnostic);
      return 1;
    }
    chain.publicationAttempt->expectedActiveRefDigest =
        wafer::digestActiveQualifiedOptimizationSetRefV1(active.activeRef);
  }
  wafer::AdoptionDigest attemptDigest =
      wafer::digestOptimizationSetPublicationAttemptV1(
          *chain.publicationAttempt);

  std::vector<ExecutionPlan> plan = buildExecutionPlan(cases);
  if (plan.size() > wafer::getCurrentOptimizationSetQualificationPolicyV1()
                        .totalProcessLaunchCap) {
    fail("qualification execution plan exceeds process launch policy");
    return 1;
  }
  std::vector<CompletedExecution> executions;
  executions.reserve(plan.size());
  for (const ExecutionPlan &item : plan) {
    size_t variant = static_cast<size_t>(item.caseKey.inputVariant);
    CompletedExecution completed;
    if (!runIsolatedExecution(executable, options.workDirectory,
                              cases[item.caseIndex], item, runDigest,
                              snapshots[item.caseIndex][variant], completed))
      return 1;
    executions.push_back(std::move(completed));
  }
  if (!verifyExecutionGates(cases, executions))
    return 1;
  auto hostAfter = computeHostEnvironmentDigest();
  if (!hostAfter || *hostAfter != *hostBefore) {
    fail("host kernel/affinity/governor identity changed during run");
    return 1;
  }

  wafer::GateEvidenceBundleV1 equivalentGates =
      buildEquivalentGateEvidence(cases, executions);
  wafer::GateEvidenceBundleV1 productionGates =
      buildProductionGateEvidence(cases, executions);
  for (const CompletedExecution &execution : executions)
    chain.invocationTerminals.insert(
        chain.invocationTerminals.end(),
        execution.result.invocationTerminals.begin(),
        execution.result.invocationTerminals.end());
  std::sort(chain.invocationTerminals.begin(), chain.invocationTerminals.end(),
            [](const auto &lhs, const auto &rhs) {
              return wafer::digestInvocationIdentityV1(lhs.identity) <
                     wafer::digestInvocationIdentityV1(rhs.identity);
            });
  chain.observations =
      buildObservations(chain.run, runDigest, attemptDigest,
                        chain.invocationTerminals, productionGates);
  chain.optimizationBatch.emplace();
  chain.optimizationBatch->proposalDigest =
      *chain.input.optimizationProposalDigest;
  chain.optimizationBatch->qualificationIdentity =
      chain.run.qualificationIdentity;
  chain.optimizationBatch->qualificationPolicy = chain.run.qualificationPolicy;
  chain.optimizationBatch->equivalentIR2x2GateResults =
      std::move(equivalentGates);
  chain.optimizationBatch->productionAllOnGateResults =
      std::move(productionGates);
  chain.optimizationBatch->batchStatus =
      wafer::OptimizationBatchStatusV1::Qualified;
  chain.optimizationBatch->qualificationRunDigest = runDigest;
  chain.optimizationBatch->optimizationPublicationAttemptDigest = attemptDigest;

  chain.resultManifest.qualificationRunDigest = runDigest;
  for (const wafer::InvocationTelemetryV1 &terminal : chain.invocationTerminals)
    chain.resultManifest.invocationTerminals.push_back(
        {wafer::digestInvocationIdentityV1(terminal.identity),
         wafer::digestInvocationTelemetryV1(terminal)});
  for (const wafer::QualificationObservationV1 &observation :
       chain.observations)
    chain.resultManifest.observationBindings.push_back(
        {observation.mechanismKey,
         wafer::digestQualificationObservationV1(observation)});
  chain.resultManifest.optimizationBatchObservationDigest =
      wafer::digestOptimizationBatchObservationV1(*chain.optimizationBatch);
  chain.runTerminal.qualificationRunDigest = runDigest;
  chain.runTerminal.outcome =
      wafer::AdoptionQualificationRunOutcomeV1::CompletedEvidence;
  chain.runTerminal.resultManifestDigest =
      wafer::digestAdoptionQualificationResultManifestV1(chain.resultManifest);
  if (!wafer::validateCompletedQualificationEvidenceV1(chain, &diagnostic)) {
    fail("completed qualification evidence rejected: " + diagnostic);
    return 1;
  }
  if (!store.publishCompletedRun(chain, &diagnostic)) {
    fail("cannot publish completed qualification run: " + diagnostic);
    return 1;
  }
  wafer::QualifiedOptimizationSetV1 qualifiedSet;
  qualifiedSet.proposalDigest = *chain.input.optimizationProposalDigest;
  qualifiedSet.batchObservationDigest =
      *chain.resultManifest.optimizationBatchObservationDigest;
  wafer::OptimizationSetPublicationResultV1 publication;
  if (!store.publishOptimizationSet(runDigest, qualifiedSet, publication,
                                    &diagnostic) ||
      publication.terminal.outcome !=
          wafer::OptimizationSetPublicationOutcomeV1::QualifiedPublished) {
    fail("cannot publish qualified optimization set: " + diagnostic);
    return 1;
  }
  wafer::ActiveQualifiedOptimizationSelectionV1 readback;
  if (!store.loadActiveQualifiedOptimizationSet(readback, &diagnostic) ||
      readback.activeRef.qualificationRunDigest != runDigest ||
      readback.activeRef.setDigest !=
          wafer::digestQualifiedOptimizationSetV1(qualifiedSet)) {
    fail("active qualified set readback failed: " + diagnostic);
    return 1;
  }
  llvm::outs() << "qualification published run=" << wafer::toHex(runDigest)
               << " set=" << wafer::toHex(readback.activeRef.setDigest)
               << " executions=" << executions.size()
               << " terminals=" << chain.invocationTerminals.size()
               << " generation=" << readback.activeRef.generation << "\n";
  return 0;
#endif
}

} // namespace

#ifdef WAFER_ENABLE_SYSTEMC_MODEL
extern "C" int sc_main(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
  Options options;
  if (!parseOptions(argc, argv, options)) {
    printUsage();
    return 2;
  }
  if (options.worker)
    return runWorker(options);
  llvm::SmallString<256> executable;
  if (std::error_code error = llvm::sys::fs::real_path(argv[0], executable)) {
    fail("cannot resolve qualification executable: " + error.message());
    return 1;
  }
  return runCoordinator(executable, options);
}
