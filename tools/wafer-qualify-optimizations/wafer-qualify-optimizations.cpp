//===- wafer-qualify-optimizations.cpp - Optimization qualification -----===//

#include "Compiler/OptimizationQualificationCompiler.h"

#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Support/OptimizationArtifactDigest.h"
#include "Wafer/Support/OptimizationInvocation.h"
#include "Wafer/Support/OptimizationQualification.h"
#include "Wafer/Target/TargetProfile.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
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

struct Options {
  std::string inputProgramDirectory;
  std::string outputProgramDirectory;
  uint32_t rankCount = 0;
  uint32_t corpusId = 0;
  uint64_t executionOrdinal = 0;
  std::string configurationSpelling = "all-on";
  wafer::EquivalentInputVariantV1 inputVariant =
      wafer::EquivalentInputVariantV1::Original;
};

static bool parseU64(llvm::StringRef text, uint64_t &value) {
  return !text.empty() && !text.getAsInteger(10, value);
}

static bool parseOptions(int argc, char **argv, Options &options) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef argument(argv[index]);
    auto take = [&](llvm::StringRef name) -> std::optional<llvm::StringRef> {
      if (argument != name || index + 1 == argc)
        return std::nullopt;
      return llvm::StringRef(argv[++index]);
    };
    if (auto value = take("--input-program-dir"))
      options.inputProgramDirectory = value->str();
    else if (auto value = take("--output-program-dir"))
      options.outputProgramDirectory = value->str();
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
    } else if (auto value = take("--configuration")) {
      options.configurationSpelling = value->str();
    } else if (auto value = take("--input-variant")) {
      if (*value == "original")
        options.inputVariant = wafer::EquivalentInputVariantV1::Original;
      else if (*value == "metamorphic")
        options.inputVariant = wafer::EquivalentInputVariantV1::Metamorphic;
      else
        return false;
    } else {
      return false;
    }
  }
  return !options.inputProgramDirectory.empty() &&
         !options.outputProgramDirectory.empty() && options.rankCount != 0 &&
         options.corpusId != 0;
}

static std::optional<wafer::OptimizationConfiguration>
parseConfiguration(const wafer::OptimizationQualificationProposal &proposal,
                   llvm::StringRef spelling, std::string &diagnostic) {
  if (spelling == "all-on")
    return wafer::getAllOnOptimizationConfiguration();
  if (spelling == "all-off")
    return wafer::getAllOffOptimizationConfiguration();
  constexpr llvm::StringLiteral prefix = "disable-one:";
  if (!spelling.starts_with(prefix)) {
    diagnostic = "configuration must be all-on, all-off, or disable-one:<key>";
    return std::nullopt;
  }
  uint64_t key = 0;
  if (!parseU64(spelling.drop_front(prefix.size()), key) || key > UINT32_MAX) {
    diagnostic = "disable-one mechanism key is invalid";
    return std::nullopt;
  }
  wafer::OptimizationConfiguration configuration =
      wafer::getDisableOneOptimizationConfiguration(
          proposal, wafer::MechanismKey{static_cast<uint32_t>(key)},
          &diagnostic);
  if (!wafer::validateOptimizationConfiguration(proposal, configuration,
                                                &diagnostic))
    return std::nullopt;
  return configuration;
}

static const char *outcomeName(wafer::InvocationOutcome outcome) {
  switch (outcome) {
  case wafer::InvocationOutcome::Applied:
    return "applied";
  case wafer::InvocationOutcome::NoChange:
    return "no-change";
  case wafer::InvocationOutcome::NotApplicable:
    return "not-applicable";
  case wafer::InvocationOutcome::Unsupported:
    return "unsupported";
  case wafer::InvocationOutcome::ResourceExhausted:
    return "resource-exhausted";
  case wafer::InvocationOutcome::Invalid:
    return "invalid";
  case wafer::InvocationOutcome::Cancelled:
    return "cancelled";
  }
  return "unknown";
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parseOptions(argc, argv, options)) {
    llvm::errs()
        << "usage: wafer-qualify-optimizations --input-program-dir <dir> "
           "--output-program-dir <dir> --execution-ranks <1|16> "
           "--corpus-id <1|2> [--execution-ordinal <n>] "
           "[--input-variant <original|metamorphic>] "
           "[--configuration <all-on|all-off|disable-one:key>]\n";
    return 2;
  }
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  std::string diagnostic;
  std::optional<wafer::OptimizationConfiguration> configuration =
      parseConfiguration(proposal, options.configurationSpelling, diagnostic);
  if (!configuration) {
    llvm::errs() << "wafer-qualify-optimizations: " << diagnostic << "\n";
    return 2;
  }
  std::vector<wafer::RegistryRefV1> corpusRows =
      wafer::getCurrentMandatoryCorpusRowsV1();
  auto corpus = llvm::find_if(
      corpusRows, [&](const auto &row) { return row.id == options.corpusId; });
  if (corpus == corpusRows.end()) {
    llvm::errs() << "wafer-qualify-optimizations: unknown corpus id\n";
    return 2;
  }
  if (options.executionOrdinal >=
      std::numeric_limits<uint64_t>::max() /
          wafer::kOptimizationInvocationLocalOrdinalLimit) {
    llvm::errs() << "wafer-qualify-optimizations: execution ordinal is out of "
                    "range\n";
    return 2;
  }
  auto inputDigest = wafer::digestOptimizationArtifactDirectoryV1(
      options.inputProgramDirectory,
      "wafer.optimization-qualification-probe-input-v1");
  if (!inputDigest) {
    llvm::errs()
        << "wafer-qualify-optimizations: cannot digest input program\n";
    return 1;
  }
  wafer::AdoptionDigest scopeDigest = wafer::digestOptimizationBytesV1(
      "wafer.optimization-qualification-probe-scope-v1", *inputDigest);
  wafer::OptimizationInvocationScopeContextV1 scope;
  scope.scopeKind = wafer::InvocationScopeKindV1::QualificationRun;
  scope.scopeDigest = scopeDigest;
  scope.qualificationCase = wafer::QualificationCaseKeyV1{
      *corpus, options.rankCount, options.inputVariant};
  scope.invocationOrdinalBase = options.executionOrdinal *
                                wafer::kOptimizationInvocationLocalOrdinalLimit;
  auto journal = std::make_shared<wafer::OptimizationInvocationJournalV1>();
  wafer::ScopedOptimizationInvocationRecorder recorder(journal, scope);
  if (!recorder.installed()) {
    llvm::errs() << "wafer-qualify-optimizations: cannot install recorder\n";
    return 1;
  }
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      options.rankCount, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  if (!executionConfig) {
    llvm::errs() << "wafer-qualify-optimizations: "
                 << llvm::toString(executionConfig.takeError()) << "\n";
    return 2;
  }
  auto request = wafer::compiler::CompilationRequest::create(
      options.inputProgramDirectory, *executionConfig);
  auto toolchain = wafer::compiler::TargetToolchain::create(
      WAFER_PYTHON_EXECUTABLE, WAFER_DEVICE_LINKER_SCRIPT);
  if (!request || !toolchain) {
    if (!request)
      llvm::consumeError(request.takeError());
    if (!toolchain)
      llvm::consumeError(toolchain.takeError());
    llvm::errs() << "wafer-qualify-optimizations: invalid compiler inputs\n";
    return 2;
  }
  auto product = wafer::compiler::qualification_internal::
      compileProgramWithOptimizationConfiguration(
          std::move(*request), options.outputProgramDirectory,
          WAFER_XLA_SPMD_PARTITIONER_HELPER, *toolchain, proposal,
          *configuration, options.inputVariant, llvm::errs());
  if (mlir::failed(product))
    return 1;
  if (!journal->sealScope(scope.scopeKind, scope.scopeDigest, &diagnostic)) {
    llvm::errs() << "wafer-qualify-optimizations: cannot seal invocation "
                    "scope: "
                 << diagnostic << "\n";
    return 1;
  }
  std::vector<wafer::InvocationTelemetryV1> terminals =
      journal->committedTerminals();
  std::sort(terminals.begin(), terminals.end(),
            [](const auto &lhs, const auto &rhs) {
              return std::tie(lhs.identity.mechanismKey.semanticId,
                              lhs.identity.invocationOrdinal) <
                     std::tie(rhs.identity.mechanismKey.semanticId,
                              rhs.identity.invocationOrdinal);
            });
  for (const auto &terminal : terminals) {
    uint64_t work = terminal.workSummary.orderedCounters.empty()
                        ? 0
                        : terminal.workSummary.orderedCounters.front().value;
    llvm::outs() << "mechanism=" << terminal.identity.mechanismKey.semanticId
                 << " ordinal=" << terminal.identity.invocationOrdinal
                 << " outcome=" << outcomeName(terminal.outcome)
                 << " rewrites=" << terminal.rewriteCount
                 << " actions=" << terminal.backendActions.size()
                 << " work=" << work << "\n";
  }
  llvm::outs() << "static";
  for (const wafer::MetricEvidenceV1 &metric :
       product->staticMetrics.orderedMetrics) {
    llvm::outs() << " metric" << metric.metricId << "=";
    if (metric.value)
      llvm::outs() << *metric.value;
    else
      llvm::outs() << "unknown";
  }
  llvm::outs() << "\n";
  llvm::outs() << "wafer-qualify-optimizations: single execution passed with "
               << terminals.size() << " invocation terminals\n";
  return 0;
}
