//===- DriverOptions.cpp - Wafer compiler driver options -----------------===//

#include "DriverInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>

namespace wafer::compile_driver {

void printHelp() {
  llvm::outs() << "usage: wafer-compile --input-program-dir <dir> "
                  "--output-dir <dir> --num-partitions <1> "
                  "[--optimization-policy <search|none>] "
                  "[--search-width <count>] [--search-trials <count>] "
                  "[--compile-timing] "
                  "[--dump-compiler-ir <dir>] "
                  "[--profile]\n";
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
  llvm::outs() << "test-only internal entry also accepts: "
                  "[--target-model "
                  "--model-input <index>=<npy> "
                  "--model-expected <index>=<npy> "
                  "[--model-atol <value>] [--model-rtol <value>] "
                  "[--model-report-numeric-statistics] "
                  "--target-model-max-scalar-evaluations <count> "
                  "--target-model-max-fused-multiply-adds <count> "
                  "--target-model-max-movement-bytes <bytes> "
                  "--target-model-max-movement-segments <count> "
                  "[--target-model-numeric-policy "
                  "<formal|onednn-then-formal|managed-reference> "
                  "[--target-model-onednn-record <record>] "
                  "--target-model-max-onednn-total-bytes <bytes> "
                  "--target-model-max-onednn-scratchpad-bytes <bytes> "
                  "--target-model-max-onednn-reorder-bytes <bytes>]]\n";
#endif
}

namespace {

bool setOption(std::optional<std::string> &slot, llvm::StringRef option,
               llvm::StringRef value) {
  if (slot) {
    llvm::errs() << "wafer-compile: duplicate option: " << option << "\n";
    return true;
  }
  slot = value.str();
  return false;
}

bool parseValueOption(int argc, char **argv, int &index, llvm::StringRef arg,
                      llvm::StringRef option,
                      std::optional<std::string> &slot) {
  if (arg == option) {
    if (index + 1 >= argc) {
      llvm::errs() << "wafer-compile: missing value for " << option << "\n";
      return true;
    }
    return setOption(slot, option, argv[++index]);
  }

  std::string prefix = (option + "=").str();
  if (arg.starts_with(prefix))
    return setOption(slot, option, arg.drop_front(prefix.size()));
  return false;
}

#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
bool parseRepeatedValueOption(int argc, char **argv, int &index,
                              llvm::StringRef arg, llvm::StringRef option,
                              std::vector<std::string> &values) {
  if (arg == option) {
    if (index + 1 >= argc) {
      llvm::errs() << "wafer-compile: missing value for " << option << "\n";
      return true;
    }
    values.emplace_back(argv[++index]);
    return false;
  }
  std::string prefix = (option + "=").str();
  if (arg.starts_with(prefix)) {
    values.push_back(arg.drop_front(prefix.size()).str());
    return false;
  }
  return false;
}
#endif

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

bool isDirectory(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/true) ==
         llvm::sys::fs::file_type::directory_file;
}

llvm::Error requireExecutableFile(llvm::StringRef path,
                                  llvm::StringRef factName) {
  if (path.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "no " + factName + " configured");
  llvm::SmallString<256> canonical;
  if (std::error_code error = llvm::sys::fs::real_path(path, canonical))
    return llvm::createStringError(error, factName + " cannot be resolved: '" +
                                              path + "'");
  if (!isRegularFile(canonical))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   factName + " is not a regular file: '" +
                                       path + "'");
  if (!llvm::sys::fs::can_execute(canonical))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   factName + " is not executable: '" + path +
                                       "'");
  return llvm::Error::success();
}

llvm::Expected<std::string>
findExecutableRelativeResource(llvm::StringRef relativePath,
                               llvm::StringRef factName) {
  static int executableAnchor = 0;
  std::string executable =
      llvm::sys::fs::getMainExecutable(/*argv0=*/nullptr, &executableAnchor);
  if (executable.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "failed to resolve the compiler "
                                   "executable");
  llvm::SmallString<256> resource(llvm::sys::path::parent_path(executable));
  llvm::sys::path::append(resource, relativePath);
  llvm::SmallString<256> canonical;
  if (std::error_code error = llvm::sys::fs::real_path(resource, canonical))
    return llvm::createStringError(error, "failed to resolve installed " +
                                              factName + ": '" +
                                              resource.str().str() + "'");
  return canonical.str().str();
}

} // namespace

bool parseCommandLine(int argc, char **argv, CommandLineOptions &options) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef arg(argv[index]);
    if (arg == "--help") {
      printHelp();
      return true;
    }
    if (arg == "--input-program-dir" ||
        arg.starts_with("--input-program-dir=")) {
      if (parseValueOption(argc, argv, index, arg, "--input-program-dir",
                           options.inputProgramDirectory))
        return false;
      continue;
    }
    if (arg == "--output-dir" || arg.starts_with("--output-dir=")) {
      if (parseValueOption(argc, argv, index, arg, "--output-dir",
                           options.outputDirectory))
        return false;
      continue;
    }
    if (arg == "--num-partitions" || arg.starts_with("--num-partitions=")) {
      if (parseValueOption(argc, argv, index, arg, "--num-partitions",
                           options.numPartitions))
        return false;
      continue;
    }
    if (arg == "--optimization-policy" ||
        arg.starts_with("--optimization-policy=")) {
      if (parseValueOption(argc, argv, index, arg, "--optimization-policy",
                           options.optimizationPolicy))
        return false;
      continue;
    }
    if (arg == "--search-width" || arg.starts_with("--search-width=")) {
      if (parseValueOption(argc, argv, index, arg, "--search-width",
                           options.searchWidth))
        return false;
      continue;
    }
    if (arg == "--search-trials" || arg.starts_with("--search-trials=")) {
      if (parseValueOption(argc, argv, index, arg, "--search-trials",
                           options.searchTrials))
        return false;
      continue;
    }
    if (arg == "--compile-timing") {
      if (options.compileTiming) {
        llvm::errs() << "wafer-compile: duplicate option: --compile-timing\n";
        return false;
      }
      options.compileTiming = true;
      continue;
    }
    if (arg == "--profile") {
      if (options.profile) {
        llvm::errs() << "wafer-compile: duplicate option: --profile\n";
        return false;
      }
      options.profile = true;
      continue;
    }
    if (arg == "--dump-compiler-ir" || arg.starts_with("--dump-compiler-ir=")) {
      if (parseValueOption(argc, argv, index, arg, "--dump-compiler-ir",
                           options.compilerIRDumpDirectory))
        return false;
      continue;
    }
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
    if (arg == "--target-model") {
      if (options.targetModel) {
        llvm::errs() << "wafer-compile: duplicate option: --target-model\n";
        return false;
      }
      options.targetModel = true;
      continue;
    }
    if (arg == "--model-report-numeric-statistics") {
      if (options.modelReportNumericStatistics) {
        llvm::errs() << "wafer-compile: duplicate option: "
                        "--model-report-numeric-statistics\n";
        return false;
      }
      options.modelReportNumericStatistics = true;
      continue;
    }
    if (arg == "--target-model-max-scalar-evaluations" ||
        arg.starts_with("--target-model-max-scalar-evaluations=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-scalar-evaluations",
                           options.targetModelMaximumScalarEvaluations))
        return false;
      continue;
    }
    if (arg == "--target-model-max-fused-multiply-adds" ||
        arg.starts_with("--target-model-max-fused-multiply-adds=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-fused-multiply-adds",
                           options.targetModelMaximumFusedMultiplyAdds))
        return false;
      continue;
    }
    if (arg == "--target-model-max-movement-bytes" ||
        arg.starts_with("--target-model-max-movement-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-movement-bytes",
                           options.targetModelMaximumMovementBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-movement-segments" ||
        arg.starts_with("--target-model-max-movement-segments=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-movement-segments",
                           options.targetModelMaximumMovementSegments))
        return false;
      continue;
    }
    if (arg == "--target-model-numeric-policy" ||
        arg.starts_with("--target-model-numeric-policy=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-numeric-policy",
                           options.targetModelNumericPolicy))
        return false;
      continue;
    }
    if (arg == "--target-model-onednn-record" ||
        arg.starts_with("--target-model-onednn-record=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg,
                                   "--target-model-onednn-record",
                                   options.targetModelOneDNNRecords))
        return false;
      continue;
    }
    if (arg == "--target-model-max-onednn-total-bytes" ||
        arg.starts_with("--target-model-max-onednn-total-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-onednn-total-bytes",
                           options.targetModelMaximumOneDNNTotalBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-onednn-scratchpad-bytes" ||
        arg.starts_with("--target-model-max-onednn-scratchpad-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-onednn-scratchpad-bytes",
                           options.targetModelMaximumOneDNNScratchpadBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-onednn-reorder-bytes" ||
        arg.starts_with("--target-model-max-onednn-reorder-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-onednn-reorder-bytes",
                           options.targetModelMaximumOneDNNReorderBytes))
        return false;
      continue;
    }
    if (arg == "--model-input" || arg.starts_with("--model-input=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg, "--model-input",
                                   options.modelInputs))
        return false;
      continue;
    }
    if (arg == "--model-expected" || arg.starts_with("--model-expected=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg, "--model-expected",
                                   options.modelExpected))
        return false;
      continue;
    }
    if (arg == "--model-atol" || arg.starts_with("--model-atol=")) {
      if (parseValueOption(argc, argv, index, arg, "--model-atol",
                           options.modelAtol))
        return false;
      continue;
    }
    if (arg == "--model-rtol" || arg.starts_with("--model-rtol=")) {
      if (parseValueOption(argc, argv, index, arg, "--model-rtol",
                           options.modelRtol))
        return false;
      continue;
    }
#endif

    llvm::errs() << "wafer-compile: unknown argument: " << arg << "\n";
    return false;
  }
  return true;
}

bool requireOption(const std::optional<std::string> &value,
                   llvm::StringRef option) {
  if (value)
    return true;
  llvm::errs() << "wafer-compile: missing required " << option << "\n";
  return false;
}

llvm::Expected<DriverToolFacts> resolveDriverToolFacts() {
  DriverToolFacts facts;
  bool helperOverridden = false;
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
  // The test seam treats an explicitly set variable as authoritative, so an
  // empty value exercises the no-helper failure instead of falling through.
  if (const char *environment =
          std::getenv("WAFER_TEST_XLA_SPMD_PARTITIONER_HELPER")) {
    facts.spmdPartitionerHelper = environment;
    helperOverridden = true;
  }
#endif
  if (!helperOverridden) {
    llvm::SmallString<256> helperRelative;
    llvm::sys::path::append(helperRelative, "..", "libexec", "wafer");
    llvm::sys::path::append(helperRelative, "wafer_xla_spmd_partitioner");
    llvm::Expected<std::string> helper = findExecutableRelativeResource(
        helperRelative, "XLA SPMD partitioner helper");
    if (!helper)
      return helper.takeError();
    facts.spmdPartitionerHelper = std::move(*helper);
  }
  if (facts.spmdPartitionerHelper.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "no XLA SPMD partitioner helper "
                                   "configured");
  if (!isRegularFile(facts.spmdPartitionerHelper))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "installed XLA SPMD partitioner helper is "
                                   "not a regular file: '" +
                                       facts.spmdPartitionerHelper + "'");
  llvm::SmallString<256> linkerRelative;
  llvm::sys::path::append(linkerRelative, "..", "share", "wafer");
  llvm::sys::path::append(linkerRelative, "wafer_device_link.py");
  llvm::Expected<std::string> linkerScript =
      findExecutableRelativeResource(linkerRelative, "device linker script");
  if (!linkerScript)
    return linkerScript.takeError();
  facts.deviceLinkerScript = std::move(*linkerScript);
  if (!isRegularFile(facts.deviceLinkerScript))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "installed device linker script is not a "
                                   "regular file: '" +
                                       facts.deviceLinkerScript + "'");

  // The pinned TX8 dependency root is an external dependency configured for
  // the deployment, never a source or build tree path inside the binary.
  if (const char *environment = std::getenv("TX8_DEPS_ROOT")) {
    facts.tx8DepsRoot = environment;
    if (!facts.tx8DepsRoot.empty()) {
      llvm::SmallString<256> canonical;
      if (std::error_code error =
              llvm::sys::fs::real_path(facts.tx8DepsRoot, canonical))
        return llvm::createStringError(error,
                                       "failed to resolve TX8_DEPS_ROOT: '" +
                                           facts.tx8DepsRoot + "'");
      facts.tx8DepsRoot = canonical.str().str();
    }
  }
  if (facts.tx8DepsRoot.empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "TX8_DEPS_ROOT environment variable must name the pinned TX8 "
        "dependency root");
  if (!isDirectory(facts.tx8DepsRoot))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "TX8_DEPS_ROOT is not a directory: '" +
                                       facts.tx8DepsRoot + "'");

  llvm::SmallString<256> waferIncludeRelative;
  llvm::sys::path::append(waferIncludeRelative, "..", "share", "wafer");
  llvm::sys::path::append(waferIncludeRelative, "include");
  llvm::Expected<std::string> waferInclude = findExecutableRelativeResource(
      waferIncludeRelative, "Wafer include directory");
  if (!waferInclude)
    return waferInclude.takeError();
  facts.waferIncludeDir = std::move(*waferInclude);
  if (!isDirectory(facts.waferIncludeDir))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "installed Wafer include resource is not "
                                   "a directory: '" +
                                       facts.waferIncludeDir + "'");
  llvm::SmallString<256> crtSourceRelative;
  llvm::sys::path::append(crtSourceRelative, "..", "share", "wafer");
  llvm::sys::path::append(crtSourceRelative, "crt", "src", "wafer_tx81_crt.c");
  llvm::Expected<std::string> crtSource =
      findExecutableRelativeResource(crtSourceRelative, "Wafer CRT source");
  if (!crtSource)
    return crtSource.takeError();
  facts.waferCrtSource = std::move(*crtSource);
  if (!isRegularFile(facts.waferCrtSource))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "installed Wafer CRT source is not a "
                                   "regular file: '" +
                                       facts.waferCrtSource + "'");
  llvm::SmallString<256> crtIncludeRelative;
  llvm::sys::path::append(crtIncludeRelative, "..", "share", "wafer");
  llvm::sys::path::append(crtIncludeRelative, "crt", "include");
  llvm::Expected<std::string> crtInclude = findExecutableRelativeResource(
      crtIncludeRelative, "Wafer CRT include directory");
  if (!crtInclude)
    return crtInclude.takeError();
  facts.waferCrtIncludeDir = std::move(*crtInclude);
  if (!isDirectory(facts.waferCrtIncludeDir))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "installed Wafer CRT include resource is "
                                   "not a directory: '" +
                                       facts.waferCrtIncludeDir + "'");

  llvm::ErrorOr<std::string> python = llvm::sys::findProgramByName("python3");
  if (!python)
    return llvm::createStringError(python.getError(),
                                   "failed to resolve python3 on PATH");
  facts.pythonExecutable = *python;
  llvm::ErrorOr<std::string> clangXX = llvm::sys::findProgramByName("clang++");
  if (!clangXX)
    return llvm::createStringError(clangXX.getError(),
                                   "failed to resolve clang++ on PATH");
  facts.llvmClangXX = *clangXX;

  if (llvm::Error error = requireExecutableFile(facts.spmdPartitionerHelper,
                                                "XLA SPMD partitioner helper"))
    return std::move(error);
  if (llvm::Error error =
          requireExecutableFile(facts.pythonExecutable, "python3"))
    return std::move(error);
  if (llvm::Error error = requireExecutableFile(facts.llvmClangXX, "clang++"))
    return std::move(error);
  // Store canonical paths: downstream consumers validate the exact file
  // facts and must never re-resolve through symlinks or PATH.
  llvm::SmallString<256> canonical;
  for (std::string *fact : {&facts.pythonExecutable, &facts.llvmClangXX}) {
    if (std::error_code error = llvm::sys::fs::real_path(*fact, canonical))
      return llvm::createStringError(
          error, "failed to resolve toolchain path: '" + *fact + "'");
    *fact = canonical.str().str();
  }
  return facts;
}

std::optional<std::vector<IndexedPath>>
parseIndexedPaths(llvm::ArrayRef<std::string> values, llvm::StringRef option) {
  std::vector<IndexedPath> parsed;
  for (const std::string &storage : values) {
    llvm::StringRef value(storage);
    auto [indexText, path] = value.split('=');
    int64_t index = -1;
    if (path.empty() || indexText.getAsInteger(10, index) || index < 0) {
      llvm::errs() << "wafer-compile: invalid " << option << " value: " << value
                   << "\n";
      return std::nullopt;
    }
    if (llvm::any_of(parsed, [&](const IndexedPath &item) {
          return item.index == index;
        })) {
      llvm::errs() << "wafer-compile: duplicate " << option
                   << " index: " << index << "\n";
      return std::nullopt;
    }
    parsed.push_back({index, path.str()});
  }
  return parsed;
}

std::optional<double> parseTolerance(const std::optional<std::string> &value,
                                     llvm::StringRef option,
                                     double defaultValue) {
  if (!value)
    return defaultValue;
  errno = 0;
  char *end = nullptr;
  double parsed = std::strtod(value->c_str(), &end);
  if (errno != 0 || end != value->c_str() + value->size() ||
      !std::isfinite(parsed) || parsed < 0.0) {
    llvm::errs() << "wafer-compile: invalid " << option << " value: " << *value
                 << "\n";
    return std::nullopt;
  }
  return parsed;
}

std::optional<OptimizationConfig>
parseOptimizationConfig(const CommandLineOptions &options) {
  OptimizationConfig config = OptimizationConfig::none();
  if (options.optimizationPolicy) {
    if (*options.optimizationPolicy == "search") {
      SearchLimits limits;
      if (options.searchWidth) {
        std::optional<uint64_t> width =
            parsePositiveCount(options.searchWidth, "--search-width");
        if (!width)
          return std::nullopt;
        limits.width = *width;
      }
      if (options.searchTrials) {
        std::optional<uint64_t> trials =
            parsePositiveCount(options.searchTrials, "--search-trials");
        if (!trials)
          return std::nullopt;
        limits.trials = *trials;
      }
      config = OptimizationConfig::search(limits);
    } else if (*options.optimizationPolicy == "none")
      config = OptimizationConfig::none();
    else {
      llvm::errs() << "wafer-compile: invalid --optimization-policy value: "
                   << *options.optimizationPolicy
                   << " (expected search or none)\n";
      return std::nullopt;
    }
  }

  if ((options.searchWidth || options.searchTrials) && !config.isSearch()) {
    llvm::StringRef option =
        options.searchWidth ? "--search-width" : "--search-trials";
    llvm::errs() << "wafer-compile: " << option
                 << " requires --optimization-policy=search\n";
    return std::nullopt;
  }

  return config;
}

std::optional<uint64_t>
parsePositiveCount(const std::optional<std::string> &value,
                   llvm::StringRef option) {
  if (!value) {
    llvm::errs() << "wafer-compile: target model requires " << option << "\n";
    return std::nullopt;
  }
  uint64_t parsed = 0;
  if (llvm::StringRef(*value).getAsInteger(10, parsed) || parsed == 0) {
    llvm::errs() << "wafer-compile: invalid " << option << " value: " << *value
                 << "\n";
    return std::nullopt;
  }
  return parsed;
}

} // namespace wafer::compile_driver
