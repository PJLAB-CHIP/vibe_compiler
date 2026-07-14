//===- wafer-cmodel-qualify-bulk.cpp - Offline bulk qualification -------===//

#include "Wafer/Target/BulkQualification.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>

namespace {

enum class Mode { Calibrate, Freeze, Validate };

struct Options {
  std::optional<Mode> mode;
  std::string spec;
  std::string calibration;
  std::string heldOutSpec;
  std::string policy;
  std::string output;
  std::optional<uint64_t> formalScalarBudget;
  std::optional<uint64_t> formalFMABudget;
  std::optional<uint64_t> maximumTotalBytes;
  std::optional<uint64_t> maximumScratchpadBytes;
  std::optional<uint64_t> maximumReorderBytes;
  std::optional<double> maximumAbsoluteError;
  std::optional<double> maximumRelativeError;
};

void printUsage() {
  llvm::outs()
      << "usage:\n"
         "  wafer-cmodel-qualify-bulk --mode calibrate --spec <canonical-json> "
         "--output <path> <budgets>\n"
         "  wafer-cmodel-qualify-bulk --mode freeze --calibration <path> "
         "--held-out-spec <canonical-json> --maximum-absolute-error <value> "
         "--maximum-relative-error <value> --output <path>\n"
         "  wafer-cmodel-qualify-bulk --mode validate --policy <path> "
         "--output <path> <budgets>\n"
         "budgets: --formal-scalar-budget <count> --formal-fma-budget <count> "
         "--maximum-total-bytes <bytes> --maximum-scratchpad-bytes <bytes> "
         "--maximum-reorder-bytes <bytes>\n";
}

llvm::Expected<uint64_t> parseUnsigned(llvm::StringRef text,
                                       llvm::StringRef option) {
  uint64_t value = 0;
  if (text.empty() || text.getAsInteger(10, value))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   option + " requires an unsigned integer");
  return value;
}

llvm::Expected<double> parseNonnegativeDouble(llvm::StringRef text,
                                              llvm::StringRef option) {
  std::string storage = text.str();
  errno = 0;
  char *end = nullptr;
  double value = std::strtod(storage.c_str(), &end);
  if (errno != 0 || end != storage.c_str() + storage.size() ||
      !std::isfinite(value) || value < 0.0)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   option +
                                       " requires a finite nonnegative number");
  return value;
}

llvm::Expected<Options> parseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      printUsage();
      std::exit(0);
    }
    auto value = [&]() -> llvm::Expected<llvm::StringRef> {
      if (index + 1 >= argc)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "missing value for " + argument);
      return llvm::StringRef(argv[++index]);
    };
    if (argument == "--mode") {
      llvm::Expected<llvm::StringRef> text = value();
      if (!text)
        return text.takeError();
      if (*text == "calibrate")
        options.mode = Mode::Calibrate;
      else if (*text == "freeze")
        options.mode = Mode::Freeze;
      else if (*text == "validate")
        options.mode = Mode::Validate;
      else
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "unknown --mode " + *text);
      continue;
    }
    auto setString = [&](std::string &slot) -> llvm::Error {
      if (!slot.empty())
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "duplicate option " + argument);
      llvm::Expected<llvm::StringRef> text = value();
      if (!text)
        return text.takeError();
      if (text->empty())
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       argument + " must not be empty");
      slot = text->str();
      return llvm::Error::success();
    };
    if (argument == "--spec") {
      if (llvm::Error error = setString(options.spec))
        return std::move(error);
      continue;
    }
    if (argument == "--calibration") {
      if (llvm::Error error = setString(options.calibration))
        return std::move(error);
      continue;
    }
    if (argument == "--held-out-spec") {
      if (llvm::Error error = setString(options.heldOutSpec))
        return std::move(error);
      continue;
    }
    if (argument == "--policy") {
      if (llvm::Error error = setString(options.policy))
        return std::move(error);
      continue;
    }
    if (argument == "--output") {
      if (llvm::Error error = setString(options.output))
        return std::move(error);
      continue;
    }
    auto setUnsigned = [&](std::optional<uint64_t> &slot) -> llvm::Error {
      if (slot)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "duplicate option " + argument);
      llvm::Expected<llvm::StringRef> text = value();
      if (!text)
        return text.takeError();
      llvm::Expected<uint64_t> parsed = parseUnsigned(*text, argument);
      if (!parsed)
        return parsed.takeError();
      slot = *parsed;
      return llvm::Error::success();
    };
    if (argument == "--formal-scalar-budget") {
      if (llvm::Error error = setUnsigned(options.formalScalarBudget))
        return std::move(error);
      continue;
    }
    if (argument == "--formal-fma-budget") {
      if (llvm::Error error = setUnsigned(options.formalFMABudget))
        return std::move(error);
      continue;
    }
    if (argument == "--maximum-total-bytes") {
      if (llvm::Error error = setUnsigned(options.maximumTotalBytes))
        return std::move(error);
      continue;
    }
    if (argument == "--maximum-scratchpad-bytes") {
      if (llvm::Error error = setUnsigned(options.maximumScratchpadBytes))
        return std::move(error);
      continue;
    }
    if (argument == "--maximum-reorder-bytes") {
      if (llvm::Error error = setUnsigned(options.maximumReorderBytes))
        return std::move(error);
      continue;
    }
    auto setDouble = [&](std::optional<double> &slot) -> llvm::Error {
      if (slot)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "duplicate option " + argument);
      llvm::Expected<llvm::StringRef> text = value();
      if (!text)
        return text.takeError();
      llvm::Expected<double> parsed = parseNonnegativeDouble(*text, argument);
      if (!parsed)
        return parsed.takeError();
      slot = *parsed;
      return llvm::Error::success();
    };
    if (argument == "--maximum-absolute-error") {
      if (llvm::Error error = setDouble(options.maximumAbsoluteError))
        return std::move(error);
      continue;
    }
    if (argument == "--maximum-relative-error") {
      if (llvm::Error error = setDouble(options.maximumRelativeError))
        return std::move(error);
      continue;
    }
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown option " + argument);
  }
  if (!options.mode)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "--mode is required");
  if (options.output.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "--output is required");
  return options;
}

llvm::Error requireBudgets(const Options &options) {
  if (!options.formalScalarBudget || !options.formalFMABudget ||
      !options.maximumTotalBytes || !options.maximumScratchpadBytes ||
      !options.maximumReorderBytes)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "calibrate/validate requires every formal and bulk budget");
  return llvm::Error::success();
}

int fail(llvm::Error error) {
  llvm::errs() << "wafer-cmodel-qualify-bulk: "
               << llvm::toString(std::move(error)) << '\n';
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::Expected<Options> options = parseOptions(argc, argv);
  if (!options)
    return fail(options.takeError());
  if (*options->mode == Mode::Freeze) {
    if (options->calibration.empty() || options->heldOutSpec.empty() ||
        !options->maximumAbsoluteError || !options->maximumRelativeError)
      return fail(llvm::createStringError(
          llvm::errc::invalid_argument,
          "freeze requires calibration, held-out spec and both tolerances"));
    if (llvm::Error error = wafer::freezeBulkBackendPolicy(
            options->calibration, options->heldOutSpec, options->output,
            {*options->maximumAbsoluteError, *options->maximumRelativeError}))
      return fail(std::move(error));
    llvm::outs() << "bulk qualification policy frozen: " << options->output
                 << '\n';
    return 0;
  }
  if (llvm::Error error = requireBudgets(*options))
    return fail(std::move(error));
  llvm::Expected<wafer::BulkExecutionEnvironment> environment =
      wafer::createManagedBulkExecutionEnvironment();
  if (!environment)
    return fail(environment.takeError());
  wafer::FormalNumericWorkBudget formal =
      wafer::FormalNumericWorkBudget::create(*options->formalScalarBudget,
                                             *options->formalFMABudget);
  wafer::BulkNumericWorkBudget bulk = wafer::BulkNumericWorkBudget::create(
      *options->maximumTotalBytes, *options->maximumScratchpadBytes,
      *options->maximumReorderBytes);
  if (*options->mode == Mode::Calibrate) {
    if (options->spec.empty())
      return fail(llvm::createStringError(llvm::errc::invalid_argument,
                                          "calibrate requires --spec"));
    if (llvm::Error error = wafer::calibrateBulkBackend(
            *environment, options->spec, options->output, formal, bulk))
      return fail(std::move(error));
    llvm::outs() << "bulk calibration published: " << options->output << '\n';
    return 0;
  }
  if (options->policy.empty())
    return fail(llvm::createStringError(llvm::errc::invalid_argument,
                                        "validate requires --policy"));
  if (llvm::Error error = wafer::validateBulkBackend(
          *environment, options->policy, options->output, formal, bulk))
    return fail(std::move(error));
  llvm::Expected<wafer::VerifiedBulkQualificationRecord> record =
      wafer::loadVerifiedBulkQualificationRecord(options->output);
  if (!record)
    return fail(record.takeError());
  llvm::outs() << "bulk qualification validated: " << options->output
               << " digest=" << record->getRecordDigest() << '\n';
  return 0;
}
